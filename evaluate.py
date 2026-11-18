"""Evaluate routers and baselines on the TEST split (accuracy + latency).

Methods
  always_fast    never escalate
  always_slow    every applicable operation on every item
  random         escalates each operation at R1's own rate, at random
  oracle         the labels themselves (the best any router can do)
  R0             rules router (thresholds tuned on val)
  R1             embedding + features classifier (trained on train, tuned on val)
  cascade        post-generation: k local samples; low agreement -> Gemini
  hybrid         R1 for retrieve/approval, cascade for strong

Models are only called for (item, path) results that labeling did not
already produce; those go to data/eval_runs.jsonl and are reused next time.

Run: python evaluate.py                    (everything)
     python evaluate.py --skip-cascade     (no k-sample runs; much faster)
     python evaluate.py --cascade-k 3
Outputs: results/summary.md, results/metrics.json, results/predictions.jsonl,
         results/curve_r1.csv (+ .png if matplotlib is installed)
"""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, OPS, append_jsonl, path_applies, read_jsonl, run_path, write_jsonl

RESULTS = Path("results")
EVAL_RUNS = DATA / "eval_runs.jsonl"
NEEDED_OP = {"T1": ["retrieve"], "T2": ["strong"], "T1+T2": ["retrieve", "strong"], "T3": ["approval"]}


# ----------------------------------------------------------- run store ---
class RunStore:
    """Results per (item, path, sample), from labeling first, then eval runs."""

    def __init__(self, generate, allow_run=True):
        self.generate, self.allow_run = generate, allow_run
        self.runs = {}
        for f in (DATA / "label_runs.jsonl", EVAL_RUNS):
            for r in read_jsonl(f):
                self.runs[(r["item_id"], r["path"], r["sample"], r.get("variant", "label"))] = r
        self.new_calls = Counter()

    def get(self, item, path, sample=0, variant="label", temperature=None, seed=None):
        key = (item["id"], path, sample, variant)
        if key not in self.runs:
            if not self.allow_run:
                return None
            r = run_path(item, path, self.generate, tag_prefix=f"eval:{variant}:", sample=sample,
                         temperature=temperature, seed=seed)
            r.update({"item_id": item["id"], "variant": variant})
            append_jsonl(EVAL_RUNS, r)
            self.runs[key] = r
            self.new_calls["gemini" if "strong" in path else "local"] += 1
        return self.runs[key]


def exec_path(item, ops):
    if item["task_type"] == "action":
        return None
    r = "retrieve" in ops and path_applies(item, "retrieve")
    s = "strong" in ops
    return {(False, False): "fast", (True, False): "retrieve",
            (False, True): "strong", (True, True): "retrieve+strong"}[(r, s)]


def effective_ops(item, ops):
    """Drop operations that do not apply to this item (e.g. retrieve on math)."""
    out = set(ops)
    if item["task_type"] == "action":
        return out & {"approval"}
    out.discard("approval")
    if "retrieve" in out and not path_applies(item, "retrieve"):
        out.discard("retrieve")
    return out


def path_label(item, ops):
    if item["task_type"] == "action":
        return "approval" if "approval" in ops else "fast"
    return exec_path(item, ops)


def true_label(item):
    if item["task_type"] == "action":
        return "approval" if item["risky"] else "fast"
    return item["label_path"]

# ------------------------------------------------------------- cascade ---
def cascade_samples(item, store, k, temp):
    return [store.get(item, "fast", sample=s, variant=f"cascade_t{temp}", temperature=temp, seed=2000 + s)
            for s in range(k)]


def agreement(samples):
    keys = [s["answer_key"] for s in samples]
    top_key, top_n = Counter(keys).most_common(1)[0]
    majority = next(s for s in samples if s["answer_key"] == top_key)
    return top_n / len(samples), majority


def tune_cascade(val_items, store, k, temp, skip_ids=(), objective="f1"):
    """Agreement threshold for escalating to the strong model, tuned on val.
    objective="f1": best F1 for 'needs strong' (the same rule R1 uses);
    objective="acc": best accuracy of the escalate decision (our first version,
    which under class imbalance never escalates; kept for comparison).
    skip_ids: items handled by another component (the hybrid leaves out items
    R1 sends to retrieval)."""
    rows = []
    for it in val_items:
        if it["task_type"] == "action" or it["id"] in skip_ids:
            continue
        a, _ = agreement(cascade_samples(it, store, k, temp))
        rows.append((a, "strong" in it["correct_ops"]))
    best = (-1.0, 0.0)
    for th in sorted({a for a, _ in rows} | {1.01}):
        if objective == "acc":
            score = sum((a < th) == need for a, need in rows) / len(rows)
        else:
            tp = sum(a < th and need for a, need in rows)
            fp = sum(a < th and not need for a, need in rows)
            fn = sum(a >= th and need for a, need in rows)
            score = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if score > best[0]:
            best = (score, th)
    return best[1]

# ------------------------------------------------------------- metrics ---
def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def evaluate_method(name, items, decisions, store):
    """decisions: item_id -> {"ops": [...], "router_ms": float,
    "answer": optional run dict used instead of the exec path (cascade)}"""
    rows = []
    for it in items:
        d = decisions[it["id"]]
        pred = effective_ops(it, d["ops"])
        gold = set(it["correct_ops"])
        latency = d["router_ms"]
        if it["task_type"] == "action":
            correct = ("approval" in pred) == it["risky"]
        else:
            p = exec_path(it, pred)
            run = d.get("answer") if (p == "fast" and d.get("answer")) else store.get(it, p)
            correct = bool(run and run["correct"])
            if run and not (p == "fast" and d.get("answer")):
                latency += run["wall_ms"]
        rows.append({"id": it["id"], "category": it["category"], "pred_ops": sorted(pred),
                     "gold_ops": sorted(gold), "pred_path": path_label(it, pred), "true_path": true_label(it),
                     "e2e_correct": correct, "latency_ms": round(latency, 1),
                     "router_ms": round(d["router_ms"], 3), "method": name,
                     "source": it.get("source"),
                     "uses_gemini": it["task_type"] != "action" and "strong" in pred,
                     "scores": d.get("scores")})

    m = {"method": name, "n": len(rows)}
    m["e2e_accuracy"] = sum(r["e2e_correct"] for r in rows) / len(rows)
    m["ops_exact_match"] = sum(r["pred_ops"] == r["gold_ops"] for r in rows) / len(rows)
    m["path_accuracy"] = sum(bool(r["pred_ops"]) == bool(r["gold_ops"]) for r in rows) / len(rows)
    m["escalation_rate"] = sum(bool(r["pred_ops"]) for r in rows) / len(rows)
    m["gemini_rate"] = sum(r["uses_gemini"] for r in rows) / len(rows)  # cost proxy

    for op in OPS:
        tp = sum(op in r["pred_ops"] and op in r["gold_ops"] for r in rows)
        fp = sum(op in r["pred_ops"] and op not in r["gold_ops"] for r in rows)
        fn = sum(op not in r["pred_ops"] and op in r["gold_ops"] for r in rows)
        m[f"{op}_precision"] = tp / (tp + fp) if tp + fp else None
        m[f"{op}_recall"] = tp / (tp + fn) if tp + fn else None
        m[f"{op}_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    for cat, needed in NEEDED_OP.items():
        rs = [r for r in rows if r["category"] == cat]
        m[f"{cat}_recall"] = (sum(all(o in r["pred_ops"] for o in needed) for r in rs) / len(rs)) if rs else None
    t0 = [r for r in rows if r["category"] == "T0"]
    m["T0_false_escalation"] = sum(bool(r["pred_ops"]) for r in t0) / len(t0) if t0 else None
    t3 = [r for r in rows if r["category"] == "T3"]
    m["stakes_miss_rate"] = sum("approval" not in r["pred_ops"] for r in t3) / len(t3) if t3 else None
    twins = [r for r in rows if r["id"].startswith("tw-")]
    m["twin_false_approval"] = sum("approval" in r["pred_ops"] for r in twins) / len(twins) if twins else None
    # Risky-action results by who wrote the items (e.g. claude-written vs teammate)
    by_src = defaultdict(list)
    for r in rows:
        if r["category"] == "T3":
            by_src[r["source"]].append("approval" not in r["pred_ops"])
    m["stakes_miss_by_source"] = {s: sum(v) / len(v) for s, v in by_src.items()}

    lat = [r["latency_ms"] for r in rows]
    rt = [r["router_ms"] for r in rows]
    m.update({"latency_mean_ms": statistics.mean(lat), "latency_p50_ms": pct(lat, .5),
              "latency_p95_ms": pct(lat, .95), "router_p50_ms": pct(rt, .5), "router_p95_ms": pct(rt, .95)})
    m["confusion"] = dict(Counter(f"{r['true_path']} -> {r['pred_path']}" for r in rows))
    return m, rows

# ------------------------------------------------------------ reporting ---
def fmt(v, pctg=True):
    if v is None:
        return "n/a"
    return f"{100 * v:.0f}%" if pctg else f"{v:,.0f}"


def write_summary(metrics, thresholds, n_excluded, split_counts, extra_lines=()):
    RESULTS.mkdir(exist_ok=True)
    lines = ["# Results (test split)", "",
             f"Test items by category: {split_counts}. Unsolvable items excluded: {n_excluded}.", "",
             "## Accuracy and latency", "",
             "| Method | End-to-end acc | Ops exact match | Path acc | T1 recall | T2 recall | T3 recall | "
             "Stakes miss | Twin false approval | T0 false escalation | Escalation rate | Sent to Gemini | "
             "Mean latency (ms) | p95 latency (ms) | Router p50 (ms) |",
             "| --- |" + " --- |" * 14]
    for m in metrics:
        lines.append(
            f"| {m['method']} | {fmt(m['e2e_accuracy'])} | {fmt(m['ops_exact_match'])} | {fmt(m['path_accuracy'])} | "
            f"{fmt(m['T1_recall'])} | {fmt(m['T2_recall'])} | {fmt(m['T3_recall'])} | {fmt(m['stakes_miss_rate'])} | "
            f"{fmt(m['twin_false_approval'])} | {fmt(m['T0_false_escalation'])} | {fmt(m['escalation_rate'])} | "
            f"{fmt(m['gemini_rate'])} | "
            f"{fmt(m['latency_mean_ms'], False)} | {fmt(m['latency_p95_ms'], False)} | "
            f"{m['router_p50_ms']:.2f} |")
    lines += ["", "## Per-operation precision / recall / F1", "",
              "| Method | " + " | ".join(f"{op} P | {op} R | {op} F1" for op in OPS) + " |",
              "| --- |" + " --- |" * (3 * len(OPS))]
    for m in metrics:
        cells = []
        for op in OPS:
            cells += [fmt(m[f"{op}_precision"]), fmt(m[f"{op}_recall"]), fmt(m[f"{op}_f1"])]
        lines.append(f"| {m['method']} | " + " | ".join(cells) + " |")
    lines += ["", "## Confusion (true path -> chosen path)", ""]
    for m in metrics:
        if m["method"] in ("R0", "R1", "cascade_acc", "cascade", "hybrid_v1", "hybrid"):
            lines.append(f"**{m['method']}**: " + ", ".join(f"{k}: {v}" for k, v in sorted(m["confusion"].items())))
            lines.append("")
    lines += ["## Stakes-miss rate by who wrote the T3 items", "",
              "| Method | " + " | ".join(sorted({s for m in metrics for s in m["stakes_miss_by_source"]})) + " |",
              "| --- |" + " --- |" * len({s for m in metrics for s in m["stakes_miss_by_source"]})]
    srcs = sorted({s for m in metrics for s in m["stakes_miss_by_source"]})
    for m in metrics:
        lines.append(f"| {m['method']} | " + " | ".join(fmt(m["stakes_miss_by_source"].get(s)) for s in srcs) + " |")
    lines += ["", "## Tuned thresholds (on validation)", "", "```", json.dumps(thresholds, indent=2), "```", "",
              "Latency = router time + time of the chosen path. Action items count router time only. "
              "Cascade router time is the time of its k local samples, run sequentially."]
    lines += list(extra_lines)
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def auc(pos, neg):
    """Probability a random positive scores above a random negative (ties = 0.5)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def diagnostics(test, r1_scores, store, k, temp, have_cascade):
    """Extra analyses from saved results only: no new model calls."""
    lines = ["", "## Diagnostics", ""]

    # 1. Does R1's approval score separate risky actions from their twins?
    acts = [it for it in test if it["task_type"] == "action"]
    if acts:
        pos = [r1_scores[it["id"]]["approval"] for it in acts if it["risky"]]
        neg = [r1_scores[it["id"]]["approval"] for it in acts if not it["risky"]]
        a = auc(pos, neg)
        lines += [f"**R1 approval: risky vs benign twin.** AUC = {a:.2f} "
                  f"(0.5 = cannot tell them apart, 1.0 = perfect), n = {len(pos)} risky, {len(neg)} twins.", "",
                  "| Approval threshold | Risky caught | Twins wrongly approved |", "| --- | --- | --- |"]
        for t in (0.3, 0.5, 0.65, 0.8, 0.9, 0.95):
            lines.append(f"| {t:.2f} | {sum(p >= t for p in pos)}/{len(pos)} | {sum(n >= t for n in neg)}/{len(neg)} |")
        lines.append("")

    # 2. Cascade: how often do the k samples agree, by category?
    if have_cascade:
        rows = defaultdict(list)
        for it in test:
            if it["task_type"] == "action":
                continue
            samples = [store.runs.get((it["id"], "fast", s, f"cascade_t{temp}")) for s in range(k)]
            if all(samples):
                rows[it["category"]].append(agreement(samples)[0])
        lines += [f"**Cascade agreement by category** ({k} samples at temperature {temp}).", "",
                  "| Category | n | All samples agree | Mean agreement |", "| --- | --- | --- | --- |"]
        for cat in sorted(rows):
            v = rows[cat]
            lines.append(f"| {cat} | {len(v)} | {sum(x >= 0.999 for x in v)}/{len(v)} | {statistics.mean(v):.2f} |")
        # What if the cascade escalated on ANY disagreement?
        t2 = rows.get("T2", []) + rows.get("T1+T2", [])
        others = [x for c, v in rows.items() if c not in ("T2", "T1+T2") for x in v]
        if t2:
            caught = sum(x < 0.999 for x in t2)
            false = sum(x < 0.999 for x in others)
            lines += ["", f"If the cascade escalated on any disagreement: {caught}/{len(t2)} T2 items caught, "
                          f"{false}/{len(others)} other items escalated needlessly."]
        lines.append("")

    # 3. Latency by backend and task type (labeling runs, cached excluded)
    lat = defaultdict(list)
    ids = {it["id"]: it for it in test}
    for (iid, path, sample, variant), r in store.runs.items():
        if iid in ids and variant == "label" and not r.get("cached") and not r.get("cold") and path in ("fast", "strong"):
            lat[(ids[iid]["task_type"], path)].append(r["wall_ms"])
    if lat:
        lines += ["**Latency by path and task type** (test items, single uncached calls).", "",
                  "| Task type | Path | n | Median (ms) | Mean (ms) |", "| --- | --- | --- | --- | --- |"]
        for (tt, path), v in sorted(lat.items()):
            lines.append(f"| {tt} | {'local 9B' if path == 'fast' else 'Gemini'} | {len(v)} | "
                         f"{statistics.median(v):,.0f} | {statistics.mean(v):,.0f} |")
        lines.append("")
    return lines


def sweep_r1(r1, test_items, scores, store):
    """Accuracy vs latency as the retrieve/strong thresholds move together."""
    import numpy as np
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        th = dict(r1.thresholds, retrieve=float(t), strong=float(t))
        dec = {}
        for it in test_items:
            s = scores[it["id"]]
            dec[it["id"]] = {"ops": [op for op in OPS if s[op] >= th[op]], "router_ms": 0.0}
        m, _ = evaluate_method(f"R1@{t:.2f}", test_items, dec, store)
        rows.append((round(float(t), 2), m["e2e_accuracy"], m["latency_mean_ms"], m["escalation_rate"]))
    with (RESULTS / "curve_r1.csv").open("w", encoding="utf-8") as f:
        f.write("threshold,e2e_accuracy,mean_latency_ms,escalation_rate\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r[2] for r in rows], [100 * r[1] for r in rows], marker="o")
        ax.set_xlabel("Mean latency per request (ms)")
        ax.set_ylabel("End-to-end accuracy (%)")
        ax.set_title("R1: accuracy vs latency as escalation thresholds vary")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "curve_r1.png", dpi=150)
    except ImportError:
        pass

# ----------------------------------------------------------------- main ---
def main(generate=None, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade-k", type=int, default=5)
    ap.add_argument("--cascade-temp", type=float, default=0.7)
    ap.add_argument("--skip-cascade", action="store_true")
    ap.add_argument("--skip-always-slow", action="store_true",
                    help="skip the always-slow baseline (its Gemini runs are the main API cost of evaluation)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results", help="folder for the result files")
    args = ap.parse_args(argv)
    global RESULTS
    RESULTS = Path(args.out)

    if generate is None:
        from models import generate as real_generate, warmup
        warmup()
        generate = real_generate
    store = RunStore(generate)

    from routers import R0Rules, R1Classifier

    labeled = read_jsonl(DATA / "labeled.jsonl")
    items = [it for it in labeled if it["category"] != "unsolvable"]
    n_excluded = len(labeled) - len(items)
    split = {s: [it for it in items if it["split"] == s] for s in ("train", "val", "test")}
    test = split["test"]
    print({s: len(v) for s, v in split.items()}, f"(excluded unsolvable: {n_excluded})")

    thresholds, decisions = {}, {}

    # Baselines that need no training
    decisions["always_fast"] = {it["id"]: {"ops": [], "router_ms": 0.0} for it in test}
    if not args.skip_always_slow:
        decisions["always_slow"] = {it["id"]: {"ops": list(OPS), "router_ms": 0.0} for it in test}
    decisions["oracle"] = {it["id"]: {"ops": it["correct_ops"], "router_ms": 0.0} for it in test}

    # R0
    r0 = R0Rules()
    thresholds["R0"] = r0.tune(split["val"])
    decisions["R0"] = {it["id"]: r0.route(it) for it in test}

    # R1
    print("Training R1 ...")
    r1 = R1Classifier()
    r1.fit(split["train"])
    thresholds["R1"] = r1.tune(split["val"])
    decisions["R1"] = {it["id"]: r1.route(it) for it in test}
    r1_scores = {it["id"]: decisions["R1"][it["id"]]["scores"] for it in test}

    # Random at R1's own per-operation rates
    rng = random.Random(args.seed)
    rate = {op: sum(op in d["ops"] for d in decisions["R1"].values()) / len(test) for op in OPS}
    decisions["random"] = {it["id"]: {"ops": [op for op in OPS if rng.random() < rate[op]], "router_ms": 0.0}
                           for it in test}

    # Cascade and hybrid (post-generation)
    if not args.skip_cascade:
        print(f"Cascade: {args.cascade_k} local samples per item on val + test ...")
        th = tune_cascade(split["val"], store, args.cascade_k, args.cascade_temp)
        th_acc = tune_cascade(split["val"], store, args.cascade_k, args.cascade_temp, objective="acc")
        thresholds["cascade"] = {"agreement_below": th, "objective": "F1", "k": args.cascade_k,
                                 "temperature": args.cascade_temp}
        thresholds["cascade_acc"] = {"agreement_below": th_acc, "objective": "accuracy of escalate decision"}
        # Hybrid: R1 decides retrieve/approval; the cascade threshold is tuned only on the
        # validation items R1 does NOT send to retrieval (the ones the cascade will see).
        r1_val_retrieve = {it["id"] for it in split["val"] if "retrieve" in r1.route(it)["ops"]}
        th_h = tune_cascade(split["val"], store, args.cascade_k, args.cascade_temp, skip_ids=r1_val_retrieve)
        thresholds["hybrid"] = {"agreement_below": th_h, "tuned_on": "val items not routed to retrieve by R1"}
        decisions["cascade"], decisions["cascade_acc"] = {}, {}
        decisions["hybrid_v1"], decisions["hybrid"] = {}, {}
        for it in test:
            r1d = decisions["R1"][it["id"]]
            if it["task_type"] == "action":
                decisions["cascade"][it["id"]] = {"ops": [], "router_ms": 0.0}
                decisions["cascade_acc"][it["id"]] = {"ops": [], "router_ms": 0.0}
                approve = {"ops": [o for o in r1d["ops"] if o == "approval"], "router_ms": r1d["router_ms"]}
                decisions["hybrid_v1"][it["id"]] = dict(approve)
                decisions["hybrid"][it["id"]] = dict(approve)
                continue
            samples = cascade_samples(it, store, args.cascade_k, args.cascade_temp)
            a, majority = agreement(samples)
            sample_ms = sum(s["wall_ms"] for s in samples)
            decisions["cascade"][it["id"]] = {"ops": ["strong"] if a < th else [], "router_ms": sample_ms,
                                              "answer": majority}
            decisions["cascade_acc"][it["id"]] = {"ops": ["strong"] if a < th_acc else [],
                                                  "router_ms": sample_ms, "answer": majority}
            # v1: R1 retrieval plus the plain cascade threshold (tuned on all val items, incl. T1).
            ops_v1 = [o for o in r1d["ops"] if o == "retrieve"] + (["strong"] if a < th else [])
            decisions["hybrid_v1"][it["id"]] = {"ops": ops_v1, "router_ms": r1d["router_ms"] + sample_ms,
                                                "answer": majority}
            # hybrid: if R1 retrieves, retrieval is the remedy (no samples needed);
            # otherwise escalate to the strong model when samples disagree.
            if "retrieve" in r1d["ops"]:
                decisions["hybrid"][it["id"]] = {"ops": ["retrieve"], "router_ms": r1d["router_ms"]}
            else:
                decisions["hybrid"][it["id"]] = {"ops": ["strong"] if a < th_h else [],
                                                 "router_ms": r1d["router_ms"] + sample_ms, "answer": majority}

    order = ["always_fast", "always_slow", "random", "R0", "R1", "cascade_acc", "cascade", "hybrid_v1", "hybrid", "oracle"]
    metrics, all_rows = [], []
    for name in order:
        if name in decisions:
            m, rows = evaluate_method(name, test, decisions[name], store)
            metrics.append(m)
            all_rows += rows
            print(f"{name:<12} e2e={100 * m['e2e_accuracy']:5.1f}%  exact={100 * m['ops_exact_match']:5.1f}%  "
                  f"stakes_miss={fmt(m['stakes_miss_rate'])}  mean_latency={m['latency_mean_ms']:8.0f} ms")

    RESULTS.mkdir(exist_ok=True)
    write_jsonl(RESULTS / "predictions.jsonl", all_rows)
    (RESULTS / "metrics.json").write_text(json.dumps({"metrics": metrics, "thresholds": thresholds},
                                                     indent=2, default=str), encoding="utf-8")
    diag = diagnostics(test, r1_scores, store, args.cascade_k, args.cascade_temp, not args.skip_cascade)
    write_summary(metrics, thresholds, n_excluded, dict(Counter(it["category"] for it in test)), diag)
    sweep_r1(r1, test, r1_scores, store)
    print(f"\nNew model calls during evaluation: {dict(store.new_calls)}")
    print(f"Wrote {RESULTS}/summary.md, metrics.json, predictions.jsonl, curve_r1.csv")


if __name__ == "__main__":
    main()
