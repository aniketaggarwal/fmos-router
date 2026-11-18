"""5-fold cross-validation over ALL labeled requests (not just the test split).

Why: the test split has only 14 T1, 8 T2 and 15 T3 requests. Cross-validation
gives every request one out-of-fold prediction, so routing metrics use all
71 T1, 43 T2 and 75 T3 requests.

Folds are stratified by type and keep each risky action with its twin.
Inside each fold, 25% of the training groups are held out to tune thresholds,
exactly as validation is used in evaluate.py.

Reports (results/cv_summary.md):
  1. R1 feature ablation: embedding only / hand features only / both
  2. Per-trigger routing for each variant, with counts
  3. Risk detection: AUC risky-vs-twin and results per risk category
  4. With --with-cascade: the agreement cascade and the hybrid, out of fold
     (draws k local samples for every non-action request that has none yet;
      about 1-1.5 hours on the first run, cached afterwards; no API calls)

Metrics here are routing decisions only (which operations were chosen).
End-to-end accuracy would need new strong-model calls for requests that
labeling never sent there, so it stays in evaluate.py's test-split results.

Run: python cv.py
     python cv.py --with-cascade --k 3
"""
import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import DATA, OPS, read_jsonl

RESULTS = Path("results")
NEEDED = {"T1": ["retrieve"], "T2": ["strong"], "T1+T2": ["retrieve", "strong"], "T3": ["approval"]}


def make_folds(items, k, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[it.get("pair_id") or it["id"]].append(it)
    by_cat = defaultdict(list)
    for key, members in groups.items():
        cat = "T3" if any(m["category"] == "T3" for m in members) else members[0]["category"]
        by_cat[cat].append(key)
    fold_of = {}
    for cat, keys in by_cat.items():
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            for m in groups[key]:
                fold_of[m["id"]] = i % k
    return fold_of


def split_train_val(items, seed, val_share=0.25):
    folds = make_folds(items, 4, seed)  # one of four parts = 25%
    return [it for it in items if folds[it["id"]] != 0], [it for it in items if folds[it["id"]] == 0]


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report(name, items, preds, scores=None):
    """preds: id -> set of ops. Returns markdown rows + a metrics dict."""
    m = {"name": name}
    for cat, need in NEEDED.items():
        its = [it for it in items if it["category"] == cat]
        m[cat] = (sum(all(o in preds[it["id"]] for o in need) for it in its), len(its))
    t1 = [it for it in items if it["category"] == "T1"]
    m["T1_to_strong"] = (sum("strong" in preds[it["id"]] for it in t1), len(t1))
    t0 = [it for it in items if it["category"] == "T0" and it["task_type"] != "action"]
    m["T0_escalated"] = (sum(bool(preds[it["id"]]) for it in t0), len(t0))
    twins = [it for it in items if it["task_type"] == "action" and not it["risky"]]
    m["twins_approved"] = (sum("approval" in preds[it["id"]] for it in twins), len(twins))
    m["exact"] = sum(preds[it["id"]] == set(it["correct_ops"]) for it in items) / len(items)
    if scores:
        acts = [it for it in items if it["task_type"] == "action"]
        m["auc"] = auc([scores[it["id"]]["approval"] for it in acts if it["risky"]],
                       [scores[it["id"]]["approval"] for it in acts if not it["risky"]])
    return m


def frac(x):
    a, b = x
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def r1_out_of_fold(items, folds, k, features, seed):
    from routers import R1Classifier
    preds, scores = {}, {}
    for f in range(k):
        train_all = [it for it in items if folds[it["id"]] != f]
        test = [it for it in items if folds[it["id"]] == f]
        tr, va = split_train_val(train_all, seed + f)
        r1 = R1Classifier(features=features, use_cache=True)
        r1.fit(tr)
        r1.tune(va)
        for it in test:
            d = r1.route(it)
            preds[it["id"]], scores[it["id"]] = set(d["ops"]), d["scores"]
        # in-sample decisions on the training part, used to tune the hybrid
        for it in train_all:
            scores.setdefault(("train", f, it["id"]), r1.route(it)["ops"])
    return preds, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--with-cascade", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    items = [it for it in read_jsonl(DATA / "labeled.jsonl") if it["category"] != "unsolvable"]
    folds = make_folds(items, args.folds, args.seed)
    print(f"{len(items)} requests, {args.folds} folds: {dict(Counter(it['category'] for it in items))}")

    results, risk_rows = [], []
    oof = {}
    for feat in ("both", "embed", "hand"):
        t0 = time.time()
        preds, scores = r1_out_of_fold(items, folds, args.folds, feat, args.seed)
        oof[feat] = (preds, scores)
        results.append(report(f"R1 ({feat})", items, preds, scores))
        print(f"R1 {feat:<6} done in {time.time() - t0:.0f}s")

    # Risk by category, R1 with both feature sets
    preds, scores = oof["both"]
    by_cat = defaultdict(lambda: {"pos": [], "neg": [], "caught": 0, "n": 0, "fa": 0, "nt": 0})
    for it in items:
        if it["task_type"] != "action":
            continue
        c = by_cat[it.get("risk_category") or "unknown"]
        s = scores[it["id"]]["approval"]
        if it["risky"]:
            c["pos"].append(s); c["n"] += 1; c["caught"] += "approval" in preds[it["id"]]
        else:
            c["neg"].append(s); c["nt"] += 1; c["fa"] += "approval" in preds[it["id"]]
    for cat, c in sorted(by_cat.items()):
        a = auc(c["pos"], c["neg"])
        risk_rows.append(f"| {cat} | {c['caught']}/{c['n']} | {c['fa']}/{c['nt']} | {a:.2f} |" if a is not None
                         else f"| {cat} | {c['caught']}/{c['n']} | {c['fa']}/{c['nt']} | n/a |")

    # Optional: cascade and hybrid out of fold
    if args.with_cascade:
        from evaluate import RunStore, agreement, cascade_samples
        from models import generate, warmup
        warmup()
        store = RunStore(generate)
        nonact = [it for it in items if it["task_type"] != "action"]
        agree = {}
        t0 = time.time()
        for n, it in enumerate(nonact, 1):
            agree[it["id"]] = agreement(cascade_samples(it, store, args.k, args.temp))[0]
            if n % 25 == 0:
                print(f"  cascade samples: {n}/{len(nonact)}  {time.time() - t0:.0f}s")

        def tune(rows):  # F1 on "needs strong", same rule as evaluate.py
            best = (-1.0, 0.0)
            for th in sorted({a for a, _ in rows} | {1.01}):
                tp = sum(a < th and need for a, need in rows)
                fp = sum(a < th and not need for a, need in rows)
                fn = sum(a >= th and need for a, need in rows)
                score = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                if score > best[0]:
                    best = (score, th)
            return best[1]

        r1_preds, r1_scores = oof["both"]
        casc, hyb, hyb_any = {}, {}, {}
        fold_th = []
        for f in range(args.folds):
            train = [it for it in nonact if folds[it["id"]] != f]
            th = tune([(agree[it["id"]], "strong" in it["correct_ops"]) for it in train])
            th_h = tune([(agree[it["id"]], "strong" in it["correct_ops"]) for it in train
                         if "retrieve" not in r1_scores[("train", f, it["id"])]])
            fold_th.append((f, th, th_h))
            for it in items:
                if folds[it["id"]] != f:
                    continue
                if it["task_type"] == "action":
                    casc[it["id"]] = set()
                    hyb[it["id"]] = r1_preds[it["id"]] & {"approval"}
                    hyb_any[it["id"]] = r1_preds[it["id"]] & {"approval"}
                    continue
                casc[it["id"]] = {"strong"} if agree[it["id"]] < th else set()
                if "retrieve" in r1_preds[it["id"]]:
                    hyb[it["id"]] = {"retrieve"}
                    hyb_any[it["id"]] = {"retrieve"}
                else:
                    hyb[it["id"]] = {"strong"} if agree[it["id"]] < th_h else set()
                    # sensitivity check: fixed rule, escalate on any disagreement (no tuning)
                    hyb_any[it["id"]] = {"strong"} if agree[it["id"]] < 0.999 else set()
        results.append(report(f"Cascade (k={args.k})", items, casc))
        results.append(report(f"Hybrid (k={args.k})", items, hyb))
        results.append(report(f"Hybrid, fixed rule: any disagreement (k={args.k})", items, hyb_any))

        # Diagnostic: why does the hybrid miss T2 requests the cascade catches?
        th_of = {f: (th, th_h) for f, th, th_h in fold_th}
        reasons = Counter()
        for it in items:
            if it["category"] not in ("T2", "T1+T2") or "strong" in hyb[it["id"]]:
                continue
            th, th_h = th_of[folds[it["id"]]]
            if "retrieve" in r1_preds[it["id"]]:
                reasons[f"R1 chose retrieval ({it['task_type']}), so no samples were drawn"] += 1
            elif agree[it["id"]] >= 0.999:
                reasons["all samples agreed (same wrong answer every time)"] += 1
            elif agree[it["id"]] >= th_h:
                reasons["samples partly disagreed, but not enough for the tuned threshold"] += 1
            else:
                reasons["other"] += 1
        diag_lines = ["", "## Hybrid diagnostic", "",
                      "Agreement thresholds per fold (escalate when agreement is below):", "",
                      "| Fold | Cascade | Hybrid |", "| --- | --- | --- |"]
        diag_lines += [f"| {f} | {th:.2f} | {th_h:.2f} |" for f, th, th_h in fold_th]
        diag_lines += ["", "T2 requests the hybrid did not send to the strong model, by reason:", ""]
        diag_lines += [f"- {k}: {v}" for k, v in reasons.most_common()] or ["- none"]
        caught_by_casc = sum(1 for it in items if it["category"] in ("T2", "T1+T2")
                             and "strong" in casc[it["id"]] and "strong" not in hyb[it["id"]])
        diag_lines.append(f"- Of these, caught by the cascade: {caught_by_casc}")
    else:
        diag_lines = []

    lines = ["# Cross-validated routing (all labeled requests)", "",
             f"{len(items)} requests in {args.folds} folds (stratified by type, risky actions kept with their twins). "
             "Every request gets one out-of-fold decision. Routing metrics only: see the module docstring.", "",
             "| Router | T1 to retrieval | T2 to strong model | T1+T2 both | T3 to approval | "
             "T1 sent to strong model | T0 answerable escalated | Twins approved | Ops exact match | Approval AUC |",
             "| --- |" + " --- |" * 9]
    for m in results:
        a = f"{m['auc']:.2f}" if m.get("auc") is not None else "n/a"
        lines.append(f"| {m['name']} | {frac(m['T1'])} | {frac(m['T2'])} | {frac(m['T1+T2'])} | {frac(m['T3'])} | "
                     f"{frac(m['T1_to_strong'])} | {frac(m['T0_escalated'])} | {frac(m['twins_approved'])} | "
                     f"{100 * m['exact']:.0f}% | {a} |")
    lines += ["", "R1 (both) = the main router; (embed) uses only the sentence embedding; (hand) only the eleven hand features.",
              "", "## Risk detection by category (R1, both feature sets, out of fold)", "",
              "| Risk category | Risky caught | Twins approved | AUC risky vs twin |", "| --- | --- | --- | --- |"] + risk_rows
    lines += diag_lines
    text = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cv_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
