"""Out-of-distribution test for risk detection (T3).

The routers are trained and tuned exactly as in evaluate.py (train/val splits
of data/labeled.jsonl, whose T3 items were all written by one author). They
are then scored on a second T3 set written independently by a different
author (data/external_t3_teammate.jsonl), which was never used for training,
tuning or design. No model calls are made, so this costs no API tokens.

Usage:  python t3_external.py [--file data/external_t3_teammate.jsonl]
Output: results/t3_external.md
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

from build_dataset import action_items
from common import DATA, read_jsonl

RESULTS = Path("results")
# Words that mark a request as a test / personal environment. Used only to
# check whether the router leans on these surface cues.
SANDBOX_CUE = re.compile(r"\b(my|local|localhost|test|sandbox|dev|mock|dummy|staging|demo)\b", re.I)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(name, router, items, threshold):
    rows = []
    for it in items:
        s = router.scores(it)["approval"]
        rows.append((it, s, s >= threshold))
    risky = [r for r in rows if r[0]["risky"]]
    twins = [r for r in rows if not r[0]["risky"]]
    res = {
        "name": name,
        "threshold": threshold,
        "caught": sum(r[2] for r in risky), "n_risky": len(risky),
        "twins_flagged": sum(r[2] for r in twins), "n_twins": len(twins),
        "auc": auc([r[1] for r in risky], [r[1] for r in twins]),
        "by_cat": {},
        "rows": rows,
    }
    by = defaultdict(list)
    for r in rows:
        by[r[0].get("risk_category") or "unknown"].append(r)
    for cat, rs in sorted(by.items()):
        rk = [r for r in rs if r[0]["risky"]]
        tw = [r for r in rs if not r[0]["risky"]]
        res["by_cat"][cat] = (sum(r[2] for r in rk), len(rk), sum(r[2] for r in tw), len(tw),
                              auc([r[1] for r in rk], [r[1] for r in tw]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DATA / "external_t3_teammate.jsonl"))
    args = ap.parse_args()

    from routers import R0Rules, R1Classifier

    labeled = [it for it in read_jsonl(DATA / "labeled.jsonl") if it["category"] != "unsolvable"]
    split = {s: [it for it in labeled if it["split"] == s] for s in ("train", "val", "test")}
    external = action_items([args.file])
    if not external:
        raise SystemExit(f"No items in {args.file}")
    for it in external:  # T3 labels are fixed by construction: risky -> approval
        it["correct_ops"] = ["approval"] if it["risky"] else []
        it["category"] = "T3" if it["risky"] else "T0"
    seen = {it["request"] for it in labeled}
    overlap = sum(it["request"] in seen for it in external)
    print(f"train {len(split['train'])}, val {len(split['val'])}; external {len(external)} "
          f"(identical requests already in the labeled data: {overlap})")

    results = []
    r0 = R0Rules()
    r0.tune(split["val"])
    results.append(evaluate("R0 (rules)", r0, external, 1.0))
    in_dist = {}
    for feats in ("both", "embed", "hand"):
        r1 = R1Classifier(features=feats, use_cache=True)
        r1.fit(split["train"])
        th = r1.tune(split["val"])["approval"]
        label = {"both": "R1 (both)", "embed": "R1 (embed only)", "hand": "R1 (hand only)"}[feats]
        results.append(evaluate(label, r1, external, th))
        # reference: the same router on the original author's test-split T3 items
        own = [it for it in split["test"] if it.get("tool_call")]
        in_dist[label] = evaluate(label, r1, own, th)
        if feats == "both":
            main_r1 = r1

    # threshold sweep for the main router
    sweep = []
    for t in (0.30, 0.50, 0.65, 0.80, 0.90, 0.95):
        e = evaluate("R1", main_r1, external, t)
        sweep.append((t, e["caught"], e["twins_flagged"]))

    # surface-cue check: are twins with sandbox words flagged less?
    main = next(r for r in results if r["name"] == "R1 (both)")
    tw = [r for r in main["rows"] if not r[0]["risky"]]
    cue = [r for r in tw if SANDBOX_CUE.search(r[0]["request"])]
    nocue = [r for r in tw if not SANDBOX_CUE.search(r[0]["request"])]

    RESULTS.mkdir(exist_ok=True)
    n = main["n_risky"]
    L = ["# Risk detection on an independently written T3 set", "",
         f"Routers trained and tuned on the original data (train {len(split['train'])}, "
         f"val {len(split['val'])}), then applied unchanged to `{Path(args.file).name}`: "
         f"{main['n_risky']} risky requests and {main['n_twins']} benign twins. "
         f"Requests identical to labeled data: {overlap}.", "",
         "## Main result", "",
         "| Router | Threshold | Risky caught | Twins wrongly flagged | AUC | Same router on original test split: caught / twins flagged / AUC |",
         "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        ref = in_dist.get(r["name"])
        ref_s = (f"{ref['caught']}/{ref['n_risky']} / {ref['twins_flagged']}/{ref['n_twins']} / {ref['auc']:.2f}"
                 if ref else "n/a")
        L.append(f"| {r['name']} | {r['threshold']:.2f} | {r['caught']}/{r['n_risky']} | "
                 f"{r['twins_flagged']}/{r['n_twins']} | {r['auc']:.2f} | {ref_s} |")
    L += ["", "## By risk category (R1, both feature sets)", "",
          "| Category | Risky caught | Twins wrongly flagged | AUC |", "| --- | --- | --- | --- |"]
    for cat, (c, nr, f, nt, a) in main["by_cat"].items():
        L.append(f"| {cat} | {c}/{nr} | {f}/{nt} | {a:.2f} |")
    L += ["", "## Approval threshold sweep (R1, both)", "",
          "| Threshold | Risky caught | Twins wrongly flagged |", "| --- | --- | --- |"]
    for t, c, f in sweep:
        L.append(f"| {t:.2f} | {c}/{n} | {f}/{n} |")
    L += ["", "## Surface-cue check", "",
          "Twins whose request contains a sandbox word (my, local, test, sandbox, dev, mock, dummy, staging, demo):",
          f"- with such a word: {sum(r[2] for r in cue)}/{len(cue)} wrongly flagged",
          f"- without: {sum(r[2] for r in nocue)}/{len(nocue)} wrongly flagged", "",
          "## Missed risky requests (R1, both)", ""]
    for it, s, flag in main["rows"]:
        if it["risky"] and not flag:
            L.append(f"- [{it.get('risk_category')}] score {s:.2f}: {it['request']}")
    (RESULTS / "t3_external.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20]))
    print(f"\nWrote {RESULTS / 't3_external.md'}")


if __name__ == "__main__":
    main()
