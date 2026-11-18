"""Confidence intervals for the test results, from results/predictions.jsonl.

No model calls. Uses a bootstrap: resample the 164 test items with
replacement 5,000 times and see how much each number moves.

  1. 95% CI for each method's end-to-end accuracy, mean latency and Gemini rate
  2. Paired differences between key methods (same resampled items for both),
     with the share of resamples in which the first method is better

Run: python analyze.py     -> prints tables and writes results/confidence.md
"""
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

N_BOOT = 5000
SEED = 0
PAIRS = [("hybrid", "R1"), ("hybrid", "cascade"), ("R1", "always_fast"),
         ("hybrid", "always_fast"), ("cascade", "always_fast"), ("hybrid", "hybrid_v1")]

rows = [json.loads(l) for l in Path("results/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
by_method = defaultdict(dict)  # method -> item id -> row
for r in rows:
    by_method[r["method"]][r["id"]] = r
methods = [m for m in by_method if m.startswith(("always", "random", "R0", "R1", "cascade", "hybrid", "oracle"))]
ids = sorted(next(iter(by_method.values())).keys())
rng = random.Random(SEED)
samples = [[rng.choice(ids) for _ in ids] for _ in range(N_BOOT)]


def stat(method, sample, key):
    rs = [by_method[method][i] for i in sample]
    if key == "acc":
        return sum(r["e2e_correct"] for r in rs) / len(rs)
    if key == "lat":
        return statistics.mean(r["latency_ms"] for r in rs)
    if key == "gem":
        return sum(r.get("uses_gemini", False) for r in rs) / len(rs)


def ci(values):
    v = sorted(values)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


out = ["# Confidence intervals (bootstrap, 95%)", "",
       f"{len(ids)} test items, {N_BOOT} resamples.", "",
       "| Method | Accuracy | 95% CI | Mean latency (s) | 95% CI | Sent to Gemini | 95% CI |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for m in methods:
    acc = stat(m, ids, "acc")
    lat = stat(m, ids, "lat") / 1000
    gem = stat(m, ids, "gem")
    a_lo, a_hi = ci([stat(m, s, "acc") for s in samples])
    l_lo, l_hi = ci([stat(m, s, "lat") / 1000 for s in samples])
    g_lo, g_hi = ci([stat(m, s, "gem") for s in samples])
    out.append(f"| {m} | {100 * acc:.1f}% | {100 * a_lo:.1f}–{100 * a_hi:.1f}% | {lat:.1f} | "
               f"{l_lo:.1f}–{l_hi:.1f} | {100 * gem:.1f}% | {100 * g_lo:.1f}–{100 * g_hi:.1f}% |")

out += ["", "## Paired differences", "",
        "| Comparison | Accuracy difference | 95% CI | First is more accurate in | "
        "Latency difference (s) | Gemini-rate difference |",
        "| --- | --- | --- | --- | --- | --- |"]
for a, b in PAIRS:
    if a not in by_method or b not in by_method:
        continue
    diffs = [stat(a, s, "acc") - stat(b, s, "acc") for s in samples]
    d = stat(a, ids, "acc") - stat(b, ids, "acc")
    lo, hi = ci(diffs)
    better = sum(x > 0 for x in diffs) / len(diffs)
    dl = (stat(a, ids, "lat") - stat(b, ids, "lat")) / 1000
    dg = stat(a, ids, "gem") - stat(b, ids, "gem")
    out.append(f"| {a} vs {b} | {100 * d:+.1f} pts | {100 * lo:+.1f} to {100 * hi:+.1f} | "
               f"{100 * better:.0f}% of resamples | {dl:+.1f} | {100 * dg:+.1f} pts |")

out += ["", "A difference is reliable when its 95% CI excludes 0 "
            "(equivalently, the first method wins in about 97.5% or more of resamples)."]
text = "\n".join(out)
Path("results/confidence.md").write_text(text, encoding="utf-8")
print(text)
