"""Extra analyses from saved results. No model calls.

Reads data/labeled.jsonl, data/label_runs.jsonl, data/eval_runs.jsonl and
results/predictions.jsonl, and writes results/extra.md with:

  1. Worked examples (real requests and answers) for the paper's error analysis
  2. Paid-model cost per 1,000 requests for each method, from logged tokens
  3. Dataset statistics: request lengths, retrieval hit rate, labeling effort

Run: python extra_analysis.py   (after evaluate.py)
"""
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, read_jsonl

RESULTS = Path("results")
# Gemini 3.8 Flash list price, USD per 1M tokens (introductory pricing through
# 31 Dec 2026). Thinking tokens are billed as output tokens.
PRICE_IN, PRICE_OUT = 0.75, 3.75
CASCADE_VARIANT = "cascade_t0.7"


def short(s, n=220):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def load():
    labeled = {it["id"]: it for it in read_jsonl(DATA / "labeled.jsonl")}
    preds = defaultdict(dict)  # method -> id -> row
    for r in read_jsonl(RESULTS / "predictions.jsonl"):
        preds[r["method"]][r["id"]] = r
    runs = {}
    for f in (DATA / "label_runs.jsonl", DATA / "eval_runs.jsonl"):
        for r in read_jsonl(f):
            runs[(r["item_id"], r["path"], r["sample"], r.get("variant", "label"))] = r
    return labeled, preds, runs


def run_for(runs, iid, path):
    """The run used for an item's path: labeling first, then evaluation."""
    for variant in ("label", "label:", "eval"):
        r = runs.get((iid, path, 0, variant))
        if r:
            return r
    for (i, p, s, v), r in runs.items():
        if i == iid and p == path and s == 0:
            return r
    return None

# ------------------------------------------------------------ examples ---
def examples(labeled, preds, runs):
    out = ["## 1. Worked examples", ""]

    def item_block(it, extra):
        lines = [f"- **Request** ({it['id']}, {it['category']}): {short(it['request'])}"]
        if it.get("gold_answers"):
            lines.append(f"- **Gold answer:** {short(', '.join(it['gold_answers']), 120)}")
        lines += extra
        return lines

    def cascade_answers(iid):
        ans = [runs.get((iid, "fast", s, CASCADE_VARIANT)) for s in range(5)]
        return [short(a["text"], 60) for a in ans if a]

    # (a) Missing context, cascade sends to the strong model instead of retrieval
    casc = preds.get("cascade", {})
    picks = [i for i, r in casc.items() if r["true_path"] == "retrieve" and r["pred_path"] == "strong"][:2]
    out += ["### (a) Missing context: the cascade escalates to the strong model instead of retrieving", ""]
    for iid in picks:
        it = labeled[iid]
        fast = run_for(runs, iid, "fast")
        ret = run_for(runs, iid, "retrieve")
        out += item_block(it, [
            f"- **Fast model, no passage:** {short(fast['text'], 80) if fast else 'n/a'}",
            f"- **Cascade samples:** {' | '.join(cascade_answers(iid)) or 'n/a'}",
            f"- **Fast model after retrieval:** {short(ret['text'], 80) if ret else 'n/a'} "
            f"({'correct' if ret and ret['correct'] else 'wrong'})",
        ]) + [""]

    # (b) Hard request caught by disagreement but missed by R1
    r1 = preds.get("R1", {})
    picks = [i for i, r in casc.items() if r["true_path"] == "strong" and r["pred_path"] == "strong"
             and r1.get(i, {}).get("pred_path") != "strong"][:2]
    out += ["### (b) Hard request: caught by sample disagreement, missed by the request-only router", ""]
    for iid in picks:
        it = labeled[iid]
        fast = run_for(runs, iid, "fast")
        strong = run_for(runs, iid, "strong")
        out += item_block(it, [
            f"- **Fast model final line:** {short(fast['text'].strip().splitlines()[-1], 80) if fast else 'n/a'}",
            f"- **Cascade samples (final answers):** "
            f"{' | '.join(str(runs[(iid, 'fast', s, CASCADE_VARIANT)].get('answer_key')) for s in range(5) if (iid, 'fast', s, CASCADE_VARIANT) in runs)}",
            f"- **Strong model final line:** {short(strong['text'].strip().splitlines()[-1], 80) if strong else 'n/a'} "
            f"({'correct' if strong and strong['correct'] else 'wrong'})",
            f"- **R1 strong-model score:** {r1[iid]['scores']['strong']:.2f}" if r1.get(iid, {}).get("scores") else "",
        ]) + [""]

    # (c) Risky action and its twin, both sent to approval by R1
    pairs = defaultdict(dict)
    for iid, it in labeled.items():
        if it["task_type"] == "action" and iid in r1:
            pairs[it["pair_id"]]["risky" if it["risky"] else "twin"] = it
    shown = 0
    out += ["### (c) A risky action and its benign twin that R1 cannot tell apart", ""]
    for pid, p in pairs.items():
        if "risky" in p and "twin" in p and "approval" in r1[p["twin"]["id"]]["pred_ops"]:
            for role in ("risky", "twin"):
                it = p[role]
                s = r1[it["id"]].get("scores") or {}
                out.append(f"- **{role.title()}** ({it.get('risk_category') or 'n/a'}): {short(it['request'], 160)} "
                           f"Tool call: `{short(it['tool_call'], 160)}`. R1 approval score: {s.get('approval', float('nan')):.2f}")
            out.append("")
            shown += 1
            if shown == 2:
                break

    # (d) Hybrid mistakes
    hyb = preds.get("hybrid", {})
    wrong = [r for r in hyb.values() if not r["e2e_correct"]]
    out += ["### (d) Where the hybrid is still wrong", "",
            f"{len(wrong)} of {len(hyb)} test requests. By type: "
            + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['category'] for r in wrong).items())) + ".", "",
            "| Request | Type | True path | Chosen path |", "| --- | --- | --- | --- |"]
    for r in wrong[:8]:
        out.append(f"| {short(labeled[r['id']]['request'], 90)} | {r['category']} | {r['true_path']} | {r['pred_path']} |")
    out.append("")
    return out

# ---------------------------------------------------------------- cost ---
def cost(labeled, preds, runs):
    out = ["## 2. Paid-model cost per 1,000 requests", "",
           f"Gemini 3.8 Flash at ${PRICE_IN}/1M input and ${PRICE_OUT}/1M output tokens (thinking billed as output). "
           "Token counts come from the logged strong-model calls for the test requests each method sends to Gemini.", "",
           "| Method | Sent to Gemini | Input tokens / request | Output + thinking tokens / request | USD per 1,000 requests |",
           "| --- | --- | --- | --- | --- |"]
    for method, rows in preds.items():
        tin = tout = n_gem = 0
        for iid, r in rows.items():
            if not r.get("uses_gemini"):
                continue
            path = "retrieve+strong" if "retrieve" in r["pred_ops"] and labeled[iid]["task_type"] == "extractive" else "strong"
            run = run_for(runs, iid, path)
            if not run:
                continue
            n_gem += 1
            tin += run.get("prompt_tokens", 0)
            tout += run.get("output_tokens", 0) + run.get("thinking_tokens", 0)
        n = len(rows)
        usd = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6 / n * 1000
        out.append(f"| {method} | {100 * n_gem / n:.0f}% | {tin / n:.0f} | {tout / n:.0f} | ${usd:.3f} |")
    out += ["", "Local-model cost is electricity and hardware time only; its latency is reported in Table 8.", ""]
    return out

# --------------------------------------------------------------- stats ---
def stats(labeled, runs):
    out = ["## 3. Dataset statistics", "",
           "| Type | n | Mean request words | Mean context words |", "| --- | --- | --- | --- |"]
    by = defaultdict(list)
    for it in labeled.values():
        by[it["category"]].append(it)
    for cat, its in sorted(by.items()):
        rq = statistics.mean(len(it["request"].split()) for it in its)
        cx = statistics.mean(len((it.get("context") or "").split()) for it in its)
        out.append(f"| {cat} | {len(its)} | {rq:.1f} | {cx:.1f} |")

    hits = [r["retrieval"]["hit"] for (iid, path, s, v), r in runs.items()
            if path == "retrieve" and v == "label" and r.get("retrieval") and iid in labeled]
    t1_hits = [r["retrieval"]["hit"] for (iid, path, s, v), r in runs.items()
               if path == "retrieve" and v == "label" and r.get("retrieval") and labeled.get(iid, {}).get("category") == "T1"]
    label_runs = [r for k, r in runs.items() if k[3] == "label"]
    gem = sum(1 for r in label_runs if "strong" in r["path"] and not r.get("cached"))
    out += ["",
            f"- Retrieval found the gold passage in the top 2 for {sum(hits)}/{len(hits)} requests it was run on "
            f"({100 * sum(hits) / max(1, len(hits)):.0f}%); for requests labeled T1: {sum(t1_hits)}/{len(t1_hits)}.",
            f"- Labeling ran {len(label_runs)} model calls, of which {gem} were new Gemini calls.",
            ""]
    return out


def main():
    labeled, preds, runs = load()
    lines = ["# Extra analyses", ""] + examples(labeled, preds, runs) + cost(labeled, preds, runs) + stats(labeled, runs)
    text = "\n".join(l for l in lines if l is not None)
    (RESULTS / "extra.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
