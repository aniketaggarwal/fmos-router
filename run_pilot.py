"""Run the pilot: check which items really behave like their category.

For each item it runs only the paths that item needs:
  T0  local model with the passage                 -> should be right
  T1  local without passage / local + retrieval /   -> qualifies if wrong without,
      Gemini without passage                           right with retrieval
  T2  local / Gemini                               -> qualifies if local wrong,
                                                       Gemini right
  T3  nothing (labeled by construction); file is only checked

Results go to data/pilot_results.jsonl; a summary is printed at the end.
Run from the project folder: python run_pilot.py
"""
import json
import re
import statistics
import string
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

from models import GEMINI_MIN_INTERVAL_S, generate, warmup

DATA = Path("data")
TOP_K = 2               # passages added by retrieval (O1)
F1_CUTOFF = 0.6         # extractive answer counts as correct at or above this token F1
ABSTAIN = "INSUFFICIENT_INFO"

# ------------------------------------------------------------- prompts ---
P_WITH_CONTEXT = (
    "Answer the question using the passage. Reply with only the answer, in a few words.\n\n"
    "Passage:\n{context}\n\nQuestion: {q}\nAnswer:"
)
P_NO_CONTEXT = (
    "Answer the question. If you cannot answer it reliably, reply exactly: " + ABSTAIN + "\n"
    "Reply with only the answer, in a few words.\n\nQuestion: {q}\nAnswer:"
)
P_MATH = (
    "Solve the problem. Show brief working, then end with a final line of the form\n"
    "ANSWER: <number>\n\nProblem: {q}"
)

# ------------------------------------------------------------- scoring ---
def normalize(s):
    """SQuAD-style normalization: lowercase, drop punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def score_extractive(pred, golds):
    if ABSTAIN in pred:
        return False, 0.0
    best = max((token_f1(pred, g) for g in golds), default=0.0)
    exact = any(normalize(pred) == normalize(g) for g in golds)
    return exact or best >= F1_CUTOFF, round(best, 3)


def extract_number(text):
    m = re.search(r"ANSWER:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "") if nums else None


def score_math(pred, golds):
    got = extract_number(pred)
    try:
        return got is not None and abs(float(got) - float(golds[0])) < 1e-6, got
    except ValueError:
        return False, got

# ----------------------------------------------------------- retrieval ---
def tokenize(s):
    return re.findall(r"\w+", s.lower())


def load_retriever():
    passages = [json.loads(l) for l in (DATA / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
    bm25 = BM25Okapi([tokenize(p["title"] + " " + p["text"]) for p in passages])
    return bm25, passages


def retrieve(bm25, passages, query, k=TOP_K):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:k]
    return [passages[i] for i in top]

# ----------------------------------------------------------------- run ---
def run(item, backend, prompt, max_tokens, run_name):
    r = generate(prompt, backend=backend, max_tokens=max_tokens, tag=f"pilot:{item['id']}:{run_name}")[0]
    if item["task_type"] == "math":
        ok, detail = score_math(r["text"], item["gold_answers"])
    else:
        ok, detail = score_extractive(r["text"], item["gold_answers"])
    return {
        "run": run_name, "backend": backend, "correct": ok, "detail": detail,
        "text": r["text"].strip()[:300], "wall_ms": r["wall_ms"], "cached": r["cached"],
        "output_tokens": r["output_tokens"], "thinking_tokens": r["thinking_tokens"],
        "abstained": ABSTAIN in r["text"],
    }


def main():
    items = [json.loads(l) for l in (DATA / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    bm25, passages = load_retriever()
    n_gemini = sum(1 for it in items if it["category"] in ("T1", "T2"))
    print(f"{len(items)} items, about {n_gemini} Gemini calls "
          f"(~{n_gemini * GEMINI_MIN_INTERVAL_S / 60:.0f} min at the free-tier pace, less if cached)")
    warmup()

    results = []
    for it in items:
        cat, q = it["category"], it["request"]
        res = {"id": it["id"], "category": cat, "runs": []}

        if cat == "T0" and it["task_type"] == "extractive":
            res["runs"].append(run(it, "local", P_WITH_CONTEXT.format(context=it["context"], q=q), 32, "local_ctx"))
            res["qualifies"] = res["runs"][0]["correct"]

        elif cat == "T1":
            hits = retrieve(bm25, passages, q)
            res["retrieval_hit"] = any(p["id"] == it["gold_passage_id"] for p in hits)
            ctx = "\n\n".join(p["text"] for p in hits)
            no_ctx = run(it, "local", P_NO_CONTEXT.format(q=q), 32, "local_noctx")
            with_ret = run(it, "local", P_WITH_CONTEXT.format(context=ctx, q=q), 32, "local_retrieve")
            gem = run(it, "gemini", P_NO_CONTEXT.format(q=q), 32, "gemini_noctx")
            res["runs"] += [no_ctx, with_ret, gem]
            res["qualifies"] = (not no_ctx["correct"]) and with_ret["correct"]

        elif cat == "T2":
            loc = run(it, "local", P_MATH.format(q=q), 512, "local")
            gem = run(it, "gemini", P_MATH.format(q=q), 512, "gemini")
            res["runs"] += [loc, gem]
            res["qualifies"] = (not loc["correct"]) and gem["correct"]

        else:  # T3 and benign twins: labeled by construction, no model calls in the pilot
            res["qualifies"] = None

        results.append(res)
        status = {True: "QUALIFIES", False: "does not qualify", None: "by construction"}[res.get("qualifies")]
        print(f"{it['id']:<14} {cat:<3} {status}")

    with (DATA / "pilot_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --------------------------------------------------------- summary ---
    print("\n== Yield: share of candidates that behave like their category ==")
    for cat in ("T0", "T1", "T2"):
        rows = [r for r in results if r["category"] == cat and r.get("qualifies") is not None]
        if rows:
            q = sum(r["qualifies"] for r in rows)
            print(f"{cat}: {q}/{len(rows)} qualify ({100 * q / len(rows):.0f}%)")

    t1 = [r for r in results if r["category"] == "T1"]
    if t1:
        hit = sum(r["retrieval_hit"] for r in t1)
        abst = sum(r["runs"][0]["abstained"] for r in t1)
        print(f"T1 retrieval found the gold passage in top {TOP_K}: {hit}/{len(t1)}")
        print(f"T1 local model abstained without the passage: {abst}/{len(t1)}")

    print("\n== Accuracy and median latency per run type ==")
    by_run = {}
    for r in results:
        for x in r["runs"]:
            by_run.setdefault(x["run"], []).append(x)
    for name, xs in by_run.items():
        acc = sum(x["correct"] for x in xs) / len(xs)
        live = [x["wall_ms"] for x in xs if not x["cached"]] or [x["wall_ms"] for x in xs]
        print(f"{name:<16} n={len(xs):<3} accuracy={acc:.0%}  median wall={statistics.median(live):8.1f} ms")

    print(f"\n{sum(1 for r in results if r['category'] in ('T3',))} T3 items and "
          f"{sum(1 for r in results if r['id'].startswith('t0twin'))} benign twins loaded (no model calls).")
    print("Per-item details: data/pilot_results.jsonl")


if __name__ == "__main__":
    main()
