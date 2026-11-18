"""Shared pieces: file I/O, prompts, answer scoring, retrieval, and the
paths a request can take. Every other script imports from here, so a prompt
or scoring rule is defined exactly once.

Paths, cheapest first (the label for an item is the cheapest path that is correct):
  fast             local model, item as given                   ops = []
  retrieve         local model + top-k retrieved passages        ops = ["retrieve"]
  strong           Gemini, item as given                         ops = ["strong"]
  retrieve+strong  Gemini + top-k retrieved passages             ops = ["retrieve", "strong"]
Action items (T3 and their benign twins) are not answered; approval is a
label by construction.
"""
import json
import re
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path

DATA = Path("data")
TOP_K = 2
F1_CUTOFF = 0.6
ABSTAIN = "INSUFFICIENT_INFO"

PATHS = ["fast", "retrieve", "strong", "retrieve+strong"]
PATH_OPS = {
    "fast": [],
    "retrieve": ["retrieve"],
    "strong": ["strong"],
    "retrieve+strong": ["retrieve", "strong"],
}
OPS = ["retrieve", "strong", "approval"]

# -------------------------------------------------------------- file I/O ---
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# --------------------------------------------------------------- prompts ---
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
MAX_TOKENS = {"extractive": 32, "math": 512}


def build_prompt(item, context=None):
    """context=None uses the item's own context; a string overrides it."""
    ctx = item.get("context", "") if context is None else context
    if item["task_type"] == "math":
        if ctx:
            return P_MATH.format(q=item["request"]) + "\n\nReference material:\n" + ctx
        return P_MATH.format(q=item["request"])
    if ctx:
        return P_WITH_CONTEXT.format(context=ctx, q=item["request"])
    return P_NO_CONTEXT.format(q=item["request"])

# --------------------------------------------------------------- scoring ---
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


CONTAIN_MAX_TOKENS = 12  # a short answer that contains the gold answer counts as correct


def score_extractive(pred, golds):
    """Correct if: exact match, or token F1 >= 0.6, or a short answer
    (<= 12 words) contains a gold answer as whole words. The last rule stops
    "linear and branched" from failing against the gold answer "linear"."""
    if ABSTAIN in pred:
        return False, 0.0
    p = normalize(pred)
    best = max((token_f1(pred, g) for g in golds), default=0.0)
    exact = any(p == normalize(g) for g in golds)
    contained = len(p.split()) <= CONTAIN_MAX_TOKENS and any(
        normalize(g) and re.search(r"\b" + re.escape(normalize(g)) + r"\b", p) for g in golds)
    return exact or contained or best >= F1_CUTOFF, round(best, 3)


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


def score(item, text):
    if item["task_type"] == "math":
        return score_math(text, item["gold_answers"])
    return score_extractive(text, item["gold_answers"])

# ------------------------------------------------------------- retrieval ---
def tokenize(s):
    return re.findall(r"\w+", s.lower())


@lru_cache(maxsize=1)
def _retriever():
    from rank_bm25 import BM25Okapi
    passages = read_jsonl(DATA / "passages.jsonl")
    bm25 = BM25Okapi([tokenize(p["title"] + " " + p["text"]) for p in passages])
    return bm25, passages


def retrieve(query, k=TOP_K):
    """Return (top passages, their BM25 scores)."""
    bm25, passages = _retriever()
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:k]
    return [passages[i] for i in top], [float(scores[i]) for i in top]

# ----------------------------------------------------------------- paths ---
def path_applies(item, path):
    """Retrieval only makes sense for extractive QA over the passage store."""
    if item["task_type"] == "action":
        return False
    if "retrieve" in path and item["task_type"] != "extractive":
        return False
    return True


def answer_key(item, text):
    """The part of an answer used for voting: the final number for math,
    the normalized text otherwise."""
    if item["task_type"] == "math":
        return extract_number(text)
    return normalize(text)


def run_path(item, path, generate, tag_prefix="", sample=0, temperature=None, seed=None):
    """Run one path on one item and score it. `generate` is models.generate,
    passed in so tests can swap in a fake. temperature/seed override the
    defaults (used by the agreement cascade to draw varied local samples)."""
    backend = "gemini" if "strong" in path else "local"
    context = None
    retrieval = None
    if "retrieve" in path:
        hits, scores = retrieve(item["request"])
        context = "\n\n".join(p["text"] for p in hits)
        retrieval = {"ids": [p["id"] for p in hits], "scores": scores,
                     "hit": item.get("gold_passage_id") in [p["id"] for p in hits]}
    prompt = build_prompt(item, context)
    # A different seed per repeat gives Gemini independent samples (and cache entries).
    if seed is None:
        seed = None if backend == "local" else 1000 + sample
    r = generate(prompt, backend=backend, max_tokens=MAX_TOKENS[item["task_type"]],
                 temperature=temperature, seed=seed,
                 tag=f"{tag_prefix}{item['id']}:{path}:{sample}")[0]
    ok, detail = score(item, r["text"])
    return {
        "path": path, "sample": sample, "correct": ok, "detail": detail,
        "text": r["text"].strip()[:300], "wall_ms": r["wall_ms"], "cached": r["cached"],
        "cold": r["cold"], "prompt_tokens": r["prompt_tokens"],
        "output_tokens": r["output_tokens"], "thinking_tokens": r["thinking_tokens"],
        "abstained": ABSTAIN in r["text"], "retrieval": retrieval,
        "answer_key": answer_key(item, r["text"]),
    }
