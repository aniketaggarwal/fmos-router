"""Build the pilot dataset: data/pilot.jsonl and data/passages.jsonl.

  T0  SQuAD question WITH its passage            (5 items)
  T1  SQuAD question WITHOUT its passage         (8 candidates)
  T2  GSM8K math word problem                    (8 candidates)
  T3  your hand-written risky requests + twins   (from data/t3_pilot.jsonl)

T1 and T2 are "candidates": run_pilot.py checks which of them really qualify.
Run from the project folder: python build_pilot.py
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_T0, N_T1, N_T2 = 5, 8, 8
DATA = Path("data")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_squad(rng):
    squad = load_dataset("rajpurkar/squad", split="validation")

    # Passage store for retrieval (O1): every unique SQuAD validation passage.
    passage_id, passages = {}, []
    for ex in squad:
        if ex["context"] not in passage_id:
            passage_id[ex["context"]] = f"p{len(passages):05d}"
            passages.append({"id": passage_id[ex["context"]], "title": ex["title"], "text": ex["context"]})
    write_jsonl(DATA / "passages.jsonl", passages)

    # Pick questions from different passages so items are independent.
    idx = list(range(len(squad)))
    rng.shuffle(idx)
    picked, used = [], set()
    for i in idx:
        ex = squad[i]
        if ex["context"] in used:
            continue
        used.add(ex["context"])
        picked.append(ex)
        if len(picked) == N_T0 + N_T1:
            break

    items = []
    for k, ex in enumerate(picked):
        is_t1 = k >= N_T0
        items.append({
            "id": f"{'t1' if is_t1 else 't0'}-{k:04d}",
            "category": "T1" if is_t1 else "T0",
            "task_type": "extractive",
            "source": "squad",
            "request": ex["question"],
            "context": "" if is_t1 else ex["context"],      # T1: passage removed
            "gold_passage_id": passage_id[ex["context"]],
            "gold_answers": list(dict.fromkeys(ex["answers"]["text"])),
            "tool_call": None,
            "pair_id": None,
        })
    return items, len(passages)


def build_gsm8k(rng):
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(gsm)), N_T2)
    items = []
    for k, i in enumerate(idx):
        ex = gsm[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "id": f"t2-{k:04d}",
            "category": "T2",
            "task_type": "math",
            "source": "gsm8k",
            "request": ex["question"],
            "context": "",
            "gold_passage_id": None,
            "gold_answers": [gold],
            "tool_call": None,
            "pair_id": None,
        })
    return items


def load_t3():
    path = DATA / "t3_pilot.jsonl"
    if not path.exists():
        print(f"WARNING: {path} not found. Write your T3 items first (see t3_pilot_TEMPLATE.jsonl).")
        return []
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        risky = bool(r["risky"])
        items.append({
            "id": f"{'t3' if risky else 't0twin'}-{n:04d}",
            # A benign twin is a no-trigger control, so it belongs to T0.
            "category": "T3" if risky else "T0",
            "task_type": "action",
            "source": "handwritten",
            "request": r["request"],
            "context": "",
            "gold_passage_id": None,
            "gold_answers": [],
            "tool_call": r.get("tool_call"),
            "pair_id": r["pair_id"],
            "correct_ops": ["approval"] if risky else [],
        })
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], []).append(it["category"])
    bad = [p for p, cats in pairs.items() if sorted(cats) != ["T0", "T3"]]
    if bad:
        print(f"WARNING: these pair_ids do not have exactly one risky + one twin: {bad}")
    return items


def main():
    rng = random.Random(SEED)
    squad_items, n_passages = build_squad(rng)
    gsm_items = build_gsm8k(rng)
    t3_items = load_t3()
    items = squad_items + gsm_items + t3_items
    write_jsonl(DATA / "pilot.jsonl", items)

    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    print(f"Wrote {len(items)} items to data/pilot.jsonl: {counts}")
    print(f"Wrote {n_passages} passages to data/passages.jsonl")


if __name__ == "__main__":
    main()
