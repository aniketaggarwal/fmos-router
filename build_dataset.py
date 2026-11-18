"""Build the full candidate set: data/candidates.jsonl (+ data/passages.jsonl).

Items are built from RECIPES, not final categories. label.py runs them and
assigns the real category from outcomes:

  recipe        source                       likely becomes
  squad_ctx     SQuAD question + its passage   T0 (fast path is enough)
  squad_noctx   SQuAD question, passage gone   T1 if retrieval fixes it, else T0/T2/unsolvable
  gsm8k         GSM8K math problem            T2 if only Gemini solves it, T0 if local does
  action        your file data/t3_items.jsonl  T3 (risky) or T0 (benign twin), by construction

Because local-correct math and no-passage items land in T0, the router cannot
learn shortcuts like "math means slow path" or "no passage means retrieve".

Set the pool sizes from the pilot's yields, e.g. if 30% of no-passage items
qualify as T1 and you want 75, use --squad-noctx 250.

Run: python build_dataset.py --squad-ctx 80 --squad-noctx 250 --gsm8k 400
"""
import argparse
import random
from collections import Counter
from pathlib import Path

from common import DATA, read_jsonl, write_jsonl


def squad_items(n_ctx, n_noctx, rng):
    from datasets import load_dataset
    squad = load_dataset("rajpurkar/squad", split="validation")

    passage_id, passages = {}, []
    for ex in squad:
        if ex["context"] not in passage_id:
            passage_id[ex["context"]] = f"p{len(passages):05d}"
            passages.append({"id": passage_id[ex["context"]], "title": ex["title"], "text": ex["context"]})
    write_jsonl(DATA / "passages.jsonl", passages)

    # One question per passage so items are independent.
    idx = list(range(len(squad)))
    rng.shuffle(idx)
    picked, used = [], set()
    for i in idx:
        ex = squad[i]
        if ex["context"] in used:
            continue
        used.add(ex["context"])
        picked.append(ex)
        if len(picked) == n_ctx + n_noctx:
            break
    if len(picked) < n_ctx + n_noctx:
        print(f"WARNING: only {len(picked)} distinct SQuAD passages available")

    items = []
    for k, ex in enumerate(picked):
        with_ctx = k < n_ctx
        items.append({
            "id": f"{'sc' if with_ctx else 'sn'}-{k:04d}",
            "recipe": "squad_ctx" if with_ctx else "squad_noctx",
            "task_type": "extractive",
            "source": "squad",
            "request": ex["question"],
            "context": ex["context"] if with_ctx else "",
            "gold_passage_id": passage_id[ex["context"]],
            "gold_answers": list(dict.fromkeys(ex["answers"]["text"])),
            "tool_call": None,
            "pair_id": None,
        })
    return items, len(passages)


def _gsm_item(gsm, i, steps, item_id, recipe):
    ex = gsm[i]
    return {
        "id": item_id,
        "recipe": recipe,
        "task_type": "math",
        "source": "gsm8k",
        "gsm_index": i,
        "steps": steps,
        "request": ex["question"],
        "context": "",
        "gold_passage_id": None,
        "gold_answers": [ex["answer"].split("####")[-1].strip().replace(",", "")],
        "tool_call": None,
        "pair_id": None,
    }


def gsm8k_extra_items(n, min_steps, exclude, seed):
    """Extra, harder math candidates to top up T2. IDs use the GSM8K index
    (gh-<index>), so adding more later never changes existing IDs."""
    if n <= 0:
        return []
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    pool = []
    for i, ex in enumerate(gsm):
        if i in exclude:
            continue
        steps = len([l for l in ex["answer"].split("####")[0].splitlines() if l.strip()])
        if steps >= min_steps:
            pool.append((i, steps))
    random.Random(seed + 1).shuffle(pool)
    if len(pool) < n:
        print(f"NOTE: only {len(pool)} unused GSM8K problems with >= {min_steps} steps; using all of them")
    return [_gsm_item(gsm, i, steps, f"gh-{i:04d}", "gsm8k_hard") for i, steps in pool[:n]]


def gsm8k_items(n, min_steps, rng):
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    pool = []
    for i, ex in enumerate(gsm):
        working = ex["answer"].split("####")[0]
        steps = len([l for l in working.splitlines() if l.strip()])
        if steps >= min_steps:
            pool.append((i, steps))
    rng.shuffle(pool)
    if len(pool) < n:
        print(f"WARNING: only {len(pool)} GSM8K problems with >= {min_steps} steps")
    items = []
    for k, (i, steps) in enumerate(pool[:n]):
        ex = gsm[i]
        items.append({
            "id": f"gm-{k:04d}",
            "gsm_index": i,
            "recipe": "gsm8k",
            "task_type": "math",
            "source": "gsm8k",
            "steps": steps,
            "request": ex["question"],
            "context": "",
            "gold_passage_id": None,
            "gold_answers": [ex["answer"].split("####")[-1].strip().replace(",", "")],
            "tool_call": None,
            "pair_id": None,
        })
    return items


def action_items(paths):
    """Load T3 files. Several files can be combined (e.g. Claude-written and
    teammate-written); IDs and pair_ids are prefixed with the file name so
    they never collide, and each item keeps its source for later comparison."""
    items = []
    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            continue
        stem = Path(path).stem.replace("t3_items", "").strip("_") or "main"
        for n, r in enumerate(rows):
            risky = bool(r["risky"])
            items.append({
                "id": f"{'t3' if risky else 'tw'}-{stem}-{n:04d}",
                "recipe": "action",
                "task_type": "action",
                "source": r.get("source", stem),
                "risk_category": r.get("risk_category"),
                "risky": risky,
                "request": r["request"],
                "context": r.get("context", ""),
                "gold_passage_id": None,
                "gold_answers": [],
                "tool_call": r.get("tool_call"),
                "pair_id": f"{stem}:{r['pair_id']}",
            })
        print(f"Loaded {len(rows)} action items from {path}")
    if not items:
        print("WARNING: no T3 files found (data/t3_items*.jsonl). Write them before labeling.")
        return []
    pairs = Counter(it["pair_id"] for it in items)
    risky_per_pair = Counter(it["pair_id"] for it in items if it["risky"])
    bad = [p for p in pairs if pairs[p] != 2 or risky_per_pair[p] != 1]
    if bad:
        print(f"WARNING: pair_ids without exactly one risky + one twin: {bad}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-ctx", type=int, default=80)
    ap.add_argument("--squad-noctx", type=int, default=250)
    ap.add_argument("--gsm8k", type=int, default=400)
    ap.add_argument("--gsm-min-steps", type=int, default=0,
                    help="keep only GSM8K problems whose reference solution has at least this many lines")
    ap.add_argument("--gsm-extra", type=int, default=0,
                    help="extra harder GSM8K candidates (recipe gsm8k_hard) to top up T2")
    ap.add_argument("--gsm-extra-min-steps", type=int, default=5,
                    help="minimum solution lines for the extra GSM8K candidates")
    ap.add_argument("--actions", nargs="*", default=None,
                    help="T3 files to load; default: every data/t3_items*.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sq, n_pass = squad_items(args.squad_ctx, args.squad_noctx, rng)
    gm = gsm8k_items(args.gsm8k, args.gsm_min_steps, rng)
    extra = gsm8k_extra_items(args.gsm_extra, args.gsm_extra_min_steps,
                              exclude={it["gsm_index"] for it in gm}, seed=args.seed)
    action_files = args.actions if args.actions is not None else sorted(str(p) for p in DATA.glob("t3_items*.jsonl"))
    items = sq + gm + extra + action_items(action_files)
    write_jsonl(DATA / "candidates.jsonl", items)
    print(f"Wrote {len(items)} candidates: {dict(Counter(it['recipe'] for it in items))}")
    print(f"Passage store: {n_pass} passages")


if __name__ == "__main__":
    main()
