"""Label every candidate by OUTCOME, then split into train/val/test.

For each item, paths are tried cheapest first and the first one that gives a
correct answer is the label (see common.py):

  fast  ->  retrieve  ->  strong  ->  retrieve+strong  ->  unsolvable

  label path        category   correct_ops
  fast              T0         []
  retrieve          T1         ["retrieve"]
  strong            T2         ["strong"]
  retrieve+strong   T1+T2      ["retrieve", "strong"]
  (none correct)    unsolvable dropped from routing metrics
  action, risky     T3         ["approval"]      by construction
  action, twin      T0         []                by construction

The local model runs once (temperature 0 is deterministic). Gemini runs at
temperature 1, so its result is the majority of up to 3 samples (it stops at
2 when they agree). Every run is appended to data/label_runs.jsonl, so the
script can be stopped and restarted without repeating work.

Run: python label.py              (all candidates)
     python label.py --limit 40   (first 40 only, to check time and cost)
"""
import argparse
import random
import time
from collections import Counter, defaultdict

from common import DATA, PATH_OPS, PATHS, append_jsonl, path_applies, read_jsonl, run_path, write_jsonl

RUNS_FILE = DATA / "label_runs.jsonl"
CATEGORY = {"fast": "T0", "retrieve": "T1", "strong": "T2", "retrieve+strong": "T1+T2"}
SPLITS = (("train", 0.6), ("val", 0.2), ("test", 0.2))


def load_done():
    done = defaultdict(dict)  # item_id -> {(path, sample): run}
    for r in read_jsonl(RUNS_FILE):
        done[r["item_id"]][(r["path"], r["sample"])] = r
    return done


def get_run(item, path, sample, done, generate):
    key = (path, sample)
    if key not in done[item["id"]]:
        r = run_path(item, path, generate, tag_prefix="label:", sample=sample)
        r["item_id"] = item["id"]
        append_jsonl(RUNS_FILE, r)
        done[item["id"]][key] = r
    return done[item["id"]][key]


GEMINI_SAMPLES = 3  # set by --gemini-samples


def path_correct(item, path, done, generate):
    """Local: one run. Gemini: one sample, or majority of up to 3."""
    if "strong" not in path:
        return get_run(item, path, 0, done, generate)["correct"]
    if GEMINI_SAMPLES == 1:
        return get_run(item, path, 0, done, generate)["correct"]
    votes = [get_run(item, path, s, done, generate)["correct"] for s in (0, 1)]
    if votes[0] != votes[1]:
        votes.append(get_run(item, path, 2, done, generate)["correct"])
    return sum(votes) * 2 > len(votes)


def label_item(item, done, generate):
    if item["task_type"] == "action":
        risky = item["risky"]
        return {"category": "T3" if risky else "T0", "label_path": None,
                "correct_ops": ["approval"] if risky else []}
    for path in PATHS:
        if path_applies(item, path) and path_correct(item, path, done, generate):
            return {"category": CATEGORY[path], "label_path": path, "correct_ops": PATH_OPS[path]}
    return {"category": "unsolvable", "label_path": None, "correct_ops": None}


def assign_splits(items, seed):
    """Stratified by category; a T3 item and its twin always share a split."""
    rng = random.Random(seed)
    groups = defaultdict(list)  # group key -> items
    for it in items:
        groups[it.get("pair_id") or it["id"]].append(it)
    by_cat = defaultdict(list)
    for key, members in groups.items():
        cat = "T3" if any(m["category"] == "T3" for m in members) else members[0]["category"]
        by_cat[cat].append(key)
    for cat, keys in by_cat.items():
        rng.shuffle(keys)
        n = len(keys)
        cut1 = round(n * SPLITS[0][1])
        cut2 = cut1 + round(n * SPLITS[1][1])
        for i, key in enumerate(keys):
            split = "train" if i < cut1 else "val" if i < cut2 else "test"
            for m in groups[key]:
                m["split"] = split


def main(generate=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gemini-samples", type=int, choices=(1, 3), default=3,
                    help="1 = cheapest (one Gemini answer per path); 3 = majority vote (stops at 2 if they agree)")
    args = ap.parse_args()
    global GEMINI_SAMPLES
    GEMINI_SAMPLES = args.gemini_samples

    if generate is None:
        from models import generate as real_generate, warmup
        warmup()
        generate = real_generate

    items = read_jsonl(DATA / "candidates.jsonl")
    if args.limit:
        # Take items round-robin across recipes so a trial run sees every kind.
        by_recipe = defaultdict(list)
        for it in items:
            by_recipe[it["recipe"]].append(it)
        queues = list(by_recipe.values())
        mixed = []
        while len(mixed) < args.limit and any(queues):
            for q in queues:
                if q and len(mixed) < args.limit:
                    mixed.append(q.pop(0))
        items = mixed
    done = load_done()
    t0 = time.time()

    labeled = []
    for n, it in enumerate(items, 1):
        it = dict(it, **label_item(it, done, generate))
        labeled.append(it)
        if n % 10 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time() - t0:6.0f}s elapsed  "
                  f"{dict(Counter(x['category'] for x in labeled))}")

    assign_splits(labeled, args.seed)
    write_jsonl(DATA / "labeled.jsonl", labeled)

    print("\n== Category by recipe (this is the yield) ==")
    table = defaultdict(Counter)
    for it in labeled:
        table[it["recipe"]][it["category"]] += 1
    for recipe, c in table.items():
        total = sum(c.values())
        print(f"{recipe:<12} n={total:<4} " + "  ".join(f"{k}={v} ({100 * v / total:.0f}%)" for k, v in sorted(c.items())))

    print("\n== Final counts per split ==")
    split_cat = defaultdict(Counter)
    for it in labeled:
        split_cat[it["split"]][it["category"]] += 1
    for split in ("train", "val", "test"):
        print(f"{split:<6} {dict(sorted(split_cat[split].items()))}")

    runs = read_jsonl(RUNS_FILE)
    gem = [r for r in runs if "strong" in r["path"] and not r["cached"]]
    print(f"\nRuns logged: {len(runs)} (new Gemini API calls this file: {len(gem)})")
    print("Wrote data/labeled.jsonl")


if __name__ == "__main__":
    main()
