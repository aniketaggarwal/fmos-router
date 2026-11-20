# FMOS Router: Trigger-Aware Routing for a Foundation Model OS

This code decides, for each request, whether a cheap local model can answer it (fast path) or whether it needs one or more slow-path steps:

| Trigger | Meaning | Slow-path step |
| --- | --- | --- |
| T0 | Nothing wrong | none (fast path) |
| T1 | Missing information | `retrieve` (BM25 over a passage store) |
| T2 | Too hard for the small model | `strong` (Gemini) |
| T1+T2 | Both | `retrieve` + `strong` |
| T3 | Risky tool action | `approval` (ask a human) |

- **Fast model:** Qwen3.5 9B (4-bit) running locally in Ollama.
- **Strong model:** `gemini-3.8-flash` via the Google GenAI API.

---

## Setup

```powershell
pip install -r requirements.txt
ollama pull qwen3.5:9b
$env:GEMINI_API_KEY = "your-key"   # never commit this
python test_models.py               # smoke test: both models answer 5 questions
```

Settings live at the top of `models.py`:

- `LOCAL_MODEL` and `LOCAL_NUM_CTX` (6144 keeps the model fully on an 8 GB GPU).
- `GEMINI_MODEL` and `GEMINI_THINKING_LEVEL`.
- `GEMINI_MIN_INTERVAL_S`, the rate limit. Use 13 s on the free tier and about 1 s on a paid key.
- `GEMINI_CALL_BUDGET`, a hard cap on API calls.

---

## Pipeline

Run these in order. Every step saves its results, so reruns reuse them instead of calling the models again.

```powershell
# 1. Build the candidate requests
python build_dataset.py --squad-ctx 80 --squad-noctx 120 --gsm8k 250 --gsm-extra 250

# 2. Label every request by running every path (the slow step; uses the API)
python label.py

# 3. Evaluate all routers on the test split
python evaluate.py --cascade-k 3 --skip-always-slow

# 4. Statistics and extra analyses (no model calls)
python analyze.py
python extra_analysis.py

# 5. Cross-validation over all labeled requests (local samples only, no API)
python cv.py --with-cascade --k 3

# 6. Risk detection on the independently written T3 set (no model calls)
python t3_external.py
```

---

## Files

### Core

| File | What it does |
| --- | --- |
| `models.py` | One `generate()` function for both backends. It handles Ollama (`think: false`, fixed context, `keep_alive`) and Gemini (thinking level, throttling, retries, daily-quota detection, call budget). Every call is logged to `logs/calls.jsonl` with tokens and timings. Gemini answers are cached in `cache/gemini/`. |
| `common.py` | Shared pieces: path and operation names, prompts, answer scoring (exact match, F1, containment; numeric for math), BM25 retrieval over `data/passages.jsonl`, and `run_path()`, which runs one path on one request and scores it. |
| `routers.py` | The routers. `hand_features()` computes the 11 simple features. `R0Rules` is the original rules baseline. `R0bRules` is a stronger rules baseline: retrieve when there is no context and the request isn't math-like. `R1Classifier` uses a MiniLM embedding plus the hand features, with one balanced logistic regression per operation and thresholds tuned on validation (approval tuned for at least 95% recall). |

### Data and labels

| File | What it does |
| --- | --- |
| `build_dataset.py` | Builds `data/candidates.jsonl` and `data/passages.jsonl`. Sources: SQuAD with the passage (`squad_ctx`) and without it (`squad_noctx`), GSM8K, harder GSM8K (5 or more steps), and every `data/t3_items*.jsonl` file of risky actions with benign twins. |
| `label.py` | Runs fast, retrieve, strong and retrieve+strong on every request. The label is the cheapest path that answers correctly. Requests no path solves are marked `unsolvable`. Makes 60/20/20 train/val/test splits, keeping each risky action with its twin. Resumable via `data/label_runs.jsonl`. Output: `data/labeled.jsonl`. |
| `data/t3_items_claude.jsonl` | 75 risky actions with twins (5 categories × 15). Used in training and testing. |
| `data/external_t3_teammate.jsonl` | 75 independently written pairs. A held-out test set only: its name keeps it out of `build_dataset.py`. |

### Evaluation and analysis

| File | What it does | Output |
| --- | --- | --- |
| `evaluate.py` | Trains and tunes routers on train/val and scores them on test: always-fast, always-slow, random, R0, R0b, R1, cascade, hybrid, fixed-rule hybrid, retrieve-then-verify hybrid (`hybrid_seq`) and oracle. Reports accuracy, per-trigger recall, risky actions missed, wrongly flagged twins, latency and Gemini rate. | `results/summary.md`, `metrics.json`, `predictions.jsonl`, `curve_r1.csv/png` |
| `analyze.py` | Bootstrap 95% confidence intervals (5,000 resamples) and McNemar's exact test for paired comparisons. | `results/confidence.md` |
| `extra_analysis.py` | Worked examples, cost per 1,000 requests and dataset statistics. | `results/extra.md` |
| `cv.py` | 5-fold cross-validation: R1 feature ablation (embedding only, hand features only, both), R0b, risk by category, and with `--with-cascade` the cascade and all hybrids plus diagnostics. | `results/cv_summary.md` |
| `t3_external.py` | Trains routers on the original data and tests risk detection on the teammate's set: risky actions caught, twins flagged, AUC, per category, a threshold sweep and a sandbox-word check. | `results/t3_external.md` |

### Helpers

| File | What it does |
| --- | --- |
| `test_models.py` | Smoke test for both backends. |
| `build_pilot.py`, `run_pilot.py` | The small pilot run used to check scoring before full labeling. |
| `t3_pilot_TEMPLATE.jsonl` | Format template for writing new risky-action pairs. |

---

## The routers in one line each

- **R0 / R0b:** hand-written rules.
- **R1:** classifier on the request text. Decides before answering, in about 25 ms.
- **Cascade:** the local model answers k times at temperature 0.7. If the answers disagree, escalate to Gemini.
- **Hybrid:** R1 decides retrieval and approval; disagreement decides the strong model.
- **Hybrid (fixed rule):** the same split of work, escalating on any disagreement (no tuned threshold).
- **Hybrid-seq:** after retrieval, sample k answers with the retrieved passages, and escalate if they disagree or abstain.

---

## Data files (generated)

| File | Contents |
| --- | --- |
| `data/candidates.jsonl` | Unlabeled requests |
| `data/passages.jsonl` | Retrieval store |
| `data/label_runs.jsonl` | Every labeling run (path, answer, correct, tokens, timings) |
| `data/labeled.jsonl` | Requests with category, correct operations and split |
| `data/eval_runs.jsonl` | Extra runs made during evaluation (cascade and verify samples) |

## Do not commit

`cache/`, `logs/`, your API key, and `__pycache__/`. A minimal `.gitignore`:

```
cache/
logs/
__pycache__/
.env
```
