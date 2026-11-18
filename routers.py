"""Routers. Each sees ONLY what a real router would see before any model
runs: the request text, the provided context, the pending tool call, and
the retrieval index. It never sees task_type, recipe, or gold answers.

Every router returns the same schema:
  {"path": "fast" | "slow", "ops": [...], "scores": {op: float}, "router_ms": float}

  R0  rules       hand-written thresholds; the floor every router must beat
  R1  classifier  sentence embedding + hand features, one logistic
                  regression per operation, thresholds tuned on validation
"""
import json
import re
import time

import numpy as np

from common import OPS, retrieve, tokenize

RISKY_VERBS = {
    "delete", "remove", "drop", "wipe", "erase", "purge", "truncate", "destroy",
    "transfer", "pay", "send", "wire", "refund", "charge", "withdraw",
    "grant", "revoke", "share", "export", "publish", "upload", "disable",
    "deactivate", "terminate", "cancel", "overwrite", "reset", "change", "promote",
}
BROAD_SCOPE = {"all", "every", "entire", "everyone", "production", "prod", "database",
               "customers", "users", "accounts", "public", "admin", "external", "permanently"}
QUESTION_WORDS = {"how", "many", "much", "total", "each", "per", "left", "remaining", "cost"}

# ------------------------------------------------------------- features ---
def hand_features(item):
    """Cheap, request-only features. Returns a dict of floats."""
    q = item["request"]
    ctx = item.get("context", "") or ""
    tool = item.get("tool_call")
    q_tok = tokenize(q)
    tool_text = json.dumps(tool) if tool else ""
    all_tok = set(q_tok) | set(tokenize(tool_text))

    content = [t for t in q_tok if len(t) > 3]
    ctx_set = set(tokenize(ctx))
    overlap = (sum(t in ctx_set for t in content) / len(content)) if (content and ctx) else 0.0

    _, scores = retrieve(q, k=2)
    numbers = re.findall(r"\d+(?:\.\d+)?", q)

    return {
        "has_context": float(bool(ctx)),
        "context_len": len(ctx_set) / 100.0,
        "q_ctx_overlap": overlap,
        "bm25_top1": scores[0] / 10.0 if scores else 0.0,
        "bm25_gap": (scores[0] - scores[1]) / 10.0 if len(scores) > 1 else 0.0,
        "q_len": len(q_tok) / 50.0,
        "n_numbers": len(numbers) / 5.0,
        "quant_words": sum(t in QUESTION_WORDS for t in q_tok) / 3.0,
        "has_tool": float(bool(tool)),
        "risky_verbs": float(sum(t in RISKY_VERBS for t in all_tok)),
        "broad_scope": float(sum(t in BROAD_SCOPE for t in all_tok)),
    }


def feature_vector(item):
    f = hand_features(item)
    return np.array([f[k] for k in sorted(f)], dtype=np.float32), f

# ------------------------------------------------------------ decisions ---
def decide(scores, thresholds):
    ops = [op for op in OPS if scores.get(op, 0.0) >= thresholds.get(op, 1.01)]
    path = "slow" if ops else "fast"
    return path, ops


class R0Rules:
    """retrieve: no context and the retrieval index has a strong match.
    strong:   no rule (the blueprint's floor has none for hard questions).
    approval: a tool call with any risky verb."""
    name = "R0"

    def __init__(self, bm25_threshold=10.0):
        self.bm25_threshold = bm25_threshold  # raw BM25 score of the best passage

    def scores(self, item):
        f = hand_features(item)
        raw_bm25 = f["bm25_top1"] * 10.0
        return {
            "retrieve": float((not f["has_context"]) and raw_bm25 >= self.bm25_threshold),
            "strong": 0.0,
            "approval": float(f["has_tool"] and f["risky_verbs"] > 0),
        }

    def tune(self, items):
        """Pick the BM25 threshold that best matches retrieve labels on validation."""
        raw = sorted(hand_features(it)["bm25_top1"] * 10.0 for it in items if not it.get("context"))
        grid = sorted(set([0.0] + [raw[int(q * (len(raw) - 1))] for q in np.linspace(0, 1, 21)])) if raw else [10.0]
        best = None
        for th in grid:
            self.bm25_threshold = th
            correct = sum(
                (self.scores(it)["retrieve"] >= 1) == ("retrieve" in it["correct_ops"]) for it in items)
            if best is None or correct > best[0]:
                best = (correct, th)
        self.bm25_threshold = best[1]
        return {"bm25_threshold": best[1]}

    def route(self, item):
        t0 = time.perf_counter()
        s = self.scores(item)
        path, ops = decide(s, {"retrieve": 1.0, "strong": 1.0, "approval": 1.0})
        return {"path": path, "ops": ops, "scores": s, "router_ms": (time.perf_counter() - t0) * 1000}


class R1Classifier:
    """Sentence embedding of request + tool call, plus hand features;
    one balanced logistic regression per operation."""
    name = "R1"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    _embedder = None          # shared across instances (loaded once)
    _cache = {}               # (kind, item id) -> vector; used by cross-validation

    def __init__(self, features="both", use_cache=False):
        """features: "both" (default), "embed" (sentence embedding only) or
        "hand" (the eleven hand features only), for the feature ablation.
        use_cache: reuse vectors across instances (cv.py); off for timed runs,
        so measured router latency includes computing the features."""
        from sentence_transformers import SentenceTransformer
        if R1Classifier._embedder is None:
            R1Classifier._embedder = SentenceTransformer(self.EMBED_MODEL, device="cpu")
        self.embedder = R1Classifier._embedder
        self.features, self.use_cache = features, use_cache
        self.models, self.thresholds = {}, {}

    def _embed(self, item):
        text = item["request"] + (" " + json.dumps(item["tool_call"]) if item.get("tool_call") else "")
        return self.embedder.encode([text], normalize_embeddings=True)[0]

    def _part(self, kind, item):
        key = (kind, item["id"])
        if self.use_cache and key in R1Classifier._cache:
            return R1Classifier._cache[key]
        v = self._embed(item) if kind == "embed" else feature_vector(item)[0]
        if self.use_cache:
            R1Classifier._cache[key] = v
        return v

    def _x(self, item):
        if self.features == "embed":
            return self._part("embed", item)
        if self.features == "hand":
            return self._part("hand", item)
        return np.concatenate([self._part("embed", item), self._part("hand", item)])

    def fit(self, items):
        from sklearn.linear_model import LogisticRegression
        X = np.stack([self._x(it) for it in items])
        for op in OPS:
            y = np.array([op in it["correct_ops"] for it in items], dtype=int)
            if y.sum() == 0 or y.sum() == len(y):
                self.models[op] = float(y.mean())  # constant: nothing to learn
                continue
            m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            m.fit(X, y)
            self.models[op] = m

    def scores(self, item):
        x = self._x(item).reshape(1, -1)
        out = {}
        for op, m in self.models.items():
            out[op] = float(m) if isinstance(m, float) else float(m.predict_proba(x)[0, 1])
        return out

    def tune(self, items, approval_recall=0.95):
        """retrieve/strong: threshold with the best F1 on validation.
        approval: the highest threshold that still catches >= 95% of risky
        items, because a missed risky action costs more than a false alarm."""
        cached = [(self.scores(it), it) for it in items]
        grid = np.linspace(0.05, 0.95, 19)
        for op in OPS:
            pos = [s[op] for s, it in cached if op in it["correct_ops"]]
            neg = [s[op] for s, it in cached if op not in it["correct_ops"]]
            if not pos:
                self.thresholds[op] = 0.5
                continue
            if op == "approval":
                ok = [t for t in grid if np.mean([p >= t for p in pos]) >= approval_recall]
                self.thresholds[op] = float(max(ok)) if ok else float(grid[0])
                continue
            best_t, best_f1 = 0.5, -1
            for t in grid:
                tp = sum(p >= t for p in pos)
                fp = sum(n >= t for n in neg)
                fn = len(pos) - tp
                f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                if f1 > best_f1:
                    best_t, best_f1 = float(t), f1
            self.thresholds[op] = best_t
        return dict(self.thresholds)

    def route(self, item, thresholds=None):
        t0 = time.perf_counter()
        s = self.scores(item)
        path, ops = decide(s, thresholds or self.thresholds)
        return {"path": path, "ops": ops, "scores": s, "router_ms": (time.perf_counter() - t0) * 1000}
