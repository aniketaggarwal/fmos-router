"""Model wrapper for the FMOS trap-router project.

One entry point, generate(), for both the local fast model (Ollama) and the
slow model (Gemini). Every call is timed and appended to logs/calls.jsonl.

Fixed setup (change only on purpose, and note it in the write-up):
  fast path : qwen3.5:9b via Ollama, thinking off, num_ctx 6144
  slow path : gemini-3.8-flash, thinking level "low"
"""
import hashlib
import json
import random
import re
import time
from pathlib import Path

import requests
from google import genai
from google.genai import errors, types

# ---------------------------------------------------------------- config ---
OLLAMA_URL = "http://localhost:11434/api/chat"
LOCAL_MODEL = "qwen3.5:9b"
LOCAL_NUM_CTX = 6144          # largest context that stays 100% on the GPU
LOCAL_KEEP_ALIVE = "60m"      # keep the model loaded between calls

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_THINKING_LEVEL = "low"  # "low" | "medium" | "high"
# Thinking tokens may count against max_output_tokens, so Gemini gets this
# extra room on top of the answer budget. Watch finish_reason for MAX_TOKENS.
GEMINI_THINKING_ALLOWANCE = 2048
# Free tier allows 5 requests/minute for gemini-3.8-flash, so space real API
# calls at least this far apart. Lower it once you are on a paid key.
GEMINI_MIN_INTERVAL_S = 13.0
GEMINI_MAX_RETRIES = 8
# Hard safety cap on REAL Gemini API calls across all runs (cached answers do
# not count). The count is read from logs/calls.jsonl, so it survives restarts.
# Raise it on purpose if you need more.
GEMINI_CALL_BUDGET = 400

DEFAULT_TEMPERATURE = {"local": 0.0, "gemini": 1.0}  # Google recommends 1.0 for Gemini 3
COLD_LOAD_MS = 500            # a local call with load time above this is "cold"

LOG_PATH = Path("logs/calls.jsonl")
CACHE_DIR = Path("cache/gemini")

_gemini_client = None
_last_gemini_call = 0.0  # perf_counter time of the last real Gemini request
_gemini_calls_used = None  # loaded lazily from the log


def gemini_calls_used():
    """Real (non-cached) Gemini API calls made so far, from logs/calls.jsonl."""
    global _gemini_calls_used
    if _gemini_calls_used is None:
        n = 0
        if LOG_PATH.exists():
            with LOG_PATH.open(encoding="utf-8") as f:
                for line in f:
                    if '"backend": "gemini"' in line and '"cached": false' in line:
                        n += 1
        _gemini_calls_used = n
    return _gemini_calls_used


# --------------------------------------------------------------- helpers ---
def _ns_to_ms(ns):
    """Ollama reports nanoseconds; this is the only place units are converted."""
    return round((ns or 0) / 1e6, 2)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()  # reads GEMINI_API_KEY from the environment
    return _gemini_client


def _throttle_gemini():
    """Wait so real Gemini calls stay under the per-minute limit."""
    global _last_gemini_call
    gap = time.perf_counter() - _last_gemini_call
    if gap < GEMINI_MIN_INTERVAL_S:
        time.sleep(GEMINI_MIN_INTERVAL_S - gap)
    _last_gemini_call = time.perf_counter()


def _server_retry_delay(err):
    """Read the 'Please retry in 29.4s' hint from a 429 message, if present."""
    m = re.search(r"retry in ([\d.]+)s", str(err))
    return float(m.group(1)) if m else None


def _log(record):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------ local model ---
def _call_local(prompt, system, temperature, max_tokens, seed):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"temperature": temperature, "num_ctx": LOCAL_NUM_CTX, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed

    body = {
        "model": LOCAL_MODEL,
        "messages": messages,
        "think": False,
        "stream": False,
        "keep_alive": LOCAL_KEEP_ALIVE,
        "options": options,
    }

    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=body, timeout=300)
    wall_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()

    load_ms = _ns_to_ms(data.get("load_duration"))
    return {
        "backend": "local",
        "model": LOCAL_MODEL,
        "text": data["message"]["content"],
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "cached_prompt_tokens": data.get("prompt_eval_cached_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "thinking_tokens": 0,
        "wall_ms": round(wall_ms, 2),
        "load_ms": load_ms,
        "prompt_eval_ms": _ns_to_ms(data.get("prompt_eval_duration")),
        "eval_ms": _ns_to_ms(data.get("eval_duration")),
        "cold": load_ms > COLD_LOAD_MS,
        "cached": False,
        "finish_reason": data.get("done_reason"),
        "retries": 0,
    }


# ----------------------------------------------------------- gemini model ---
def _call_gemini(prompt, system, temperature, max_tokens, seed, sample_index,
                 use_cache=True, max_retries=GEMINI_MAX_RETRIES):
    # Everything that changes the output is part of the cache key, including
    # sample_index, so n samples are n separate cache entries.
    key = _sha(json.dumps({
        "model": GEMINI_MODEL,
        "thinking": GEMINI_THINKING_LEVEL,
        "allowance": GEMINI_THINKING_ALLOWANCE,
        "system": system,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "sample_index": sample_index,
    }, sort_keys=True, ensure_ascii=False))
    path = CACHE_DIR / f"{key}.json"

    if use_cache and path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        result["cached"] = True  # timings are from the original API call
        return result

    global _gemini_calls_used
    if gemini_calls_used() >= GEMINI_CALL_BUDGET:
        raise RuntimeError(
            f"Gemini call budget reached ({GEMINI_CALL_BUDGET} real calls). Cached results are kept. "
            "Raise GEMINI_CALL_BUDGET in models.py if you want to continue.")

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens + GEMINI_THINKING_ALLOWANCE,
        seed=seed,
        thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
    )
    client = _get_gemini()

    for attempt in range(max_retries):
        _throttle_gemini()
        t0 = time.perf_counter()
        try:
            r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                # The daily quota will not come back in seconds; stop cleanly.
                # Everything already answered is cached, so a rerun resumes.
                raise RuntimeError(
                    f"Gemini daily quota exhausted for {GEMINI_MODEL}. Cached results are kept; "
                    "rerun after the quota resets (midnight Pacific) or with a paid key.") from e
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                wait = min(60, 2 ** attempt) + random.uniform(0, 1)
                hint = _server_retry_delay(e) if e.code == 429 else None
                if hint is not None:
                    wait = max(wait, hint + 1)  # never retry sooner than the server asks
                print(f"[gemini] HTTP {e.code}, retry {attempt + 1} in {wait:.1f}s")
                time.sleep(wait)  # waiting time is NOT counted in wall_ms
                continue
            raise
        wall_ms = (time.perf_counter() - t0) * 1000  # successful attempt only
        _gemini_calls_used += 1

        u = r.usage_metadata
        result = {
            "backend": "gemini",
            "model": GEMINI_MODEL,
            "thinking_level": GEMINI_THINKING_LEVEL,
            "text": r.text or "",
            "prompt_tokens": (u.prompt_token_count or 0) if u else 0,
            "cached_prompt_tokens": (u.cached_content_token_count or 0) if u else 0,
            "output_tokens": (u.candidates_token_count or 0) if u else 0,
            "thinking_tokens": (u.thoughts_token_count or 0) if u else 0,
            "wall_ms": round(wall_ms, 2),
            "load_ms": None,
            "prompt_eval_ms": None,
            "eval_ms": None,
            "cold": False,
            "cached": False,
            "finish_reason": str(r.candidates[0].finish_reason) if r.candidates else None,
            "retries": attempt,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result


# ---------------------------------------------------------------- public ---
def generate(prompt, *, backend="local", system=None, temperature=None, n=1,
             max_tokens=256, seed=None, tag=None, log=True, use_cache=True):
    """Call a model n times and return a list of result dicts.

    temperature=None uses the backend default (local 0.0, gemini 1.0).
    seed, if given, becomes seed + i for sample i, so samples differ.
    tag is a free-form label saved in the log (e.g. "pilot:t1-0003").
    """
    if backend not in ("local", "gemini"):
        raise ValueError(f"unknown backend: {backend}")
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE[backend]

    results = []
    for i in range(n):
        s = None if seed is None else seed + i
        if backend == "local":
            r = _call_local(prompt, system, temperature, max_tokens, s)
        else:
            r = _call_gemini(prompt, system, temperature, max_tokens, s, i, use_cache=use_cache)
        r.update({
            "ts": time.time(),
            "tag": tag,
            "sample_index": i,
            "n": n,
            "temperature": temperature,
            "seed": s,
            "max_tokens": max_tokens,
            "prompt_sha": _sha((system or "") + "\n" + prompt)[:12],
        })
        if log:
            _log(r)
        results.append(r)
    return results


def warmup(rounds=2):
    """Load the local model and settle the GPU before timing anything."""
    for _ in range(rounds):
        generate("Reply with the single word: ready", backend="local", max_tokens=5, log=False)
