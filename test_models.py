"""Smoke test for models.py. Run from the project folder: python test_models.py"""
from models import generate, warmup

QUESTIONS = [
    "What is the capital of Australia? Answer in one word.",
    "What is 17 * 23? Answer with the number only.",
    "Who wrote 'Pride and Prejudice'? Answer with the name only.",
    "What is the chemical symbol for gold? Answer with the symbol only.",
    "How many sides does a hexagon have? Answer with the number only.",
]


def row(r):
    flags = []
    if r["cold"]:
        flags.append("COLD")
    if r["cached"]:
        flags.append("CACHED")
    text = r["text"].strip().replace("\n", " ")[:40]
    return (f"{r['backend']:<7} {text:<42} out={r['output_tokens']:<4} "
            f"think={r['thinking_tokens']:<5} wall={r['wall_ms']:>9.1f} ms  {' '.join(flags)}")


print("Warming up the local model...")
warmup()

print("\n== Local: 5 questions, temperature 0 ==")
for q in QUESTIONS:
    print(row(generate(q, backend="local", tag="smoke")[0]))

print("\n== Local: 5 samples, temperature 0.7 (answers may differ) ==")
for r in generate("Name one fruit that is red. One word.", backend="local",
                  temperature=0.7, n=5, tag="smoke-samples"):
    print(row(r))

print("\n== Gemini: 5 questions ==")
for q in QUESTIONS:
    print(row(generate(q, backend="gemini", tag="smoke")[0]))

print("\n== Gemini cache check: repeat question 1 (should say CACHED) ==")
print(row(generate(QUESTIONS[0], backend="gemini", tag="smoke-cache")[0]))