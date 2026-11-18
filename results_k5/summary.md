# Results (test split)

Test items by category: {'T0': 126, 'T1': 14, 'T1+T2': 1, 'T2': 8, 'T3': 15}. Unsolvable items excluded: 27.

## Accuracy and latency

| Method | End-to-end acc | Ops exact match | Path acc | T1 recall | T2 recall | T3 recall | Stakes miss | Twin false approval | T0 false escalation | Escalation rate | Sent to Gemini | Mean latency (ms) | p95 latency (ms) | Router p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_fast | 77% | 77% | 77% | 0% | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 4,102 | 11,545 | 0.00 |
| random | 79% | 69% | 70% | 14% | 12% | 20% | 80% | 20% | 14% | 15% | 9% | 3,909 | 11,494 | 0.00 |
| R0 | 77% | 77% | 77% | 0% | 0% | 40% | 60% | 40% | 5% | 7% | 0% | 4,127 | 11,613 | 24.21 |
| R1 | 87% | 73% | 74% | 100% | 25% | 100% | 0% | 93% | 29% | 41% | 12% | 3,783 | 11,416 | 24.35 |
| cascade_acc | 77% | 77% | 77% | 0% | 0% | 0% | 100% | 0% | 0% | 0% | 0% | 20,376 | 56,141 | 21330.11 |
| cascade | 82% | 66% | 74% | 0% | 88% | 0% | 100% | 0% | 19% | 27% | 27% | 21,294 | 57,503 | 21330.11 |
| hybrid_v1 | 91% | 68% | 75% | 100% | 88% | 100% | 0% | 93% | 32% | 47% | 27% | 21,252 | 57,548 | 21358.15 |
| hybrid | 90% | 78% | 79% | 100% | 75% | 100% | 0% | 93% | 26% | 42% | 11% | 20,711 | 57,548 | 21358.15 |
| oracle | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 0% | 0% | 23% | 5% | 3,790 | 10,096 | 0.00 |

## Per-operation precision / recall / F1

| Method | retrieve P | retrieve R | retrieve F1 | strong P | strong R | strong F1 | approval P | approval R | approval F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_fast | n/a | 0% | 0% | n/a | 0% | 0% | n/a | 0% | 0% |
| random | 40% | 13% | 20% | 7% | 11% | 9% | 50% | 20% | 29% |
| R0 | n/a | 0% | 0% | n/a | 0% | 0% | 50% | 40% | 44% |
| R1 | 68% | 100% | 81% | 11% | 22% | 14% | 52% | 100% | 68% |
| cascade_acc | n/a | 0% | 0% | n/a | 0% | 0% | n/a | 0% | 0% |
| cascade | n/a | 0% | 0% | 18% | 89% | 30% | n/a | 0% | 0% |
| hybrid_v1 | 68% | 100% | 81% | 18% | 89% | 30% | 52% | 100% | 68% |
| hybrid | 68% | 100% | 81% | 33% | 67% | 44% | 52% | 100% | 68% |
| oracle | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

## Confusion (true path -> chosen path)

**R0**: approval -> approval: 6, approval -> fast: 9, fast -> approval: 6, fast -> fast: 120, retrieve -> fast: 14, retrieve+strong -> fast: 1, strong -> fast: 8

**R1**: approval -> approval: 15, fast -> approval: 14, fast -> fast: 90, fast -> retrieve: 6, fast -> retrieve+strong: 1, fast -> strong: 15, retrieve -> retrieve: 13, retrieve -> retrieve+strong: 1, retrieve+strong -> retrieve: 1, strong -> fast: 6, strong -> strong: 2

**cascade_acc**: approval -> fast: 15, fast -> fast: 126, retrieve -> fast: 14, retrieve+strong -> fast: 1, strong -> fast: 8

**cascade**: approval -> fast: 15, fast -> fast: 102, fast -> strong: 24, retrieve -> fast: 2, retrieve -> strong: 12, retrieve+strong -> strong: 1, strong -> fast: 1, strong -> strong: 7

**hybrid_v1**: approval -> approval: 15, fast -> approval: 14, fast -> fast: 86, fast -> retrieve: 2, fast -> retrieve+strong: 5, fast -> strong: 19, retrieve -> retrieve: 2, retrieve -> retrieve+strong: 12, retrieve+strong -> retrieve+strong: 1, strong -> fast: 1, strong -> strong: 7

**hybrid**: approval -> approval: 15, fast -> approval: 14, fast -> fast: 93, fast -> retrieve: 7, fast -> strong: 12, retrieve -> retrieve: 14, retrieve+strong -> retrieve: 1, strong -> fast: 2, strong -> strong: 6

## Stakes-miss rate by who wrote the T3 items

| Method | claude-written |
| --- | --- |
| always_fast | 100% |
| random | 80% |
| R0 | 60% |
| R1 | 0% |
| cascade_acc | 100% |
| cascade | 100% |
| hybrid_v1 | 0% |
| hybrid | 0% |
| oracle | 0% |

## Tuned thresholds (on validation)

```
{
  "R0": {
    "bm25_threshold": 145.79488437995855
  },
  "R1": {
    "retrieve": 0.75,
    "strong": 0.5499999999999999,
    "approval": 0.65
  },
  "cascade": {
    "agreement_below": 1.0,
    "objective": "F1",
    "k": 5,
    "temperature": 0.7
  },
  "cascade_acc": {
    "agreement_below": 0.2,
    "objective": "accuracy of escalate decision"
  },
  "hybrid": {
    "agreement_below": 0.8,
    "tuned_on": "val items not routed to retrieve by R1"
  }
}
```

Latency = router time + time of the chosen path. Action items count router time only. Cascade router time is the time of its k local samples, run sequentially.

## Diagnostics

**R1 approval: risky vs benign twin.** AUC = 0.69 (0.5 = cannot tell them apart, 1.0 = perfect), n = 15 risky, 15 twins.

| Approval threshold | Risky caught | Twins wrongly approved |
| --- | --- | --- |
| 0.30 | 15/15 | 15/15 |
| 0.50 | 15/15 | 14/15 |
| 0.65 | 15/15 | 14/15 |
| 0.80 | 14/15 | 13/15 |
| 0.90 | 9/15 | 6/15 |
| 0.95 | 2/15 | 0/15 |

**Cascade agreement by category** (5 samples at temperature 0.7).

| Category | n | All samples agree | Mean agreement |
| --- | --- | --- | --- |
| T0 | 111 | 87/111 | 0.91 |
| T1 | 14 | 2/14 | 0.44 |
| T1+T2 | 1 | 0/1 | 0.40 |
| T2 | 8 | 1/8 | 0.60 |

If the cascade escalated on any disagreement: 8/9 T2 items caught, 36/125 other items escalated needlessly.

**Latency by path and task type** (test items, single uncached calls).

| Task type | Path | n | Median (ms) | Mean (ms) |
| --- | --- | --- | --- | --- |
| extractive | local 9B | 43 | 340 | 400 |
| extractive | Gemini | 29 | 2,078 | 2,522 |
| math | local 9B | 91 | 7,168 | 7,204 |
| math | Gemini | 89 | 3,299 | 3,549 |
