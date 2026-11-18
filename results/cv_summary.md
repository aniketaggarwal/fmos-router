# Cross-validated routing (all labeled requests)

823 requests in 5 folds (stratified by type, risky actions kept with their twins). Every request gets one out-of-fold decision. Routing metrics only: see the module docstring.

| Router | T1 to retrieval | T2 to strong model | T1+T2 both | T3 to approval | T1 sent to strong model | T0 answerable escalated | Twins approved | Ops exact match | Approval AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 (both) | 71/71 (100%) | 13/43 (30%) | 0/4 (0%) | 71/75 (95%) | 6/71 (8%) | 164/555 (30%) | 64/75 (85%) | 67% | 0.79 |
| R1 (embed) | 60/71 (85%) | 10/43 (23%) | 0/4 (0%) | 73/75 (97%) | 4/71 (6%) | 218/555 (39%) | 63/75 (84%) | 59% | 0.80 |
| R1 (hand) | 70/71 (99%) | 12/43 (28%) | 1/4 (25%) | 70/75 (93%) | 8/71 (11%) | 152/555 (27%) | 71/75 (95%) | 67% | 0.50 |
| Cascade (k=3) | 0/71 (0%) | 34/43 (79%) | 0/4 (0%) | 0/75 (0%) | 64/71 (90%) | 90/555 (16%) | 0/75 (0%) | 70% | n/a |
| Hybrid (k=3) | 71/71 (100%) | 18/43 (42%) | 0/4 (0%) | 71/75 (95%) | 0/71 (0%) | 56/555 (10%) | 64/75 (85%) | 81% | n/a |
| Hybrid, fixed rule: any disagreement (k=3) | 71/71 (100%) | 30/43 (70%) | 0/4 (0%) | 71/75 (95%) | 0/71 (0%) | 103/555 (19%) | 64/75 (85%) | 77% | n/a |

R1 (both) = the main router; (embed) uses only the sentence embedding; (hand) only the eleven hand features.

## Risk detection by category (R1, both feature sets, out of fold)

| Risk category | Risky caught | Twins approved | AUC risky vs twin |
| --- | --- | --- | --- |
| access | 15/15 | 12/15 | 0.94 |
| data_deletion | 15/15 | 14/15 | 0.85 |
| data_out | 12/15 | 13/15 | 0.59 |
| money | 14/15 | 15/15 | 0.76 |
| system_config | 15/15 | 10/15 | 0.92 |

## Hybrid diagnostic

Agreement thresholds per fold (escalate when agreement is below):

| Fold | Cascade | Hybrid |
| --- | --- | --- |
| 0 | 1.00 | 0.67 |
| 1 | 1.00 | 0.67 |
| 2 | 1.00 | 1.00 |
| 3 | 1.00 | 0.67 |
| 4 | 1.00 | 0.67 |

T2 requests the hybrid did not send to the strong model, by reason:

- samples partly disagreed, but not enough for the tuned threshold: 12
- all samples agreed (same wrong answer every time): 9
- R1 chose retrieval (extractive), so no samples were drawn: 8
- Of these, caught by the cascade: 20