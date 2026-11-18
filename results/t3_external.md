# Risk detection on an independently written T3 set

Routers trained and tuned on the original data (train 494, val 165), then applied unchanged to `external_t3_teammate.jsonl`: 75 risky requests and 75 benign twins. Requests identical to labeled data: 0.

## Main result

| Router | Threshold | Risky caught | Twins wrongly flagged | AUC | Same router on original test split: caught / twins flagged / AUC |
| --- | --- | --- | --- | --- | --- |
| R0 (rules) | 1.00 | 39/75 | 38/75 | 0.51 | n/a |
| R1 (both) | 0.65 | 75/75 | 72/75 | 0.82 | 15/15 / 14/15 / 0.69 |
| R1 (embed only) | 0.55 | 75/75 | 72/75 | 0.83 | 15/15 / 14/15 / 0.71 |
| R1 (hand only) | 0.75 | 74/75 | 74/75 | 0.56 | 15/15 / 15/15 / 0.49 |

## By risk category (R1, both feature sets)

| Category | Risky caught | Twins wrongly flagged | AUC |
| --- | --- | --- | --- |
| access | 15/15 | 15/15 | 0.84 |
| data_deletion | 15/15 | 14/15 | 0.88 |
| data_out | 15/15 | 14/15 | 0.80 |
| money | 15/15 | 15/15 | 0.79 |
| system_config | 15/15 | 14/15 | 0.82 |

## Approval threshold sweep (R1, both)

| Threshold | Risky caught | Twins wrongly flagged |
| --- | --- | --- |
| 0.30 | 75/75 | 75/75 |
| 0.50 | 75/75 | 75/75 |
| 0.65 | 75/75 | 72/75 |
| 0.80 | 71/75 | 52/75 |
| 0.90 | 48/75 | 11/75 |
| 0.95 | 13/75 | 0/75 |

## Surface-cue check

Twins whose request contains a sandbox word (my, local, test, sandbox, dev, mock, dummy, staging, demo):
- with such a word: 66/69 wrongly flagged
- without: 6/6 wrongly flagged

## Missed risky requests (R1, both)

