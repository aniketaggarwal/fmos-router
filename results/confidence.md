# Confidence intervals (bootstrap, 95%)

164 test items, 5000 resamples.

| Method | Accuracy | 95% CI | Mean latency (s) | 95% CI | Sent to Gemini | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| always_fast | 76.8% | 70.1–82.9% | 4.1 | 3.5–4.7 | 0.0% | 0.0–0.0% |
| random | 78.7% | 72.0–84.8% | 3.9 | 3.3–4.5 | 8.5% | 4.9–12.8% |
| R0 | 76.8% | 70.1–82.9% | 4.1 | 3.5–4.7 | 0.0% | 0.0–0.0% |
| R1 | 87.2% | 81.7–92.1% | 3.8 | 3.2–4.4 | 11.6% | 6.7–16.5% |
| cascade_acc | 74.4% | 67.7–80.5% | 12.0 | 10.2–13.9 | 0.0% | 0.0–0.0% |
| cascade | 82.3% | 76.2–87.8% | 12.9 | 11.0–14.7 | 25.0% | 18.9–31.7% |
| hybrid_v1 | 90.9% | 86.0–95.1% | 12.8 | 10.9–14.7 | 25.0% | 18.9–31.7% |
| hybrid | 90.2% | 85.4–94.5% | 12.6 | 10.6–14.5 | 14.0% | 9.1–19.5% |
| oracle | 100.0% | 100.0–100.0% | 3.8 | 3.2–4.4 | 5.5% | 2.4–9.1% |

## Paired differences

| Comparison | Accuracy difference | 95% CI | First is more accurate in | Latency difference (s) | Gemini-rate difference |
| --- | --- | --- | --- | --- | --- |
| hybrid vs R1 | +3.0 pts | +0.0 to +6.1 | 97% of resamples | +8.8 | +2.4 pts |
| hybrid vs cascade | +7.9 pts | +0.6 to +15.2 | 98% of resamples | -0.3 | -11.0 pts |
| R1 vs always_fast | +10.4 pts | +2.4 to +18.3 | 100% of resamples | -0.3 | +11.6 pts |
| hybrid vs always_fast | +13.4 pts | +5.5 to +22.0 | 100% of resamples | +8.5 | +14.0 pts |
| cascade vs always_fast | +5.5 pts | +1.2 to +9.8 | 99% of resamples | +8.8 | +25.0 pts |
| hybrid vs hybrid_v1 | -0.6 pts | -1.8 to +0.0 | 0% of resamples | -0.3 | -11.0 pts |

A difference is reliable when its 95% CI excludes 0 (equivalently, the first method wins in about 97.5% or more of resamples).