# Experiment log

Newest first. Every number here can be reproduced from the repo.

## 2026-11-18: starting point on the synthetic tracks

Settings unchanged from the earlier tuning (σ_a = 35 m/s², τ = 6 s, σ_r = 8 m). Mean position MAE over 20 seeds per scenario:

| Track | Estimator | Clean 5 Hz | S1 | S2 σ=0.5 | S2 σ=1 | S2 σ=2 | S2 σ=5 | S3 |
|---|---|---|---|---|---|---|---|---|
| fast | baseline | 0.120 | 4.61 | 5.54 | 6.66 | 9.36 | 18.52 | 9.79 |
| fast | Singer | 0.835 | 4.66 | 5.16 | 6.02 | 8.06 | 15.46 | 9.39 |
| slow | baseline | 0.024 | 0.95 | 2.15 | 3.65 | 6.75 | 17.14 | 6.57 |
| slow | Singer | 0.117 | 0.63 | 1.68 | 3.01 | 5.73 | 14.36 | 5.76 |

Mean over S1–S3, both tracks: baseline 7.64 m, Singer 6.66 m (−12.8 %). Singer wins every noisy scenario. It's slightly worse on fast S1 (no noise, so less smoothing is best) and on clean input.

## Next steps

1. **C++ port** (`cpp/`), verified equivalent to `python/estimator.py` within 1e-9 m on all scenarios.
2. **Re-tune** σ_a, τ and σ_r on the synthetic tracks with the rules in `docs/DESIGN.md` §6. This needs a sweep tool.
3. **Consistency:** scale Q, R and P₀ so the mean NIS is about 3; re-test the gate; publish the covariance.
4. **NEES tests:** the tracks have exact truth, so the filter's covariance can be checked against its real errors.
