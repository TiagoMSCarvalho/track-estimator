# Design

Requirements, decisions and the rules behind them. Results are logged in `docs/PROGRESS.md`.

## 1. Objective

Turn intermittent, noisy 3-D position fixes (0.5–5 Hz, gaps of 0.2–2 s) into position and velocity at 50 Hz, with the lowest position error, behaving like deployable software: causal, deterministic, bounded and fast. Keep it simple, interpretable and explainable in a few minutes.

## 2. Requirements

| # | Requirement | Check |
|---|---|---|
| R1 | Interface: `reset()`, `ingest_measurement(t, x, y, z) -> bool`, `predict(t) -> TargetState` | Tests |
| R2 | `predict` returns position **and** velocity at the query time | Tests |
| R3 | Accepts irregular input, gaps 0.2–2 s | Tests, scenarios |
| R4 | Every output finite, including after one measurement, a long gap or a rejected outlier | Tests |
| R5 | Deterministic: same input gives byte-identical output | Test across processes |
| R6 | Bounded memory | By design: fixed-size state |
| R7 | One configuration for every track | By design |
| R8 | Far inside a 20 ms (50 Hz) cycle | Test: p99 call time < 5 ms; typically tens of µs |
| R9 | `ingest_measurement` returns `True` only when the measurement was used | Tests |
| R10 | Estimator core uses the standard library only | By design (portability, a direct C++ port) |

**Runner contract:** 50 Hz ticks from the first to the last input timestamp. At each tick, ingest every measurement with timestamp ≤ tick + 1e-12, then call `predict(tick)` once. Input files must be finite with strictly increasing timestamps.

## 3. Constraints → consequences

| Constraint | Rules out | Pushes to |
|---|---|---|
| Causal | Smoothers, centred averages | A forward filter |
| Interpretable, light | Neural networks, Gaussian processes, large particle filters | A model-based filter with physical parameters |
| Bounded memory, O(1) per call | Growing windows, refitting history | A recursive filter |
| Deterministic | Random sampling, wall-clock reads, iterative solvers | Closed-form updates |
| Fast over any gap | Stepping a 2 s gap at 50 Hz | Closed-form F(dt), Q(dt), scalar maths per axis |
| Simple | Anything not explainable in 2 minutes | Add complexity only when a test shows the simpler version fails |

## 4. Decisions

| # | Decision | Chosen | Rejected | What would change it |
|---|---|---|---|---|
| D1 | Estimator family | Linear Kalman filter | α-β(-γ), sliding-window least squares, EKF/UKF, particle filter, learned models | Range/bearing measurements → EKF or UKF |
| D2 | Structure | 3 independent axes | One coupled 9-state filter | Rotated, anisotropic measurement noise or a coordinated-turn model |
| D3 | Motion model | Singer: state [p, v, a], acceleration fades with τ | Constant velocity (lags in turns); constant acceleration (no bound over a 2 s gap); IMM (more complexity for a small gain in earlier tests) | An IMM search that clears the adoption rule in §6 |
| D4 | Process noise | Exact Q(dt) of the Singer model, closed form, checked against Van Loan | Discrete white-noise acceleration, a fixed Q | — |
| D5 | σ_a, τ | σ_a = 35 m/s² on x and y, τ = 6 s; z uses σ_a × √0.2 (vertical motion assumed gentler) | Per-track settings | Re-tuning on the synthetic tracks (pending) |
| D6 | Measurement noise R | Fixed σ_r = 8 m, a tuning value (only the ratio of Q to R sets the gain) | Online R adaptation (tested earlier: not worth its cost) | Known sensor noise |
| D7 | Outliers | NIS χ²(3) gate at 99.9 % (16.27); accept and restart after 3 rejections in a row | Distance gate, Huber weighting | Heavy-tailed noise → Huber; many rejections on clean data → loosen |
| D8 | Time handling | Event-driven update; `predict(t)` = F(t − t_last)·x, read-only | Propagating the state at 50 Hz | A non-linear model |
| D9 | Initialisation | First fix: p = z, v = 0, a = 0, wide variances (velocity σ = 60 m/s) | Two-point differencing | — |
| D10 | Long gaps | Coast; restart if the gap exceeds 5 s | Holding the last value; unbounded extrapolation | Longer gaps → publish a track-quality output |
| D11 | Numerics | Joseph form; P stored as 6 numbers per axis (always symmetric); finite check in `predict`, which holds the state for a query behind it | Short-form update; variance floors | — |

**Known limitation:** because σ_r is a gain knob larger than the real noise, the filter's covariance is conservative: the mean NIS sits well below its expected value of 3, and the gate only catches gross errors. Fix: scale Q, R and P₀ together until the mean NIS is about 3. The gain, and so every estimate, stays the same, and the gate becomes meaningful. Then re-test, because a tighter gate changes which measurements are rejected.

## 5. Evaluation

**Truth:** two synthetic tracks from `tools/trajectories.py`, with exact velocity: *fast* (48 m/s, turns up to 20°/s, weaves, ±6 m/s climbs) and *slow* (23 m/s, gentle).

**Scenarios** (`tools/degrade.py`, 20 seeds each, string seeds so they don't depend on `PYTHONHASHSEED`):

| ID | Timing | Noise σ per axis | Outliers | Purpose |
|---|---|---|---|---|
| clean | 5 Hz | 0 | — | Sanity check |
| S1 | Uniform 0.2–2 s | 0 | — | Prediction error only |
| S2 | Uniform 0.2–2 s | 0.5, 1, 2, 5 m | — | Noise robustness |
| S3 | Uniform 0.2–2 s | 1 m | about 3 %, 20–50 m | Gating |

This is an envelope built from the problem's timing, not a sensor model. Not modelled: range-dependent anisotropic noise, time-correlated noise, timestamp jitter, out-of-order measurements, heavy tails.

**Metrics:** position MAE, p95 (nearest rank), worst tick, velocity RMSE, mean NIS, gate rejection rate, call time. Scoring starts at the second measurement.

## 6. Decision rules

- **Model ladder:** baseline → constant velocity → Singer → IMM. Adopt a step only if the mean MAE over S1–S3 (both tracks) drops by **at least 10 %** and **no scenario worsens by more than 5 %**. Otherwise keep the simpler model.
- **Parameters:** among settings within 3 % of the best, take the most conservative (smallest gain). Report velocity RMSE alongside, since a higher gain passes more noise into velocity.
- **Held-out check:** choose settings on seeds 0–19, confirm on seeds 20–39.
- **Consistency:** mean NIS ≈ 3. Far above means P is overconfident and the gate rejects real manoeuvres; far below means P is too wide and the gate is weak.

## 7. Tests

| File | What it guarantees |
|---|---|
| `tests/test_kalman.py` | F and Q against Van Loan; covariance propagation against the full matrix product; two short steps equal one long step; τ limits (CA and CV); restart variances; a hand-worked update |
| `tests/test_estimator.py` | Finite output; bad input refused; reset; straight line tracked exactly; outlier rejected; gate pinned between 50 and 55 m; third rejection accepted; gap restart; velocity from two fixes; query behind the state; velocity noise halved against the baseline; determinism; call time |
| `tests/test_runner_scoring.py` | 50 Hz ticks; no future data used; bad files refused; interpolation; a 10 m shift scores about 10 m |
| `tests/test_trajectories.py` | Committed tracks match the generator; positions agree with velocities; constant speed |
| `tests/test_degrade.py` | Same seed gives the same file; gaps inside 0.2–2 s; noise and outliers match their labels |
