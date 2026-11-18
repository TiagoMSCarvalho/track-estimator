# track-estimator

An online estimator that turns intermittent, noisy 3-D position fixes (0.5–5 Hz) into position and velocity at a fixed 50 Hz output rate. It's built the way flight or vehicle software has to be: causal, deterministic, bounded in memory, and fast.

The estimator is a **Singer Kalman filter**, one per axis, with a chi-square outlier gate. A C++ port is in progress: the Python version is the reference it must match.

## Run it

Python 3.10+; the estimator and runner use the standard library only. The tests also need `pytest` and `numpy`.

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q                                   # all tests
python3 python/runner.py --input data/fast.csv         # 50 Hz output on stdout
python3 tools/degrade.py --out results --seeds 20      # degraded scenarios
python3 tools/scoring.py --truth data/fast.csv --input results/fast_S3_seed00.csv
python3 tools/scoring.py --estimator baseline --truth data/fast.csv --input results/fast_S3_seed00.csv
```

## How it works

- **Motion model:** each axis keeps position, velocity and acceleration. Acceleration fades with a time constant τ (Singer model), so a turn carries through a short gap without running away over a long one.
- **Any gap:** the transition F(dt) and process noise Q(dt) are exact closed forms, so irregular timing needs no fixed step.
- **Outliers:** a NIS χ²(3) gate at 99.9 % (16.27). Three rejections in a row restart the filter on the new position, so it can't lock out.
- **Long gaps:** the filter restarts after a gap over 5 s.
- **Output:** the filter changes only when a measurement arrives. `predict` is read-only and extrapolates to the query time.
- **Numerics:** Joseph-form covariance update, covariance stored as its 6 unique numbers per axis, finite checks on every output.

Design decisions and their alternatives are in [`docs/DESIGN.md`](docs/DESIGN.md); the experiment log is in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Evidence

Two synthetic truth tracks (`tools/trajectories.py`): **fast** (48 m/s, turns, weaves, climbs; up to 17 m/s²) and **slow** (23 m/s, gentle manoeuvres). Each is degraded into scenarios with random 0.2–2 s gaps, Gaussian noise of 0–5 m per axis, and about 3 % outliers of 20–50 m, 20 seeds each.

| Estimator | Mean position MAE over all degraded scenarios, both tracks |
|---|---|
| Two-point baseline | 7.64 m |
| Singer filter | **6.66 m** (13 % lower) |

The filter wins in every noisy scenario. On clean 5 Hz input the baseline is better, because almost no smoothing is needed there. The filter's settings come from an earlier tuning and are being re-checked on these tracks.

## Layout

| Path | What it is |
|---|---|
| `python/estimator.py` | The Singer filter (the reference implementation) |
| `python/baseline.py` | Two-point constant-velocity baseline |
| `python/runner.py` | Replays a measurement file at 50 Hz and writes the output CSV |
| `tools/trajectories.py` | Generates the truth tracks in `data/` |
| `tools/degrade.py` | Writes degraded scenarios with fixed seeds |
| `tools/scoring.py` | MAE, p95, worst tick and velocity RMSE against the truth |
| `tests/` | Filter maths (checked against Van Loan), behaviour, runner, scorer, tracks |
| `cpp/` | The C++ port (in progress) |

## Status

- [x] Python reference implementation and test suite
- [ ] Re-tune the settings on the synthetic tracks
- [ ] Consistency: scale Q and R so the mean NIS matches its expected value; publish the covariance
- [ ] C++ port, verified equivalent to Python within 1e-9 m
