#!/usr/bin/env python3
"""Score an estimator's output on an input file against a clean truth track.

    python3 tools/scoring.py --truth data/fast.csv --input results/fast_S3_seed00.csv [more.csv ...]
                             [--estimator singer|baseline]

The input can be the truth itself (clean 5 Hz) or a degraded copy from tools/degrade.py.
Truth position and velocity are linearly interpolated onto the runner's 50 Hz ticks.
Scoring starts at the second input measurement, the first time velocity can be known.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import runner  # noqa: E402

Vec = tuple[float, float, float]


@dataclass(frozen=True)
class Point:
    time_s: float
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class Result:
    mae_m: float  # mean 3-D position error
    p95_m: float  # 95th percentile, nearest rank
    max_m: float  # worst single tick
    vel_rmse_mps: float  # 3-D velocity RMSE
    n: int  # scored ticks


def load_recording(path: Path) -> list[Point]:
    """Positions of a track or measurement file (extra columns ignored)."""
    return [Point(*event) for event in runner.load_events(path)]


def load_velocities(path: Path) -> list[Point]:
    """The truth file's exact velocity columns, as Points (x, y, z hold vx, vy, vz)."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [Point(float(r["timestamp_s"]), float(r["vx_mps"]), float(r["vy_mps"]), float(r["vz_mps"]))
                for r in csv.DictReader(handle)]


def interpolate(recording: Sequence[Point], times: Sequence[float], time_s: float) -> Point:
    """Straight-line interpolation between the two samples around time_s; held at the ends."""
    if time_s <= times[0]:
        return Point(time_s, recording[0].x_m, recording[0].y_m, recording[0].z_m)
    if time_s >= times[-1]:
        return Point(time_s, recording[-1].x_m, recording[-1].y_m, recording[-1].z_m)
    i = bisect.bisect_right(times, time_s)
    a, b = recording[i - 1], recording[i]
    w = (time_s - a.time_s) / (b.time_s - a.time_s)
    return Point(time_s, a.x_m + w * (b.x_m - a.x_m), a.y_m + w * (b.y_m - a.y_m), a.z_m + w * (b.z_m - a.z_m))


def score(estimator_name: str, truth_csv: Path, input_csv: Path) -> Result:
    truth, velocities = load_recording(truth_csv), load_velocities(truth_csv)
    truth_times = [p.time_s for p in truth]
    events = runner.load_events(input_csv)
    states = runner.run(runner.ESTIMATORS[estimator_name](), events)
    score_start_s = events[1][0] if len(events) > 1 else math.inf

    position_errors, velocity_sq = [], []
    for s in states:
        if s.timestamp_s + runner.EVENT_TOLERANCE_S < score_start_s:
            continue
        p = interpolate(truth, truth_times, s.timestamp_s)
        v = interpolate(velocities, truth_times, s.timestamp_s)
        position_errors.append(math.dist(s.position_m, (p.x_m, p.y_m, p.z_m)))
        velocity_sq.append(math.dist(s.velocity_mps, (v.x_m, v.y_m, v.z_m)) ** 2)
    if not position_errors:
        raise ValueError(f"{input_csv}: fewer than two measurements, nothing to score")
    ranked = sorted(position_errors)
    return Result(
        statistics.fmean(ranked),
        ranked[math.ceil(0.95 * len(ranked)) - 1],
        ranked[-1],
        math.sqrt(statistics.fmean(velocity_sq)),
        len(ranked),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--truth", required=True, type=Path, help="clean track with velocity columns")
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--estimator", choices=sorted(runner.ESTIMATORS), default="singer")
    args = parser.parse_args()

    try:
        results = [score(args.estimator, args.truth, path) for path in args.input]
    except ValueError as error:
        print(f"scoring: {error}", file=sys.stderr)
        return 1
    print(f"{'input':<32} {'MAE m':>8} {'p95 m':>8} {'max m':>8} {'vRMSE m/s':>10}")
    for path, r in zip(args.input, results):
        print(f"{path.name:<32} {r.mae_m:8.3f} {r.p95_m:8.3f} {r.max_m:8.3f} {r.vel_rmse_mps:10.3f}")
    if len(results) > 1:
        def mean(field: str) -> float:
            return statistics.fmean(getattr(r, field) for r in results)
        print(f"{'mean of ' + str(len(results)):<32} {mean('mae_m'):8.3f} {mean('p95_m'):8.3f} "
              f"{mean('max_m'):8.3f} {mean('vel_rmse_mps'):10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
