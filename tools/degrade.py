#!/usr/bin/env python3
"""
Write degraded copies of the clean tracks: a measurement envelope, not a sensor model.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from scoring import Point, interpolate, load_recording

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "data" / "fast.csv", ROOT / "data" / "slow.csv")
GAP_S = (0.2, 2.0)  # 0.5-5 Hz updates
OUTLIER_M = (20.0, 50.0)
SCENARIOS = {  # name: (noise sigma per axis in m, outlier probability); S1 is S2 at sigma = 0
    "S1": (0.0, 0.0),
    "S2-0.5": (0.5, 0.0),
    "S2-1": (1.0, 0.0),
    "S2-2": (2.0, 0.0),
    "S2-5": (5.0, 0.0),
    "S3": (1.0, 0.03),
}

"""Takes the first truth timestamp and repeats the steps:
       1. Sample the truth: the true position at t, by straight-line interpolation of the track.
       2. Add Gaussian noise per axis, independently, with the scenario's sigma.
       3. With the scenario's probability, add an outlier in a random direction.
       4. Save the noisy measurement.
       5. Move time on by a random gap between 0.2 and 2.0 s."""
def degrade(recording: list[Point], sigma_m: float, outlier_rate: float, rng: random.Random) -> list[Point]:
    times = [point.time_s for point in recording]
    measurements = []
    t = times[0]
    while t <= times[-1]:
        true = interpolate(recording, times, t)
        error = [rng.gauss(0.0, sigma_m) for _ in range(3)]
        if rng.random() < outlier_rate:
            direction = [rng.gauss(0.0, 1.0) for _ in range(3)]
            size = rng.uniform(*OUTLIER_M) / math.hypot(*direction)
            error = [e + size * d for e, d in zip(error, direction)]
        measurements.append(Point(t, true.x_m + error[0], true.y_m + error[1], true.z_m + error[2]))
        t += rng.uniform(*GAP_S)
    return measurements

#Initiate the degrade with the string seed and writes the noisy data to csv
def write_scenario(target: Path, scenario: str, seed: int, out_dir: Path) -> Path:
    # A string seed goes through SHA-512, so it does not depend on PYTHONHASHSEED.
    rng = random.Random(f"{target.stem}/{scenario}/{seed}")
    measurements = degrade(load_recording(target), *SCENARIOS[scenario], rng)
    rows = [f"{p.time_s:.6f},{p.x_m:.6f},{p.y_m:.6f},{p.z_m:.6f}" for p in measurements]
    path = out_dir / f"{target.stem}_{scenario}_seed{seed:02d}.csv"
    path.write_text("\n".join(["timestamp_s,x_m,y_m,z_m", *rows]) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        for scenario in SCENARIOS:
            for seed in range(args.seeds):
                write_scenario(target, scenario, seed, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
