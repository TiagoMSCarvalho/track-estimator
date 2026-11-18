#!/usr/bin/env python3
"""Generate the clean truth tracks used by every test and scenario.

    python3 tools/trajectories.py            # rewrites data/fast.csv and data/slow.csv

Each track flies at constant speed through straight legs, turns, weaves, climbs and
descents. Position is integrated from an analytic velocity, so the file carries the
exact velocity too. Fully deterministic: no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE_HZ = 5.0
STEP_S = 1e-3  # integration step; position error per 100 s is far below a millimetre
HEADER = "timestamp_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps"


@dataclass(frozen=True)
class Segment:
    duration_s: float
    turn_dps: float = 0.0  # constant heading rate, degrees per second
    weave_deg: float = 0.0  # heading oscillation amplitude
    weave_period_s: float = 1.0
    climb_mps: float | None = None  # vertical speed reached at the end of the ramp; None keeps the current one
    climb_ramp_s: float = 3.0  # time to reach climb_mps from the previous vertical speed


@dataclass(frozen=True)
class Track:
    name: str
    speed_mps: float  # horizontal speed
    start_m: tuple[float, float, float]
    heading_deg: float
    segments: tuple[Segment, ...]


TRACKS = (
    Track("fast", 48.0, (0.0, 0.0, 100.0), 0.0, (
        Segment(15.0),
        Segment(12.0, turn_dps=15.0),
        Segment(10.0, climb_mps=6.0),
        Segment(24.0, weave_deg=25.0, weave_period_s=8.0, climb_mps=0.0),
        Segment(10.0, climb_mps=-6.0),
        Segment(15.0, turn_dps=-20.0, climb_mps=0.0),
        Segment(14.0),
    )),
    Track("slow", 23.0, (500.0, -300.0, 60.0), 90.0, (
        Segment(20.0),
        Segment(20.0, turn_dps=6.0),
        Segment(15.0, climb_mps=2.0),
        Segment(24.0, weave_deg=15.0, weave_period_s=12.0, climb_mps=0.0),
        Segment(20.0, turn_dps=-8.0, climb_mps=0.0),
        Segment(21.0),
    )),
)


def velocity_function(track: Track):
    """Return v(t) for the whole track as a function, plus the total duration."""
    pieces = []  # (start time, segment, heading at start in rad, vertical speed at start)
    t0, heading, vz = 0.0, math.radians(track.heading_deg), 0.0
    for seg in track.segments:
        pieces.append((t0, seg, heading, vz))
        if seg.weave_deg == 0.0:  # a whole number of weave periods returns to the start heading
            heading += math.radians(seg.turn_dps) * seg.duration_s
        vz = vz if seg.climb_mps is None else seg.climb_mps
        t0 += seg.duration_s

    def velocity(t: float) -> tuple[float, float, float]:
        start, seg, psi0, vz0 = pieces[0]
        for piece in pieces:
            if piece[0] <= t:
                start, seg, psi0, vz0 = piece
        tau = t - start
        psi = psi0 + math.radians(seg.turn_dps) * tau
        if seg.weave_deg:
            psi += math.radians(seg.weave_deg) * math.sin(2.0 * math.pi * tau / seg.weave_period_s)
        target = vz0 if seg.climb_mps is None else seg.climb_mps
        vz_now = vz0 + (target - vz0) * min(tau / seg.climb_ramp_s, 1.0)
        return track.speed_mps * math.cos(psi), track.speed_mps * math.sin(psi), vz_now

    return velocity, t0


def sample(track: Track) -> list[tuple[float, ...]]:
    """Rows (t, x, y, z, vx, vy, vz) at SAMPLE_RATE_HZ, by trapezoidal integration of v(t)."""
    velocity, duration_s = velocity_function(track)
    steps_per_sample = round(1.0 / (SAMPLE_RATE_HZ * STEP_S))
    position = list(track.start_m)
    rows, step = [], 0
    while True:
        t = step * STEP_S
        if step % steps_per_sample == 0:
            rows.append((t, *position, *velocity(t)))
        if t >= duration_s - 1e-9:
            return rows
        v0, v1 = velocity(t), velocity(t + STEP_S)
        position = [p + 0.5 * STEP_S * (a + b) for p, a, b in zip(position, v0, v1)]
        step += 1


def write_track(track: Track, out_dir: Path) -> Path:
    path = out_dir / f"{track.name}.csv"
    lines = [HEADER] + [",".join(f"{v:.6f}" for v in row) for row in sample(track)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    for track in TRACKS:
        print(write_track(track, ROOT / "data"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
