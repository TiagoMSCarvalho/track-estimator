#!/usr/bin/env python3
"""Replay a measurement file through an estimator and write its output at a fixed rate.

    python3 python/runner.py --input data/fast.csv [--output out.csv] [--estimator singer|baseline]

Input: CSV with columns timestamp_s, x_m, y_m, z_m (extra columns are ignored),
finite values and strictly increasing timestamps.
Output: one row per tick from the first to the last input timestamp, at RATE_HZ.
Each tick first ingests every measurement with timestamp <= tick (+ EVENT_TOLERANCE_S),
then calls predict(tick) exactly once, so the output never uses future data.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO

import baseline
import estimator

RATE_HZ = 50.0
EVENT_TOLERANCE_S = 1e-12
INPUT_COLUMNS = ("timestamp_s", "x_m", "y_m", "z_m")
OUTPUT_HEADER = "timestamp_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps"
ESTIMATORS: dict[str, Callable[[], object]] = {"singer": estimator.Estimator, "baseline": baseline.Estimator}

Event = tuple[float, float, float, float]  # (timestamp, x, y, z)


def load_events(path: Path) -> list[Event]:
    """Read and validate a measurement file; raises ValueError on bad input."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        events = []
        for line, row in enumerate(reader, start=2):
            values = tuple(float(row[c]) for c in INPUT_COLUMNS)
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{path}:{line}: non-finite value")
            if events and values[0] <= events[-1][0]:
                raise ValueError(f"{path}:{line}: timestamps must strictly increase")
            events.append(values)
    if not events:
        raise ValueError(f"{path}: no measurements")
    return events


def query_times(start_s: float, end_s: float, rate_hz: float = RATE_HZ) -> list[float]:
    """Ticks start_s, start_s + 1/rate, ... up to end_s (inclusive, with a small tolerance)."""
    count = int(math.floor((end_s - start_s) * rate_hz + 1e-9)) + 1
    return [start_s + i / rate_hz for i in range(count)]


def run(model: object, events: Sequence[Event]) -> list[estimator.TargetState]:
    """Replay the events through the estimator and return one state per tick."""
    states = []
    next_event = 0
    for tick in query_times(events[0][0], events[-1][0]):
        while next_event < len(events) and events[next_event][0] <= tick + EVENT_TOLERANCE_S:
            model.ingest_measurement(*events[next_event])
            next_event += 1
        states.append(model.predict(tick))
    return states


def write_states(states: Iterable[estimator.TargetState], out: TextIO) -> None:
    """17 significant digits: enough to round-trip a double exactly."""
    out.write(OUTPUT_HEADER + "\n")
    for s in states:
        values = (s.timestamp_s, *s.position_m, *s.velocity_mps)
        out.write(",".join(format(v, ".17g") for v in values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="default: standard output")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS), default="singer")
    args = parser.parse_args()

    try:
        states = run(ESTIMATORS[args.estimator](), load_events(args.input))
    except ValueError as error:
        print(f"runner: {error}", file=sys.stderr)
        return 1
    if args.output is None:
        write_states(states, sys.stdout)
    else:
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            write_states(states, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
