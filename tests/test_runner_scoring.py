"""The runner's clock and the scorer: if these are wrong, every number built on them is void."""

import math

import pytest

from python.estimator import TargetState
from python.runner import load_events, query_times, run
from tools.scoring import Point, interpolate, score
from tools.trajectories import ROOT

FAST = ROOT / "data" / "fast.csv"


def test_ticks_are_50_hz_and_include_both_ends():
    ticks = query_times(1.0, 3.0)
    assert len(ticks) == 101
    assert ticks[0] == 1.0 and ticks[-1] == pytest.approx(3.0)
    assert all(b - a == pytest.approx(0.02) for a, b in zip(ticks, ticks[1:]))


class Recorder:
    """Logs which measurements were ingested before each predict call."""

    def __init__(self):
        self.seen = []
        self.log = []

    def ingest_measurement(self, t, x, y, z):
        self.seen.append(t)
        return True

    def predict(self, t):
        self.log.append((t, list(self.seen)))
        return TargetState(t, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_runner_never_uses_future_measurements():
    recorder = Recorder()
    run(recorder, [(0.0, 0, 0, 0), (0.03, 0, 0, 0), (0.04, 0, 0, 0), (0.1, 0, 0, 0)])
    for tick, seen in recorder.log:
        assert all(t <= tick + 1e-12 for t in seen)
    assert recorder.log[2] == (0.04, [0.0, 0.03, 0.04])  # a measurement exactly on a tick is used at that tick


@pytest.mark.parametrize("text, problem", [
    ("timestamp_s,x_m,y_m\n0,1,2\n", "missing"),
    ("timestamp_s,x_m,y_m,z_m\n0,1,2,nan\n", "non-finite"),
    ("timestamp_s,x_m,y_m,z_m\n1,1,2,3\n1,1,2,3\n", "increase"),
    ("timestamp_s,x_m,y_m,z_m\n", "no measurements"),
])
def test_bad_input_files_are_refused(tmp_path, text, problem):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        load_events(path)


def test_interpolation_is_linear_and_held_at_the_ends():
    track = [Point(0.0, 0.0, 0.0, 0.0), Point(1.0, 10.0, -2.0, 4.0)]
    times = [0.0, 1.0]
    assert interpolate(track, times, 0.25) == Point(0.25, 2.5, -0.5, 1.0)
    assert interpolate(track, times, -1.0) == Point(-1.0, 0.0, 0.0, 0.0)
    assert interpolate(track, times, 5.0) == Point(5.0, 10.0, -2.0, 4.0)


def test_a_shifted_input_scores_its_shift(tmp_path):
    """Feeding the baseline every track point moved by 10 m in x must cost about 10 m."""
    shifted = tmp_path / "shifted.csv"
    rows = ["timestamp_s,x_m,y_m,z_m"] + [f"{t},{x + 10.0},{y},{z}" for t, x, y, z in load_events(FAST)]
    shifted.write_text("\n".join(rows) + "\n", encoding="utf-8")
    clean, moved = score("baseline", FAST, FAST), score("baseline", FAST, shifted)
    assert moved.mae_m == pytest.approx(10.0, abs=clean.mae_m + 0.1)
    assert math.isfinite(moved.vel_rmse_mps)
