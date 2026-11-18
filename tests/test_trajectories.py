"""The truth tracks: their positions and velocities must agree, or the velocity scores are wrong."""

import csv
import math

import pytest

from tools.trajectories import ROOT, TRACKS, sample, write_track


@pytest.mark.parametrize("track", TRACKS, ids=[t.name for t in TRACKS])
def test_committed_file_matches_the_generator(tmp_path, track):
    assert write_track(track, tmp_path).read_bytes() == (ROOT / "data" / f"{track.name}.csv").read_bytes()


@pytest.mark.parametrize("track", TRACKS, ids=[t.name for t in TRACKS])
def test_position_steps_match_the_average_velocity(track):
    rows = sample(track)
    for a, b in zip(rows, rows[1:]):
        dt = b[0] - a[0]
        for axis in range(1, 4):
            step = b[axis] - a[axis]
            mean_velocity = 0.5 * (a[axis + 3] + b[axis + 3])
            assert step == pytest.approx(mean_velocity * dt, abs=0.05)  # trapezoid over 0.2 s of manoeuvre


@pytest.mark.parametrize("track", TRACKS, ids=[t.name for t in TRACKS])
def test_horizontal_speed_is_constant(track):
    with (ROOT / "data" / f"{track.name}.csv").open(encoding="utf-8") as handle:
        speeds = [math.hypot(float(r["vx_mps"]), float(r["vy_mps"])) for r in csv.DictReader(handle)]
    assert speeds == pytest.approx([track.speed_mps] * len(speeds), abs=1e-5)


def test_samples_are_5_hz_from_zero():
    rows = sample(TRACKS[0])
    assert rows[0][0] == 0.0
    assert all(b[0] - a[0] == pytest.approx(0.2) for a, b in zip(rows, rows[1:]))
