"""Scenario files must be reproducible and match their label, or every score built on them is void."""

import math
import statistics

import pytest

from tools.degrade import SCENARIOS, TARGETS, write_scenario
from tools.scoring import interpolate, load_recording, score

ROOT = TARGETS[0].parents[1]
SCENARIO_NAMES = list(SCENARIOS)
TRUTH = load_recording(TARGETS[0])
TRUTH_TIMES = [p.time_s for p in TRUTH]


def errors(path):
    """(measurement point, per-axis measurement error) for each row of a scenario file."""
    rows = []
    for m in load_recording(path):
        true = interpolate(TRUTH, TRUTH_TIMES, m.time_s)
        rows.append((m, (m.x_m - true.x_m, m.y_m - true.y_m, m.z_m - true.z_m)))
    return rows


def test_same_seed_is_identical_and_different_seed_differs(tmp_path):
    first = write_scenario(TARGETS[0], "S3", 0, tmp_path).read_bytes()
    assert write_scenario(TARGETS[0], "S3", 0, tmp_path).read_bytes() == first
    assert write_scenario(TARGETS[0], "S3", 1, tmp_path).read_bytes() != first


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_gaps_stay_inside_0_2_to_2_s(tmp_path, scenario):
    times = [p.time_s for p in load_recording(write_scenario(TARGETS[0], scenario, 0, tmp_path))]
    assert times[0] == TRUTH_TIMES[0] and times[-1] <= TRUTH_TIMES[-1]
    assert all(0.2 - 2e-6 <= b - a <= 2.0 + 2e-6 for a, b in zip(times, times[1:]))


def test_zero_noise_lies_on_truth(tmp_path):
    rows = errors(write_scenario(TARGETS[0], "S1", 0, tmp_path))
    assert all(abs(e) < 1e-4 for _, error in rows for e in error)  # 6-decimal rounding of t and x


def test_noise_and_outliers_match_their_label(tmp_path):
    pooled = [row for seed in range(40) for row in errors(write_scenario(TARGETS[0], "S3", seed, tmp_path))]
    sizes = [math.hypot(*error) for _, error in pooled]
    outliers = [size for size in sizes if size > 12.0]  # sigma = 1 m: unreachable by noise alone
    assert 0.02 < len(outliers) / len(sizes) < 0.04
    assert 15.0 < min(outliers) and max(outliers) < 60.0
    inliers = [e for (_, error), size in zip(pooled, sizes) if size <= 12.0 for e in error]
    assert statistics.pstdev(inliers) == pytest.approx(1.0, abs=0.05)


def test_scenario_file_runs_through_the_scorer(tmp_path):
    path = write_scenario(TARGETS[0], "S3", 0, tmp_path)
    assert math.isfinite(score("singer", TARGETS[0], path).mae_m)
