"""Behaviour guarantees of the estimator; public methods only."""

import itertools
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from python.baseline import Estimator as Baseline
from python.estimator import Estimator
from python.runner import load_events

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "data" / "fast.csv"
FAST_EVENTS = load_events(FAST)
START, VELOCITY = (10.0, 20.0, -5.0), (3.0, -2.0, 1.0)
NAN, INF = math.nan, math.inf


def on_line(t):
    return tuple(p + v * t for p, v in zip(START, VELOCITY))


GAPS = itertools.cycle((0.2, 2.0, 0.7, 1.3))  # irregular gaps inside 0.2-2 s
LINE_TIMES = list(itertools.accumulate(itertools.islice(GAPS, 299), initial=0.0))
LINE_EVENTS = [(t, *on_line(t)) for t in LINE_TIMES[:40]]
ON_TRACK = LINE_EVENTS[:21]
T_SPIKE, X_SPIKE, Y_SPIKE, Z_SPIKE = LINE_EVENTS[21]  # after a 0.2 s gap: tight prediction
SPIKES = [(T_SPIKE + 0.1 * i, X_SPIKE + 100.0, Y_SPIKE, Z_SPIKE) for i in range(3)]


def replay(estimator, events):
    return [(estimator.ingest_measurement(*e), estimator.predict(e[0] + 0.02)) for e in events]


"Output stays finite, even for extreme inputs and queries"
@pytest.mark.parametrize("events", [
    [(0.0, 1.0, 2.0, 3.0)],
    [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (1e5, 2.0, 2.0, 2.0)],
    [(0.0, 0.0, 0.0, 0.0), (0.2, 0.0, 0.0, 0.0), (0.4, 1e6, 0.0, 0.0)],
    [(0.0, 0.0, 0.0, 0.0), (1.0, 1e308, -1e308, 0.0), (2.0, -1e308, 1e308, 0.0)],
    [(0.0, 0.0, 0.0, 0.0), (1.0, 5.0, 5.0, 5.0)],
], ids=["one_measurement", "long_gap", "large_jump", "overflow", "extreme_query"])
def test_output_is_always_finite(events):
    states = [(estimator := Estimator()).predict(0.0)]  # before any measurement
    for event in events:
        estimator.ingest_measurement(*event)
        states += [estimator.predict(event[0] + dt) for dt in (0.0, 0.02, 10.0, 1e308)]
    assert all(math.isfinite(v) for s in states for v in (*s.position_m, *s.velocity_mps))

"NaN, inf, equal or older timestamps are refused and change nothing"
@pytest.mark.parametrize("bad", [
    (NAN, 1.0, 1.0, 1.0), (2.0, NAN, 1.0, 1.0), (2.0, 1.0, INF, 1.0), (2.0, 1.0, 1.0, -INF),
    (INF, 1.0, 1.0, 1.0), (1.0, 9.0, 9.0, 9.0), (0.5, 9.0, 9.0, 9.0),
], ids=["nan_t", "nan_x", "inf_y", "neginf_z", "inf_t", "equal_t", "older_t"])
def test_bad_measurement_is_refused_and_changes_nothing(bad):
    replay(estimator := Estimator(), [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)])
    before = estimator.predict(1.5)
    assert estimator.ingest_measurement(*bad) is False
    assert estimator.predict(1.5) == before

"No state survives reset()"
def test_reset_restores_a_fresh_estimator():
    replay(estimator := Estimator(), [*ON_TRACK, *SPIKES[:2]])  # ends at t > 0: refused if the clock survives
    estimator.reset()
    stream = [*ON_TRACK, SPIKES[0]]  # a third rejection in a row if the count survives
    assert replay(estimator, stream) == replay(Estimator(), stream)

"A straight line with irregular gaps is tracked exactly"
def test_straight_line_is_tracked_exactly():
    results = replay(estimator := Estimator(), LINE_EVENTS)
    assert all(accepted is True for accepted, _ in results)
    state = estimator.predict(LINE_EVENTS[-1][0] + 1.0)
    assert state.position_m == pytest.approx(on_line(state.timestamp_s), abs=1e-3)
    assert state.velocity_mps == pytest.approx(VELOCITY, abs=1e-3)

"A 100 m jump is rejected and does not disturb the track"
def test_single_outlier_is_rejected_and_error_stays_bounded():
    replay(estimator := Estimator(), ON_TRACK)
    assert estimator.ingest_measurement(*SPIKES[0]) is False
    for _, state in replay(estimator, LINE_EVENTS[22:]):
        assert math.dist(state.position_m, on_line(state.timestamp_s)) < 5.0

"D7 - Pins the gate: with the frozen config, NIS is 15.5 at 50 m and 18.8 at 55 m"
@pytest.mark.parametrize("jump_m, accepted", [(50.0, True), (55.0, False)])
def test_gate_threshold_sits_between_these_jumps(jump_m, accepted):
    replay(estimator := Estimator(), ON_TRACK)
    assert estimator.ingest_measurement(T_SPIKE, X_SPIKE + jump_m, Y_SPIKE, Z_SPIKE) is accepted

"D7 - Three rejections in a row mean the target really moved: follow it, don't lock out"
def test_third_rejection_in_a_row_is_accepted():
    replay(estimator := Estimator(), ON_TRACK)
    assert [estimator.ingest_measurement(*spike) for spike in SPIKES] == [False, False, True]
    assert estimator.predict(SPIKES[2][0]).position_m == pytest.approx(SPIKES[2][1:])
    assert estimator.ingest_measurement(T_SPIKE + 0.3, X_SPIKE + 500.0, Y_SPIKE, Z_SPIKE) is False  # count restarted

"D7 - Only rejections in a row count: an accepted measurement clears them"
def test_separated_rejections_never_force_a_restart():
    replay(estimator := Estimator(), ON_TRACK)
    good = [(spike[0] + 0.05, *on_line(spike[0] + 0.05)) for spike in SPIKES]
    assert [estimator.ingest_measurement(*e) for pair in zip(SPIKES, good) for e in pair] == [False, True] * 3

"D10 - After a gap over 5 s the old state is useless: restart, even on a distant fix"
def test_long_gap_restarts_on_the_new_position():
    replay(estimator := Estimator(), ON_TRACK)
    resumed = ON_TRACK[-1][0] + 6.0
    assert estimator.ingest_measurement(resumed, 1000.0, 0.0, 0.0) is True
    assert estimator.predict(resumed).position_m == (1000.0, 0.0, 0.0)

"D9 - The wide initial velocity uncertainty lets two measurements give the velocity"
def test_second_measurement_already_gives_velocity():
    replay(estimator := Estimator(), LINE_EVENTS[1:3])  # 2 s apart
    assert estimator.predict(LINE_EVENTS[2][0]).velocity_mps == pytest.approx(VELOCITY, abs=0.7)  # shared with acceleration at first

"A query far behind the state must not overflow the exponential"
def test_query_before_the_last_measurement_holds_the_state():
    replay(estimator := Estimator(), ON_TRACK)
    state = estimator.predict(-1e6)
    assert state.position_m == pytest.approx(estimator.predict(ON_TRACK[-1][0]).position_m)

"A rejected measurement still moves the clock on, or time could run backwards"
def test_older_than_a_rejected_measurement_is_refused():
    replay(estimator := Estimator(), ON_TRACK)
    assert estimator.ingest_measurement(*SPIKES[0]) is False
    older = T_SPIKE - 0.1
    assert estimator.ingest_measurement(older, *on_line(older)) is False

"D1 - The reason for the filter: noise must not pass straight into velocity"
def test_filter_halves_the_velocity_error_on_a_noisy_line():
    rng = random.Random("noisy line")
    events = [(t, *(p + rng.gauss(0.0, 1.0) for p in on_line(t))) for t in LINE_TIMES]

    def velocity_rmse(estimator):
        states = [state for _, state in replay(estimator, events)[20:]]  # after the start-up transient
        return math.sqrt(sum(math.dist(s.velocity_mps, VELOCITY) ** 2 for s in states) / len(states))

    assert velocity_rmse(Estimator()) < 0.5 * velocity_rmse(Baseline())
    assert sum(not accepted for accepted, _ in replay(Estimator(), events)) <= 0.01 * len(events)  # D7 limit

"Determinism: run-to-run output is byte-identical, even across processes"
def test_runner_output_is_byte_identical_across_processes():
    command = [sys.executable, str(ROOT / "python" / "runner.py"), "--input", str(FAST)]
    outputs = {subprocess.run(command, capture_output=True, check=True).stdout for _ in range(2)}
    assert len(outputs) == 1

"Each call is far below a 20 ms (50 Hz) cycle"
def test_call_time_is_far_below_the_20ms_cycle():
    estimator, ticks = Estimator(), [time.perf_counter()]
    for event in FAST_EVENTS:
        replay(estimator, [event])
        ticks.append(time.perf_counter())
    durations = sorted(b - a for a, b in zip(ticks, ticks[1:]))
    assert durations[int(0.99 * len(durations))] < 5e-3  # p99 ignores OS jitter
