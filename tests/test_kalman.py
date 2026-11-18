"""The filter's exact-discretisation maths, which public methods can only show indirectly."""

import numpy as np
import pytest

import python.estimator as kf

ACCEL_SIGMA = 12.0
UNCERTAIN = (4.0, 1.5, 0.5, 9.0, 2.0, 16.0)  # upper triangle: pp, pv, pa, vv, va, aa


def expm(matrix):
    scaled = matrix / 2.0**12
    result, term = np.eye(len(matrix)), np.eye(len(matrix))
    for k in range(1, 16):
        term = term @ scaled / k
        result = result + term
    for _ in range(12):
        result = result @ result
    return result


def van_loan(dt_s, accel_sigma):
    """F and Q from the continuous model alone, so they share nothing with the closed forms."""
    a = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / kf.ACCEL_TAU_S]])
    psd = np.diag([0.0, 0.0, 2.0 * accel_sigma**2 / kf.ACCEL_TAU_S])
    block = expm(np.block([[-a, psd], [np.zeros((3, 3)), a.T]]) * dt_s)
    f = block[3:, 3:].T
    return f, f @ block[:3, 3:]


@pytest.mark.parametrize("dt_s", [0.02, 0.2, 1.0, 2.0, 5.0])
def test_f_and_q_match_van_loan(dt_s):
    f, q = van_loan(dt_s, ACCEL_SIGMA)
    assert kf.transition(dt_s) == pytest.approx((f[0, 2], f[1, 2], f[2, 2]), rel=1e-9)
    assert kf.process_noise(dt_s, ACCEL_SIGMA) == pytest.approx(q[np.triu_indices(3)], rel=1e-5, abs=1e-9)  # abs: round-off


def test_process_noise_from_zero_uncertainty_is_q():
    assert kf.propagate_covariance((0.0,) * 6, 2.0, ACCEL_SIGMA) == kf.process_noise(2.0, ACCEL_SIGMA)


def test_covariance_propagation_matches_the_full_matrix_product():
    pp, pv, pa, vv, va, aa = UNCERTAIN
    p = np.array([[pp, pv, pa], [pv, vv, va], [pa, va, aa]])
    f, q = van_loan(1.3, ACCEL_SIGMA)
    expected = f @ p @ f.T + q
    assert kf.propagate_covariance(UNCERTAIN, 1.3, ACCEL_SIGMA) == pytest.approx(expected[np.triu_indices(3)], rel=1e-5)


@pytest.mark.parametrize("first, second", [(0.02, 0.2), (0.2, 2.0), (1.3, 0.7)])
def test_two_short_steps_equal_one_long_step(first, second):
    split = kf.propagate_covariance(kf.propagate_covariance(UNCERTAIN, first, ACCEL_SIGMA), second, ACCEL_SIGMA)
    assert split == pytest.approx(kf.propagate_covariance(UNCERTAIN, first + second, ACCEL_SIGMA))
    state = kf.propagate_state(kf.propagate_state((1.0, 2.0, 3.0), first), second)
    assert state == pytest.approx(kf.propagate_state((1.0, 2.0, 3.0), first + second))


def test_long_tau_holds_acceleration_constant(monkeypatch):
    monkeypatch.setattr(kf, "ACCEL_TAU_S", 1e4)
    assert kf.propagate_state((0.0, 0.0, 2.0), 3.0) == pytest.approx((9.0, 6.0, 2.0), rel=1e-3)


def test_short_tau_reduces_to_constant_velocity_with_equal_q(monkeypatch):
    monkeypatch.setattr(kf, "ACCEL_TAU_S", 1e-3)
    q = 1200.0
    pp, pv, _, vv, *_ = kf.process_noise(2.0, (q / (2.0 * kf.ACCEL_TAU_S)) ** 0.5)
    assert (pp, pv, vv) == pytest.approx((q * 8 / 3, q * 2, q * 2), rel=1e-2)
    assert kf.propagate_state((0.0, 3.0, 0.0), 2.0) == pytest.approx((6.0, 3.0, 0.0))


def test_restart_opens_all_three_uncertainties():
    estimator = kf.Estimator()
    estimator.ingest_measurement(0.0, 1.0, 2.0, 3.0)
    sigma = kf.ACCEL_SIGMA_MPS2[0]
    assert estimator._covariance[0] == (kf.MEASUREMENT_SIGMA_M**2, 0.0, 0.0, kf.INITIAL_VELOCITY_SIGMA_MPS**2, 0.0, sigma**2)


def test_update_matches_a_hand_worked_gain_of_one_half():
    r = kf.MEASUREMENT_SIGMA_M**2  # position variance = R gives K = (1/2, 1/4, 1/8)
    state, covariance = kf.update_axis((0.0, 0.0, 0.0), (r, r / 2, r / 4, r / 4, r / 8, r / 16), 8.0)
    assert state == pytest.approx((4.0, 2.0, 1.0))
    assert covariance == pytest.approx((r / 2, r / 4, r / 8, r / 8, r / 16, r / 32))
