"""Singer filter, one per axis, with an outlier gate on all three.

Settings come from an earlier tuning and are to be re-checked on the synthetic tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ACCEL_SIGMA_MPS2 = (35.0, 35.0, 35.0 * math.sqrt(0.2))  # z: q at 0.2 × x and y, vertical motion assumed gentler
ACCEL_TAU_S = 6.0  # how long acceleration persists
MEASUREMENT_SIGMA_M = 8.0
INITIAL_VELOCITY_SIGMA_MPS = 60.0
GATE_NIS = 16.27  # chi-square, 3 degrees of freedom, 99.9 %
MAX_REJECTIONS = 3  # in a row: the target really moved, so restart on it
MAX_GAP_S = 5.0  # beyond this the old state says nothing useful

AxisState = tuple[float, float, float]  # (position, velocity, acceleration)
Covariance = tuple[float, float, float, float, float, float]  # upper triangle: pp, pv, pa, vv, va, aa


@dataclass(frozen=True)
class TargetState:
    timestamp_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]


def transition(dt_s: float) -> tuple[float, float, float]:
    """The non-trivial entries of F(dt): acceleration into position, into velocity, and into itself."""
    rate = 1.0 / ACCEL_TAU_S
    decay = math.exp(-rate * dt_s)
    return (rate * dt_s - 1.0 + decay) / rate**2, (1.0 - decay) / rate, decay


def propagate_state(state: AxisState, dt_s: float) -> AxisState:
    position_m, velocity_mps, accel_mps2 = state
    f_pa, f_va, f_aa = transition(dt_s)
    return (
        position_m + dt_s * velocity_mps + f_pa * accel_mps2,
        velocity_mps + f_va * accel_mps2,
        f_aa * accel_mps2,
    )


def process_noise(dt_s: float, accel_sigma: float) -> Covariance:
    """Q(dt), the exact integral of the acceleration's driving noise over dt."""
    rate = 1.0 / ACCEL_TAU_S
    decay = math.exp(-rate * dt_s)
    psd = 2.0 * accel_sigma**2 * rate
    i_e = (1.0 - decay) / rate  # integral of e^(-rate s)
    i_ee = (1.0 - decay**2) / (2.0 * rate)  # integral of e^(-2 rate s)
    i_se = (1.0 - decay * (1.0 + rate * dt_s)) / rate**2  # integral of s e^(-rate s)
    return (
        psd * (rate**2 * dt_s**3 / 3.0 + dt_s + i_ee - rate * dt_s**2 + 2.0 * rate * i_se - 2.0 * i_e) / rate**4,
        psd * (rate * dt_s**2 / 2.0 - rate * i_se - dt_s + 2.0 * i_e - i_ee) / rate**3,
        psd * (rate * i_se - i_e + i_ee) / rate**2,
        psd * (dt_s - 2.0 * i_e + i_ee) / rate**2,
        psd * (i_e - i_ee) / rate,
        psd * i_ee,
    )


def propagate_covariance(covariance: Covariance, dt_s: float, accel_sigma: float) -> Covariance:
    """P = F·P·Fᵀ + Q(dt)."""
    pp, pv, pa, vv, va, aa = covariance
    f_pa, f_va, f_aa = transition(dt_s)
    row_p = (pp + dt_s * pv + f_pa * pa, pv + dt_s * vv + f_pa * va, pa + dt_s * va + f_pa * aa)  # first row of F·P
    row_v = (vv + f_va * va, va + f_va * aa)  # second row of F·P, velocity and acceleration columns
    q = process_noise(dt_s, accel_sigma)
    return (
        row_p[0] + dt_s * row_p[1] + f_pa * row_p[2] + q[0],
        row_p[1] + f_va * row_p[2] + q[1],
        f_aa * row_p[2] + q[2],
        row_v[0] + f_va * row_v[1] + q[3],
        f_aa * row_v[1] + q[4],
        f_aa * f_aa * aa + q[5],
    )


def update_axis(state: AxisState, covariance: Covariance, innovation_m: float) -> tuple[AxisState, Covariance]:
    """One scalar measurement update, with P in Joseph form."""
    pp, pv, pa, vv, va, aa = covariance
    r = MEASUREMENT_SIGMA_M**2
    s = pp + r
    gain_p, gain_v, gain_a = pp / s, pv / s, pa / s
    keep_p = 1.0 - gain_p
    return (
        (state[0] + gain_p * innovation_m, state[1] + gain_v * innovation_m, state[2] + gain_a * innovation_m),
        (
            keep_p * keep_p * pp + gain_p * gain_p * r,
            keep_p * (pv - gain_v * pp) + gain_p * gain_v * r,
            keep_p * (pa - gain_a * pp) + gain_p * gain_a * r,
            vv - 2.0 * gain_v * pv + gain_v * gain_v * s,
            va - gain_v * pa - gain_a * pv + gain_v * gain_a * s,
            aa - 2.0 * gain_a * pa + gain_a * gain_a * s,
        ),
    )


class Estimator:
    """Updates only on measurements; predict() extrapolates and stores nothing."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._latest_seen_s: float | None = None
        self._state_time_s: float | None = None
        self._state: tuple[AxisState, ...] = ((0.0, 0.0, 0.0),) * 3
        self._covariance: tuple[Covariance, ...] = ()
        self._rejections = 0

    def _restart(self, timestamp_s: float, position_m: tuple[float, float, float]) -> None:
        self._state_time_s = timestamp_s
        self._state = tuple((p, 0.0, 0.0) for p in position_m)
        self._covariance = tuple(
            (MEASUREMENT_SIGMA_M**2, 0.0, 0.0, INITIAL_VELOCITY_SIGMA_MPS**2, 0.0, sigma**2)
            for sigma in ACCEL_SIGMA_MPS2
        )
        self._rejections = 0

    def ingest_measurement(
        self,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> bool:
        values = (timestamp_s, x_m, y_m, z_m)
        if not all(math.isfinite(value) for value in values):
            return False
        if self._latest_seen_s is not None and timestamp_s <= self._latest_seen_s:
            return False
        self._latest_seen_s = timestamp_s
        measured_m = (x_m, y_m, z_m)
        if self._state_time_s is None or timestamp_s - self._state_time_s > MAX_GAP_S:
            self._restart(timestamp_s, measured_m)
            return True

        dt_s = timestamp_s - self._state_time_s
        state = [propagate_state(s, dt_s) for s in self._state]
        covariance = [propagate_covariance(c, dt_s, a) for c, a in zip(self._covariance, ACCEL_SIGMA_MPS2)]
        innovation_m = [z - s[0] for z, s in zip(measured_m, state)]
        r = MEASUREMENT_SIGMA_M**2
        nis = sum(nu * nu / (c[0] + r) for nu, c in zip(innovation_m, covariance))
        if nis > GATE_NIS:
            self._rejections += 1
            if self._rejections < MAX_REJECTIONS:
                return False
            self._restart(timestamp_s, measured_m)
            return True

        updated = [update_axis(s, c, nu) for s, c, nu in zip(state, covariance, innovation_m)]
        self._state_time_s = timestamp_s
        self._state = tuple(axis[0] for axis in updated)
        self._covariance = tuple(axis[1] for axis in updated)
        self._rejections = 0
        return True

    def predict(self, query_time_s: float) -> TargetState:
        if not math.isfinite(query_time_s):
            raise ValueError("query_time_s must be finite")
        if self._state_time_s is None:
            return TargetState(query_time_s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        elapsed_s = max(query_time_s - self._state_time_s, 0.0)  # e^x has no safe backward extrapolation
        state = [propagate_state(s, elapsed_s) for s in self._state]
        position_m = tuple(s[0] for s in state)
        if not all(math.isfinite(value) for value in position_m):
            return TargetState(query_time_s, tuple(s[0] for s in self._state), (0.0, 0.0, 0.0))
        return TargetState(query_time_s, position_m, tuple(s[1] for s in state))
