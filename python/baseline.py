"""Two-point constant-velocity baseline: velocity from the last two fixes, no filtering."""

from __future__ import annotations

import math

from estimator import TargetState

Fix = tuple[float, float, float, float]  # (timestamp, x, y, z)


class Estimator:
    """Same interface as the Singer estimator, so the runner and scorer can use either."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous: Fix | None = None
        self._latest: Fix | None = None

    def ingest_measurement(self, timestamp_s: float, x_m: float, y_m: float, z_m: float) -> bool:
        fix = (timestamp_s, x_m, y_m, z_m)
        if not all(math.isfinite(value) for value in fix):
            return False
        if self._latest is not None and timestamp_s <= self._latest[0]:
            return False
        self._previous, self._latest = self._latest, fix
        return True

    def predict(self, query_time_s: float) -> TargetState:
        if not math.isfinite(query_time_s):
            raise ValueError("query_time_s must be finite")
        if self._latest is None:
            return TargetState(query_time_s, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        t_last, *position = self._latest
        if self._previous is None:
            return TargetState(query_time_s, tuple(position), (0.0, 0.0, 0.0))
        t_prev, *previous = self._previous
        velocity = tuple((p - q) / (t_last - t_prev) for p, q in zip(position, previous))
        elapsed_s = query_time_s - t_last
        return TargetState(query_time_s, tuple(p + v * elapsed_s for p, v in zip(position, velocity)), velocity)
