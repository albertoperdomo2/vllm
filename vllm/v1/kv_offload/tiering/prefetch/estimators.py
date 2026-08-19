# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig


class LeadTimeEstimator:
    """Predicts admission-to-first-demand lead time from queue depth.

    H = queue_position * EWMA of service time per first-scheduled request.
    A scheduler step contributes one throughput sample, so requests dequeued
    in the same batch do not contribute artificial zero-length intervals.
    Idle periods break the sample chain. Constants are UNCALIBRATED until
    V2.0 characterization.
    """

    def __init__(self, config: PrefetchConfig):
        self._initial_interval_ms = config.initial_admission_interval_ms
        self._ewma_interval_ms = self._initial_interval_ms
        self._alpha = config.admission_interval_ewma_alpha
        self._last_scheduled_at: float | None = None

    def on_first_scheduled(
        self, now: float, batch_size: int, queue_remains_nonempty: bool
    ) -> None:
        """Observe one scheduler batch of first-scheduled requests."""
        if batch_size <= 0:
            return
        if self._last_scheduled_at is not None and now > self._last_scheduled_at:
            interval_ms = (now - self._last_scheduled_at) * 1000.0 / batch_size
            self._ewma_interval_ms += self._alpha * (
                interval_ms - self._ewma_interval_ms
            )
        self._last_scheduled_at = now if queue_remains_nonempty else None

    def on_queue_idle(self) -> None:
        """Prevent idle time from entering the next throughput sample."""
        self._last_scheduled_at = None

    def reset(self) -> None:
        self._ewma_interval_ms = self._initial_interval_ms
        self._last_scheduled_at = None

    def predict_ms(self, queue_position: int) -> float:
        return max(0.0, queue_position * self._ewma_interval_ms)


class TransferCostModel:
    """Measures promotion cost as `base + per_chunk * n` from real transfers.

    Fitted by decayed weighted least squares over (batch size, elapsed)
    samples taken from completed secondary-to-primary jobs. Demand-driven
    promotions feed the same model, so the estimate is calibrated even while
    prefetch runs in shadow mode and has moved nothing itself.

    Measuring rather than configuring also makes contention self-regulating:
    when the tier is busy, observed transfers lengthen, the estimate rises,
    and the deadline gate tightens on its own.

    The configured constants are seeds, used until enough samples with
    differing batch sizes exist to identify a slope.
    """

    # Enough spread in batch size to separate intercept from slope. Decayed
    # weight saturates at 1/(1 - decay), which must exceed _MIN_SAMPLES or the
    # model would never leave its seeds.
    _MIN_SAMPLES = 20
    _MIN_VARIANCE = 0.5

    def __init__(self, config: PrefetchConfig, decay: float = 0.99):
        self._seed_base = config.transfer_base_ms
        self._seed_per_chunk = config.transfer_per_chunk_ms
        self._decay = decay
        self._w = 0.0
        self._wn = 0.0
        self._wnn = 0.0
        self._wy = 0.0
        self._wny = 0.0

    def observe(self, batch_size: int, elapsed_ms: float) -> None:
        if batch_size <= 0 or elapsed_ms < 0:
            return
        d = self._decay
        self._w = self._w * d + 1.0
        self._wn = self._wn * d + batch_size
        self._wnn = self._wnn * d + batch_size * batch_size
        self._wy = self._wy * d + elapsed_ms
        self._wny = self._wny * d + batch_size * elapsed_ms

    def _fit(self) -> tuple[float, float] | None:
        if self._w < self._MIN_SAMPLES:
            return None
        # Weighted variance of batch size; without spread the slope and the
        # intercept are not separately identifiable.
        mean_n = self._wn / self._w
        var_n = self._wnn / self._w - mean_n * mean_n
        if var_n < self._MIN_VARIANCE:
            return None
        cov = self._wny / self._w - mean_n * (self._wy / self._w)
        per_chunk = cov / var_n
        base = self._wy / self._w - per_chunk * mean_n
        if per_chunk <= 0:
            return None
        return max(0.0, base), per_chunk

    def predict_ms(self, batch_size: int) -> float:
        fit = self._fit()
        if fit is None:
            return self._seed_base + self._seed_per_chunk * batch_size
        base, per_chunk = fit
        return base + per_chunk * batch_size

    def measured(self) -> tuple[float, float] | None:
        """Fitted (base_ms, per_chunk_ms), or None while still seeded."""
        return self._fit()
