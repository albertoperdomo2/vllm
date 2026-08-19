# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig


class LeadTimeEstimator:
    """Predicts admission-to-first-demand lead time from queue depth.

    H = queue_position * EWMA of intervals between first-schedule events.
    Batched dequeues produce zero intervals, which pulls the EWMA down and
    makes the estimate conservative (fewer submissions), never optimistic.
    Constants are UNCALIBRATED until V2.0 characterization.
    """

    def __init__(self, config: PrefetchConfig):
        self._ewma_interval_ms = config.initial_admission_interval_ms
        self._alpha = config.admission_interval_ewma_alpha
        self._last_scheduled_at: float | None = None

    def on_first_scheduled(self, now: float) -> None:
        if self._last_scheduled_at is not None:
            interval_ms = (now - self._last_scheduled_at) * 1000.0
            self._ewma_interval_ms += self._alpha * (
                interval_ms - self._ewma_interval_ms
            )
        self._last_scheduled_at = now

    def predict_ms(self, queue_position: int) -> float:
        return max(0.0, queue_position * self._ewma_interval_ms)


def transfer_time_ms(config: PrefetchConfig, bundle_len: int) -> float:
    """Calibrated estimate of L_prefetch(B) for a bundle of bundle_len keys."""
    return config.transfer_base_ms + config.transfer_per_chunk_ms * bundle_len


def utility_ms(config: PrefetchConfig, bundle_len: int) -> float:
    """Expected utility U(B) in milliseconds.

    saved_critical_path_ms collapses to the full demand-fetch time because
    the deadline gate already guarantees H_remaining > L_prefetch(B), and
    E[C_eviction] is zero because speculative allocation never evicts.
    """
    saved = config.demand_load_per_chunk_ms * bundle_len
    return config.p_use * saved - config.delta_q_active_ms - config.c_failure_ms
