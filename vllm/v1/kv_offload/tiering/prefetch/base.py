# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )


class AdmissionPrefetchMetrics:
    """Metric names for the admission prefetch policy.

    The per-key counters form a terminal partition: every considered key is
    counted in considered exactly once, together with exactly one terminal
    class, so considered always equals the sum of the terminal counters.
    """

    CONSIDERED = "vllm:kv_offload_tiering_prefetch_admission_considered"
    PRIMARY_REDUNDANT = "vllm:kv_offload_tiering_prefetch_admission_primary_redundant"
    SECONDARY_ABSENT = "vllm:kv_offload_tiering_prefetch_admission_secondary_absent"
    GATE_REJECT = "vllm:kv_offload_tiering_prefetch_admission_gate_reject"
    # Capacity refusals, split by cause. A single capacity_skip counter made
    # the first benchmark unattributable: hard-cap trimming and allocator
    # refusal are entirely different failures with entirely different fixes.
    BUNDLE_OVERFLOW = "vllm:kv_offload_tiering_prefetch_admission_bundle_overflow"
    BUNDLE_TRIM = "vllm:kv_offload_tiering_prefetch_admission_bundle_trim"
    ALLOC_REFUSED = "vllm:kv_offload_tiering_prefetch_admission_alloc_refused"
    SUBMITTED = "vllm:kv_offload_tiering_prefetch_admission_submitted"
    SHADOW_SUBMIT = "vllm:kv_offload_tiering_prefetch_admission_shadow_submit"
    LOOKUP_UNRESOLVED = "vllm:kv_offload_tiering_prefetch_admission_lookup_unresolved"
    CANCELLED = "vllm:kv_offload_tiering_prefetch_admission_cancelled"
    BUNDLE_OUTCOMES = "vllm:kv_offload_tiering_prefetch_admission_bundle_outcomes"
    TRANSITIONS = "vllm:kv_offload_tiering_prefetch_admission_transitions"
    ACTIVATION_DEFERRED = (
        "vllm:kv_offload_tiering_prefetch_admission_activation_deferred"
    )
    OWNER_RELEASES = "vllm:kv_offload_tiering_prefetch_admission_owner_releases"
    ACTIVE_OWNER = "vllm:kv_offload_tiering_prefetch_admission_active_owner"
    BUNDLE_SIZE = "vllm:kv_offload_tiering_prefetch_admission_bundle_chunks"
    DEADLINE_MARGIN = (
        "vllm:kv_offload_tiering_prefetch_admission_deadline_margin_seconds"
    )
    TRANSFER_COST_BASE = "vllm:kv_offload_tiering_prefetch_transfer_cost_base_seconds"
    TRANSFER_COST_PER_CHUNK = (
        "vllm:kv_offload_tiering_prefetch_transfer_cost_per_chunk_seconds"
    )
    LEAD_TIME = "vllm:kv_offload_tiering_prefetch_admission_lead_time_seconds"
    ACTUAL_LEAD_TIME = (
        "vllm:kv_offload_tiering_prefetch_admission_actual_lead_time_seconds"
    )

    TERMINAL_COUNTERS = (
        PRIMARY_REDUNDANT,
        SECONDARY_ABSENT,
        GATE_REJECT,
        BUNDLE_OVERFLOW,
        BUNDLE_TRIM,
        ALLOC_REFUSED,
        SUBMITTED,
        SHADOW_SUBMIT,
        LOOKUP_UNRESOLVED,
        CANCELLED,
    )


_LEAD_TIME_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def build_admission_prefetch_metric_definitions(
    prefetch_config: dict[str, Any],
) -> dict[str, OffloadingMetricMetadata]:
    """Metric definitions for the admission prefetch policy.

    Must be callable from the raw extra_config["prefetch"] dict alone,
    because TieringOffloadingSpec.build_metric_definitions is a classmethod
    invoked before any manager exists.
    """
    per_key = {
        AdmissionPrefetchMetrics.CONSIDERED: "keys considered by the policy",
        AdmissionPrefetchMetrics.PRIMARY_REDUNDANT: (
            "keys already resident or loading in the primary tier"
        ),
        AdmissionPrefetchMetrics.SECONDARY_ABSENT: (
            "keys resolved absent from the secondary tier"
        ),
        AdmissionPrefetchMetrics.GATE_REJECT: ("keys rejected by the deadline gate"),
        AdmissionPrefetchMetrics.BUNDLE_OVERFLOW: (
            "keys dropped because too many bundles were already live"
        ),
        AdmissionPrefetchMetrics.BUNDLE_TRIM: (
            "candidate keys past the per-bundle ceiling, declined at admission "
            "and never probed; a high share means max_bundle_chunks is the "
            "binding constraint"
        ),
        AdmissionPrefetchMetrics.ALLOC_REFUSED: (
            "keys the primary tier refused to allocate a speculative block for"
        ),
        AdmissionPrefetchMetrics.SUBMITTED: ("keys submitted for proactive promotion"),
        AdmissionPrefetchMetrics.SHADOW_SUBMIT: (
            "keys that would have been submitted in shadow mode"
        ),
        AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED: (
            "keys whose residency lookup never resolved before the bundle "
            "reached a terminal state"
        ),
        AdmissionPrefetchMetrics.CANCELLED: (
            "keys cancelled by request completion, preemption, or cache reset"
        ),
    }
    definitions: dict[str, OffloadingMetricMetadata] = {
        name: OffloadingCounterMetadata(
            documentation=f"Admission prefetch: {doc}",
            labelnames=("tier",),
        )
        for name, doc in per_key.items()
    }
    definitions[AdmissionPrefetchMetrics.BUNDLE_OUTCOMES] = OffloadingCounterMetadata(
        documentation="Admission prefetch: terminal bundle outcomes",
        labelnames=("tier", "outcome"),
    )
    definitions[AdmissionPrefetchMetrics.TRANSITIONS] = OffloadingCounterMetadata(
        documentation="Admission prefetch: bundle state transitions",
        labelnames=("transition",),
    )
    definitions[AdmissionPrefetchMetrics.ACTIVATION_DEFERRED] = (
        OffloadingCounterMetadata(
            documentation=(
                "Admission prefetch: JIT activation deferred by demand or "
                "existing speculative work"
            ),
            labelnames=("tier", "reason"),
        )
    )
    definitions[AdmissionPrefetchMetrics.OWNER_RELEASES] = OffloadingCounterMetadata(
        documentation="Admission prefetch: request-owned bundle releases",
        labelnames=("tier", "reason"),
    )
    definitions[AdmissionPrefetchMetrics.ACTIVE_OWNER] = OffloadingGaugeMetadata(
        documentation=(
            "Admission prefetch: whether one request currently owns speculative "
            "lookup, transfer, or retained residency"
        ),
        labelnames=("tier",),
    )
    definitions[AdmissionPrefetchMetrics.LEAD_TIME] = OffloadingHistogramMetadata(
        documentation="Admission prefetch: predicted lead time at admission",
        buckets=_LEAD_TIME_BUCKETS,
    )
    definitions[AdmissionPrefetchMetrics.ACTUAL_LEAD_TIME] = (
        OffloadingHistogramMetadata(
            documentation=(
                "Admission prefetch: actual admission-to-first-schedule lead time"
            ),
            buckets=_LEAD_TIME_BUCKETS,
        )
    )
    definitions[AdmissionPrefetchMetrics.BUNDLE_SIZE] = OffloadingHistogramMetadata(
        documentation=(
            "Admission prefetch: contiguous resident prefix length at gate time, "
            "in chunks"
        ),
        buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    )
    definitions[AdmissionPrefetchMetrics.DEADLINE_MARGIN] = OffloadingHistogramMetadata(
        documentation=(
            "Admission prefetch: remaining lead time minus predicted transfer "
            "time at gate time. Negative means the deadline gate rejected the "
            "bundle; the distribution shows whether lead time is the binding "
            "constraint or a comfortable margin."
        ),
        buckets=(
            -10.0,
            -1.0,
            -0.1,
            -0.01,
            0.0,
            0.01,
            0.1,
            1.0,
            10.0,
            60.0,
        ),
    )
    definitions[AdmissionPrefetchMetrics.TRANSFER_COST_BASE] = OffloadingGaugeMetadata(
        documentation=(
            "Admission prefetch: measured fixed cost per promotion job. "
            "Includes completion-detection latency of up to one scheduler "
            "step, so it is an upper bound on device fixed cost, not a "
            "measurement of it. Reported only once enough real transfers "
            "have been observed to fit it."
        ),
        labelnames=("tier",),
    )
    definitions[AdmissionPrefetchMetrics.TRANSFER_COST_PER_CHUNK] = (
        OffloadingGaugeMetadata(
            documentation=(
                "Admission prefetch: measured marginal cost per promoted chunk. "
                "This is what the deadline gate spends; a rising value means the "
                "tier is contended."
            ),
            labelnames=("tier",),
        )
    )
    return definitions


@dataclass
class AdmissionSubmitResult:
    """Per-key disposition of a live bundle submission."""

    submitted: list[OffloadKey] = field(default_factory=list)
    primary_redundant: list[OffloadKey] = field(default_factory=list)
    capacity_skipped: list[OffloadKey] = field(default_factory=list)


class PrefetchHost(Protocol):
    """The manager-side surface a prefetch policy drives.

    Implemented by TieringOffloadingManager.
    """

    def prefetch_primary_lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult: ...

    def prefetch_secondary_lookup(
        self, tier_idx: int, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult: ...

    def prefetch_demand_idle(self, tier_idx: int) -> bool:
        """Whether demand lookup and transfer work is idle on this tier."""
        ...

    def prefetch_speculation_idle(self, tier_idx: int) -> bool:
        """Whether no speculative transfer is pending or in flight."""
        ...

    def prefetch_tier_allowed(self, tier_idx: int, req_context: ReqContext) -> bool: ...

    def prefetch_tier_label(self, tier_idx: int) -> tuple[str]: ...

    def prefetch_transfer_cost_ms(self, tier_idx: int, n_chunks: int) -> float:
        """Measured promotion cost for a batch of n_chunks on this tier."""
        ...

    def prefetch_submit(
        self, tier_idx: int, keys: Sequence[OffloadKey], req_context: ReqContext
    ) -> AdmissionSubmitResult: ...

    def prefetch_release_owner(self, req_id: str, *, demanded: bool = False) -> None:
        """Release request-owned speculative residency and in-flight ownership."""
        ...


class PrefetchPolicy(ABC):
    """Pluggable proactive prefetch policy for TieringOffloadingManager.

    The manager calls on_request_enqueued() for every local request and
    on_request_admitted() only when the connector offers an opted-in
    candidate. It calls step() once per scheduler step from on_schedule_end(),
    after finished-job processing and before the promotion flush.
    """

    def __init__(
        self,
        config: PrefetchConfig,
        host: PrefetchHost,
        clock: Callable[[], float] = time.monotonic,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        self.config = config
        self.host = host
        self.clock = clock
        self._stats = OffloadingConnectorStats()

    @abstractmethod
    def on_request_admitted(
        self, req_context: ReqContext, offload_keys: Sequence[OffloadKey]
    ) -> None: ...

    def on_request_enqueued(self, req_context: ReqContext) -> None:
        """Observe every scheduler admission, including non-opted-in requests."""
        return

    @abstractmethod
    def step(self, context: ScheduleEndContext) -> None: ...

    @abstractmethod
    def on_request_finished(self, req_id: str) -> None: ...

    @abstractmethod
    def on_promotion_finished(
        self, keys: Sequence[OffloadKey], success: bool
    ) -> None: ...

    def on_prefetch_demand(self, key: OffloadKey, primary_result: LookupResult) -> None:
        """Observe demand reaching a key owned by an in-flight prefetch."""
        return

    @abstractmethod
    def has_pending_work(self) -> bool: ...

    @abstractmethod
    def reset(self) -> None: ...

    def get_stats(self) -> "OffloadingConnectorStats | None":
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        stats = self._stats
        if stats.is_empty():
            return None
        self._stats = OffloadingConnectorStats()
        return stats
