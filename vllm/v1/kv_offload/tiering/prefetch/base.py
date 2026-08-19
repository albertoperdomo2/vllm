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
    CAPACITY_SKIP = "vllm:kv_offload_tiering_prefetch_admission_capacity_skip"
    SUBMITTED = "vllm:kv_offload_tiering_prefetch_admission_submitted"
    SHADOW_SUBMIT = "vllm:kv_offload_tiering_prefetch_admission_shadow_submit"
    LOOKUP_UNRESOLVED = "vllm:kv_offload_tiering_prefetch_admission_lookup_unresolved"
    CANCELLED = "vllm:kv_offload_tiering_prefetch_admission_cancelled"
    BUNDLE_OUTCOMES = "vllm:kv_offload_tiering_prefetch_admission_bundle_outcomes"
    TRANSITIONS = "vllm:kv_offload_tiering_prefetch_admission_transitions"
    LEAD_TIME = "vllm:kv_offload_tiering_prefetch_admission_lead_time_seconds"

    TERMINAL_COUNTERS = (
        PRIMARY_REDUNDANT,
        SECONDARY_ABSENT,
        GATE_REJECT,
        CAPACITY_SKIP,
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
        AdmissionPrefetchMetrics.GATE_REJECT: (
            "keys rejected by the deadline or utility gate"
        ),
        AdmissionPrefetchMetrics.CAPACITY_SKIP: (
            "keys skipped for capacity or budget reasons"
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
    definitions[AdmissionPrefetchMetrics.LEAD_TIME] = OffloadingHistogramMetadata(
        documentation="Admission prefetch: predicted lead time at admission",
        buckets=_LEAD_TIME_BUCKETS,
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

    def prefetch_tier_allowed(self, tier_idx: int, req_context: ReqContext) -> bool: ...

    def prefetch_tier_label(self, tier_idx: int) -> tuple[str]: ...

    def prefetch_submit(
        self, tier_idx: int, keys: Sequence[OffloadKey], req_context: ReqContext
    ) -> AdmissionSubmitResult: ...


class PrefetchPolicy(ABC):
    """Pluggable proactive prefetch policy for TieringOffloadingManager.

    The manager calls on_request_admitted() when the connector registers a
    new request, step() once per scheduler step from on_schedule_end()
    (after finished-job processing, before the promotion flush),
    on_request_finished() when a request completes,
    on_promotion_finished() when a submitted promotion job resolves, and
    reset() from reset_cache() after all in-flight jobs have drained.
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

    @abstractmethod
    def step(self, context: ScheduleEndContext) -> None: ...

    @abstractmethod
    def on_request_finished(self, req_id: str) -> None: ...

    @abstractmethod
    def on_promotion_finished(
        self, keys: Sequence[OffloadKey], success: bool
    ) -> None: ...

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
