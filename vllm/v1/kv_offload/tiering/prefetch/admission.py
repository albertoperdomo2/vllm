# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import operator
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.prefetch.base import (
    AdmissionPrefetchMetrics,
    PrefetchHost,
    PrefetchPolicy,
)
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig
from vllm.v1.kv_offload.tiering.prefetch.estimators import LeadTimeEstimator

logger = init_logger(__name__)

_bundle_deadline = operator.attrgetter("deadline")


class BundleState(enum.Enum):
    PENDING_LOOKUP = enum.auto()
    RESIDENT = enum.auto()
    SUBMITTED = enum.auto()
    READY = enum.auto()
    LATE = enum.auto()
    ABSENT = enum.auto()
    FAILED = enum.auto()
    GATE_REJECTED = enum.auto()
    CAPACITY_SKIPPED = enum.auto()
    SHADOW_SUBMITTED = enum.auto()
    CANCELLED = enum.auto()


_TERMINAL_STATES = frozenset(
    {
        BundleState.READY,
        BundleState.LATE,
        BundleState.ABSENT,
        BundleState.FAILED,
        BundleState.GATE_REJECTED,
        BundleState.CAPACITY_SKIPPED,
        BundleState.SHADOW_SUBMITTED,
        BundleState.CANCELLED,
    }
)


@dataclass
class Bundle:
    """One candidate prefix bundle per request.

    keys is the candidate window [k_m ..] starting at the primary-residency
    frontier; resolved_run is the length of its contiguous secondary-RESIDENT
    prefix. Keys past the first ABSENT key are dropped uncounted because the
    demand scan can never reach them (prefix-chained hashes).
    """

    req_id: str
    req_context: ReqContext
    tier_idx: int
    keys: list[OffloadKey]
    admitted_at: float
    lead_time_ms: float
    resolved_run: int = 0
    absent_found: bool = False
    state: BundleState = BundleState.PENDING_LOOKUP
    outstanding: set[OffloadKey] = field(default_factory=set)
    any_load_failed: bool = False
    demanded_while_pending: bool = False
    # Absolute demand deadline, fixed at admission. Stored rather than derived
    # so the per-step ordering is a plain attribute read.
    deadline: float = field(init=False)

    def __post_init__(self) -> None:
        self.deadline = self.admitted_at + self.lead_time_ms / 1000.0


@dataclass
class QueuedRequest:
    """Queue observation captured before the per-request opt-in gate."""

    admitted_at: float
    queue_position: int
    is_prefetch_candidate: bool = False


class AdmissionPrefetchPolicy(PrefetchPolicy):
    """Residency- and deadline-gated admission prefetch (ABC V2.1).

    Selects ordered contiguous prefix bundles at request admission, verifies
    secondary residency through the async lookup path, and submits a bundle
    only while the predicted remaining lead time exceeds its calibrated
    transfer time and its expected utility is positive. In shadow mode
    (default) gate decisions are logged and counted but nothing is reserved
    or submitted.
    """

    def __init__(
        self,
        config: PrefetchConfig,
        host: PrefetchHost,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(config, host, clock)
        self._bundles: dict[str, Bundle] = {}
        # Insertion-ordered non-terminal bundles; kept small so per-step work
        # is bounded by live bundles, not history.
        self._active: dict[str, None] = {}
        self._admitted_unscheduled: OrderedDict[str, QueuedRequest] = OrderedDict()
        self._submitted_key_owner: dict[OffloadKey, str] = {}
        self._inflight_speculative_bytes = 0
        self._estimator = LeadTimeEstimator(config)
        self._tier_label = host.prefetch_tier_label(config.tier_idx)

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------

    def _finalize_keys(self, terminal_counter: str, count: int) -> None:
        """Count keys considered and terminal atomically.

        The single write point for the per-key partition: considered and
        exactly one terminal class always advance together, so the partition
        invariant cannot be violated structurally.
        """
        if count <= 0:
            return
        self._stats.increase_counter(
            AdmissionPrefetchMetrics.CONSIDERED,
            counter_increase_value=count,
            labelvalues=self._tier_label,
        )
        self._stats.increase_counter(
            terminal_counter,
            counter_increase_value=count,
            labelvalues=self._tier_label,
        )

    def _bundle_outcome(self, outcome: str) -> None:
        self._stats.increase_counter(
            AdmissionPrefetchMetrics.BUNDLE_OUTCOMES,
            labelvalues=(self._tier_label[0], outcome),
        )

    def _set_state(self, bundle: Bundle, new_state: BundleState) -> None:
        old_state = bundle.state
        bundle.state = new_state
        self._stats.increase_counter(
            AdmissionPrefetchMetrics.TRANSITIONS,
            labelvalues=(f"{old_state.name.lower()}->{new_state.name.lower()}",),
        )
        if new_state in _TERMINAL_STATES:
            self._active.pop(bundle.req_id, None)
            del self._bundles[bundle.req_id]

    def _pending_key_count(self, bundle: Bundle) -> int:
        """Window keys with an issued but unresolved residency lookup.

        Zero once an ABSENT key froze the run: the absent key was counted at
        discovery and everything past it is dropped uncounted.
        """
        if bundle.absent_found:
            return 0
        return len(bundle.keys) - bundle.resolved_run

    # ------------------------------------------------------------------
    # Terminalization paths
    # ------------------------------------------------------------------

    def _late(self, bundle: Bundle) -> None:
        self._finalize_keys(AdmissionPrefetchMetrics.GATE_REJECT, bundle.resolved_run)
        self._finalize_keys(
            AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED,
            self._pending_key_count(bundle),
        )
        self._bundle_outcome("late")
        self._set_state(bundle, BundleState.LATE)

    def _cancel(self, bundle: Bundle, outcome: str) -> None:
        self._finalize_keys(AdmissionPrefetchMetrics.CANCELLED, bundle.resolved_run)
        self._finalize_keys(
            AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED,
            self._pending_key_count(bundle),
        )
        self._bundle_outcome(outcome)
        self._set_state(bundle, BundleState.CANCELLED)

    def _gate_reject(self, bundle: Bundle, outcome: str) -> None:
        self._finalize_keys(AdmissionPrefetchMetrics.GATE_REJECT, bundle.resolved_run)
        self._bundle_outcome(outcome)
        self._set_state(bundle, BundleState.GATE_REJECTED)

    # ------------------------------------------------------------------
    # PrefetchPolicy interface
    # ------------------------------------------------------------------

    def on_request_enqueued(self, req_context: ReqContext) -> None:
        req_id = req_context.req_id
        if req_id in self._admitted_unscheduled:
            return
        self._admitted_unscheduled[req_id] = QueuedRequest(
            admitted_at=self.clock(),
            queue_position=len(self._admitted_unscheduled),
        )

    def on_request_admitted(
        self, req_context: ReqContext, offload_keys: Sequence[OffloadKey]
    ) -> None:
        req_id = req_context.req_id
        if req_id in self._bundles:
            return
        queued = self._admitted_unscheduled.get(req_id)
        if queued is None:
            self.on_request_enqueued(req_context)
            queued = self._admitted_unscheduled[req_id]
        queued.is_prefetch_candidate = True

        lead_time_ms = self._estimator.predict_ms(queued.queue_position)
        self._stats.observe_histogram(
            AdmissionPrefetchMetrics.LEAD_TIME, lead_time_ms / 1000.0
        )

        window = list(offload_keys[: self.config.max_candidate_chunks])
        frontier = 0
        for key in window:
            result = self.host.prefetch_primary_lookup(key, req_context)
            if result in (LookupResult.HIT, LookupResult.HIT_PENDING):
                frontier += 1
            else:
                break
        self._finalize_keys(AdmissionPrefetchMetrics.PRIMARY_REDUNDANT, frontier)
        if frontier == len(window):
            return

        if not self.host.prefetch_tier_allowed(self.config.tier_idx, req_context):
            return

        if len(self._active) >= self.config.max_pending_bundles:
            self._finalize_keys(
                AdmissionPrefetchMetrics.CAPACITY_SKIP, len(window) - frontier
            )
            self._bundle_outcome("capacity_skip")
            return

        bundle = Bundle(
            req_id=req_id,
            req_context=req_context,
            tier_idx=self.config.tier_idx,
            keys=window[frontier:],
            admitted_at=queued.admitted_at,
            lead_time_ms=lead_time_ms,
        )
        self._bundles[req_id] = bundle
        self._active[req_id] = None
        for key in bundle.keys:
            self.host.prefetch_secondary_lookup(bundle.tier_idx, key, req_context)

    def step(self, context: ScheduleEndContext) -> None:
        now = self.clock()

        scheduled_count = 0
        for req_id in context.new_req_ids:
            queued = self._admitted_unscheduled.pop(req_id, None)
            if queued is not None:
                scheduled_count += 1
                if queued.is_prefetch_candidate:
                    self._stats.observe_histogram(
                        AdmissionPrefetchMetrics.ACTUAL_LEAD_TIME,
                        max(0.0, now - queued.admitted_at),
                    )
            bundle = self._bundles.get(req_id)
            if bundle is not None and bundle.state is not BundleState.SUBMITTED:
                # First demand ran during this step's scheduling pass, so
                # any unsubmitted bundle missed its window.
                self._late(bundle)

        self._estimator.on_first_scheduled(
            now,
            scheduled_count,
            queue_remains_nonempty=bool(self._admitted_unscheduled),
        )

        for req_id in context.preempted_req_ids:
            bundle = self._bundles.get(req_id)
            if bundle is not None and bundle.state is not BundleState.SUBMITTED:
                self._cancel(bundle, "cancelled_preempted")

        # Earliest deadline first. Admission order would hand the budget to the
        # requests closest to being scheduled -- the ones with the least lead
        # time and so the least to gain -- and make the bundles that can still
        # be hidden wait behind them.
        drivable = [
            bundle
            for bundle in (self._bundles.get(req_id) for req_id in self._active)
            if bundle is not None and bundle.state is not BundleState.SUBMITTED
        ]
        drivable.sort(key=_bundle_deadline)

        step_key_budget = self.config.max_promotions_per_step
        for bundle in drivable:
            step_key_budget = self._drive_bundle(bundle, now, step_key_budget)

    def _drive_bundle(self, bundle: Bundle, now: float, key_budget: int) -> int:
        """Re-drive one bundle; returns the remaining per-step key budget."""
        while not bundle.absent_found and bundle.resolved_run < len(bundle.keys):
            key = bundle.keys[bundle.resolved_run]
            result = self.host.prefetch_secondary_lookup(
                bundle.tier_idx, key, bundle.req_context
            )
            if result is LookupResult.HIT:
                bundle.resolved_run += 1
            elif result is LookupResult.MISS:
                self._finalize_keys(AdmissionPrefetchMetrics.SECONDARY_ABSENT, 1)
                bundle.absent_found = True
            else:
                break

        # Deadline is checked before the gate with one clock snapshot per
        # step, so resolving and expiring in the same step is always LATE.
        h_remaining_ms = (bundle.deadline - now) * 1000.0
        if h_remaining_ms <= 0:
            self._late(bundle)
            return key_budget

        fully_resolved = bundle.absent_found or bundle.resolved_run == len(bundle.keys)
        if not fully_resolved:
            return key_budget

        if bundle.resolved_run == 0:
            self._bundle_outcome("absent")
            self._set_state(bundle, BundleState.ABSENT)
            return key_budget

        if bundle.state is BundleState.PENDING_LOOKUP:
            self._set_state(bundle, BundleState.RESIDENT)

        # Hard caps are limits no later step can relax, so anything above them
        # is genuinely unpromotable and terminalizes here.
        hard_cap = self.config.max_promotions_per_step
        if self.config.speculative_max_bytes > 0 and self.config.chunk_bytes > 0:
            hard_cap = min(
                hard_cap,
                self.config.speculative_max_bytes // self.config.chunk_bytes,
            )
        if bundle.resolved_run > hard_cap:
            self._finalize_keys(
                AdmissionPrefetchMetrics.CAPACITY_SKIP,
                bundle.resolved_run - hard_cap,
            )
            del bundle.keys[hard_cap:]
            bundle.resolved_run = hard_cap

        if bundle.resolved_run == 0:
            self._bundle_outcome("capacity_skip")
            self._set_state(bundle, BundleState.CAPACITY_SKIPPED)
            return key_budget

        # Contention is different: the shortfall is only what earlier bundles
        # already spent this step, and the next step restores it. Defer rather
        # than spending the request's one chance on step-arrival order. The
        # deadline keeps running, so a bundle that never fits resolves as LATE.
        available_now = key_budget
        if self.config.speculative_max_bytes > 0 and self.config.chunk_bytes > 0:
            free_bytes = max(
                0,
                self.config.speculative_max_bytes - self._inflight_speculative_bytes,
            )
            available_now = min(available_now, free_bytes // self.config.chunk_bytes)
        if bundle.resolved_run > available_now:
            return key_budget

        bundle_len = bundle.resolved_run
        # Cost is measured from real promotions on this tier, so a busy tier
        # reports longer transfers and the deadline tightens on its own. That
        # feedback is what protects the active workload; there is no separate
        # utility term until a contention cost exists that is worth measuring.
        prefetch_latency_ms = self.host.prefetch_transfer_cost_ms(
            bundle.tier_idx, bundle_len
        )
        # Emitted for accepted and rejected bundles alike: the margin
        # distribution is what shows whether lead time is the binding
        # constraint or there is room to spare.
        self._stats.observe_histogram(AdmissionPrefetchMetrics.BUNDLE_SIZE, bundle_len)
        self._stats.observe_histogram(
            AdmissionPrefetchMetrics.DEADLINE_MARGIN,
            (h_remaining_ms - prefetch_latency_ms) / 1000.0,
        )
        if h_remaining_ms <= prefetch_latency_ms:
            self._gate_reject(bundle, "gate_reject_deadline")
            return key_budget

        logger.debug(
            "admission prefetch decision: req_id=%s bundle_len=%d "
            "H_remaining_ms=%.3f L_prefetch_ms=%.3f shadow=%s",
            bundle.req_id,
            bundle_len,
            h_remaining_ms,
            prefetch_latency_ms,
            self.config.shadow_mode,
        )

        if self.config.shadow_mode:
            self._finalize_keys(AdmissionPrefetchMetrics.SHADOW_SUBMIT, bundle_len)
            self._bundle_outcome("shadow_submit")
            self._set_state(bundle, BundleState.SHADOW_SUBMITTED)
            return key_budget - bundle_len

        submit_result = self.host.prefetch_submit(
            bundle.tier_idx, bundle.keys[:bundle_len], bundle.req_context
        )
        self._finalize_keys(
            AdmissionPrefetchMetrics.PRIMARY_REDUNDANT,
            len(submit_result.primary_redundant),
        )
        self._finalize_keys(
            AdmissionPrefetchMetrics.CAPACITY_SKIP, len(submit_result.capacity_skipped)
        )
        self._finalize_keys(
            AdmissionPrefetchMetrics.SUBMITTED, len(submit_result.submitted)
        )
        if submit_result.submitted:
            bundle.outstanding = set(submit_result.submitted)
            for key in submit_result.submitted:
                self._submitted_key_owner[key] = bundle.req_id
            self._inflight_speculative_bytes += (
                len(submit_result.submitted) * self.config.chunk_bytes
            )
            self._set_state(bundle, BundleState.SUBMITTED)
        elif submit_result.capacity_skipped:
            self._bundle_outcome("capacity_skip")
            self._set_state(bundle, BundleState.CAPACITY_SKIPPED)
        else:
            self._bundle_outcome("redundant")
            self._set_state(bundle, BundleState.CAPACITY_SKIPPED)
        return key_budget - len(submit_result.submitted)

    def on_prefetch_demand(self, key: OffloadKey, primary_result: LookupResult) -> None:
        if primary_result is not LookupResult.HIT_PENDING:
            return
        req_id = self._submitted_key_owner.get(key)
        if req_id is None:
            return
        bundle = self._bundles.get(req_id)
        if bundle is not None:
            bundle.demanded_while_pending = True

    def on_promotion_finished(self, keys: Sequence[OffloadKey], success: bool) -> None:
        touched: set[str] = set()
        for key in keys:
            req_id = self._submitted_key_owner.pop(key, None)
            if req_id is None:
                continue
            bundle = self._bundles.get(req_id)
            if bundle is None:
                continue
            bundle.outstanding.discard(key)
            if not success:
                bundle.any_load_failed = True
            self._inflight_speculative_bytes -= self.config.chunk_bytes
            touched.add(req_id)
        for req_id in touched:
            bundle = self._bundles.get(req_id)
            if bundle is None or bundle.outstanding:
                continue
            if bundle.any_load_failed:
                self._bundle_outcome("failed")
                self._set_state(bundle, BundleState.FAILED)
            elif bundle.demanded_while_pending:
                self._bundle_outcome("late")
                self._set_state(bundle, BundleState.LATE)
            else:
                self._bundle_outcome("ready")
                self._set_state(bundle, BundleState.READY)

    def on_request_finished(self, req_id: str) -> None:
        self._admitted_unscheduled.pop(req_id, None)
        if not self._admitted_unscheduled:
            self._estimator.on_queue_idle()
        bundle = self._bundles.get(req_id)
        if bundle is not None and bundle.state is not BundleState.SUBMITTED:
            # Removing the bundle guarantees no further lookups are issued
            # for this request, so the tier's AsyncLookupManager cleanup
            # cannot be repopulated after it runs.
            self._cancel(bundle, "cancelled_finished")

    def has_pending_work(self) -> bool:
        return bool(self._active)

    def reset(self) -> None:
        assert not self._submitted_key_owner, (
            "reset() requires all submitted promotions to be drained first"
        )
        for req_id in list(self._active):
            bundle = self._bundles.get(req_id)
            if bundle is not None:
                self._cancel(bundle, "cancelled_reset")
        self._admitted_unscheduled.clear()
        self._inflight_speculative_bytes = 0
        self._estimator.reset()
