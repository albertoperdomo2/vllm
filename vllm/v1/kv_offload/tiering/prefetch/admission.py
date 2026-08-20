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
    QUEUED = enum.auto()
    PENDING_LOOKUP = enum.auto()
    RESIDENT = enum.auto()
    # Some keys submitted, more of the resolved run still to go. Non-terminal
    # and drivable, so the remainder is picked up on later steps.
    SUBMITTING = enum.auto()
    SUBMITTED = enum.auto()
    READY = enum.auto()
    LATE = enum.auto()
    ABSENT = enum.auto()
    FAILED = enum.auto()
    GATE_REJECTED = enum.auto()
    CAPACITY_SKIPPED = enum.auto()
    REDUNDANT = enum.auto()
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
        BundleState.REDUNDANT,
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
    # Next index in keys[] still to be dispositioned. Everything in
    # keys[0:cursor] has been counted exactly once by _finalize_keys, which is
    # what keeps the terminal partition exact when a bundle submits in slices.
    cursor: int = 0
    # BUNDLE_SIZE is observed once per bundle, not once per slice.
    gate_observed: bool = False
    # Set when a bundle is abandoned while keys are still in flight; the
    # bundle-level outcome is emitted once those keys resolve.
    abandoned_outcome: str | None = None
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
        self._estimator = LeadTimeEstimator(config)
        self._tier_label = host.prefetch_tier_label(config.tier_idx)
        self._owner_req_id: str | None = None
        self._owner_deadline: float | None = None

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
        if old_state is new_state:
            # Successive slices of one bundle re-enter SUBMITTING; counting
            # that as a transition would drown the real ones.
            return
        bundle.state = new_state
        self._stats.increase_counter(
            AdmissionPrefetchMetrics.TRANSITIONS,
            labelvalues=(f"{old_state.name.lower()}->{new_state.name.lower()}",),
        )
        if new_state in _TERMINAL_STATES:
            if self.config.jit_activation and new_state is not BundleState.READY:
                self._release_owner(
                    bundle.req_id, reason=f"terminal_{new_state.name.lower()}"
                )
            self._active.pop(bundle.req_id, None)
            del self._bundles[bundle.req_id]

    def _release_owner(
        self,
        req_id: str,
        *,
        demanded: bool = False,
        reason: str = "terminal",
    ) -> None:
        if self._owner_req_id != req_id:
            return
        self.host.prefetch_release_owner(req_id, demanded=demanded)
        self._stats.increase_counter(
            AdmissionPrefetchMetrics.OWNER_RELEASES,
            labelvalues=(self._tier_label[0], reason),
        )
        self._owner_req_id = None
        self._owner_deadline = None
        self._stats.set_gauge(
            AdmissionPrefetchMetrics.ACTIVE_OWNER,
            0.0,
            labelvalues=self._tier_label,
        )

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

    def _close_resolved(self, bundle: Bundle, terminal_counter: str) -> None:
        """Dispose of the not-yet-counted part of the resolved run.

        Idempotent: advancing the cursor means a second call counts nothing,
        which is what lets a bundle terminalize in pieces without ever
        double-counting a key.
        """
        self._finalize_keys(terminal_counter, bundle.resolved_run - bundle.cursor)
        bundle.cursor = bundle.resolved_run

    def _close_pending(self, bundle: Bundle) -> None:
        """Dispose of keys whose residency probe never resolved."""
        self._finalize_keys(
            AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED,
            self._pending_key_count(bundle),
        )
        del bundle.keys[bundle.resolved_run :]

    def _abandon(
        self,
        bundle: Bundle,
        terminal_counter: str,
        outcome: str,
        terminal_state: BundleState,
    ) -> None:
        """Close a bundle's accounting, parking it if keys are still in flight.

        A partially submitted bundle cannot be dropped from _bundles: that
        would orphan its _submitted_key_owner entries. The per-key partition
        closes now; the bundle-level outcome waits for the in-flight keys.
        """
        if bundle.state is BundleState.QUEUED:
            # JIT admission has issued no lookup yet, so cancellation or a
            # pre-gate rejection owns the whole bundle. Calling these keys
            # unresolved would incorrectly attribute work that never started.
            self._finalize_keys(terminal_counter, len(bundle.keys) - bundle.cursor)
            bundle.cursor = len(bundle.keys)
        else:
            self._close_resolved(bundle, terminal_counter)
            self._close_pending(bundle)
        if bundle.outstanding:
            bundle.abandoned_outcome = outcome
            self._set_state(bundle, BundleState.SUBMITTED)
            return
        self._bundle_outcome(outcome)
        self._set_state(bundle, terminal_state)

    def _late(self, bundle: Bundle) -> None:
        self._abandon(
            bundle,
            AdmissionPrefetchMetrics.GATE_REJECT,
            "late",
            BundleState.LATE,
        )

    def _cancel(self, bundle: Bundle, outcome: str) -> None:
        self._abandon(
            bundle,
            AdmissionPrefetchMetrics.CANCELLED,
            outcome,
            BundleState.CANCELLED,
        )

    def _gate_reject(self, bundle: Bundle, outcome: str) -> None:
        self._abandon(
            bundle,
            AdmissionPrefetchMetrics.GATE_REJECT,
            outcome,
            BundleState.GATE_REJECTED,
        )

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
                AdmissionPrefetchMetrics.BUNDLE_OVERFLOW, len(window) - frontier
            )
            self._bundle_outcome("bundle_overflow")
            return

        # Probe only what could actually be submitted. Probing the whole
        # window lengthens the tier's lookup batch and pushes results past
        # bundle deadlines, which is what left 29% of candidate keys
        # unresolved in the first run.
        bundle_keys = window[frontier : frontier + self.config.max_bundle_chunks]
        # The ceiling bites here, not after probing: it is what bounds the
        # probe, so nothing past it is ever looked up. Counting the excess now
        # keeps `considered` equal to the candidate window and is the signal
        # that says whether max_bundle_chunks is the binding constraint --
        # the question the first run could not answer.
        self._finalize_keys(
            AdmissionPrefetchMetrics.BUNDLE_TRIM,
            len(window) - frontier - len(bundle_keys),
        )
        bundle = Bundle(
            req_id=req_id,
            req_context=req_context,
            tier_idx=self.config.tier_idx,
            keys=bundle_keys,
            admitted_at=queued.admitted_at,
            lead_time_ms=lead_time_ms,
        )
        if self.config.jit_activation:
            bundle.state = BundleState.QUEUED
        self._bundles[req_id] = bundle
        self._active[req_id] = None
        if not self.config.jit_activation:
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
            self._release_owner(req_id, demanded=True, reason="scheduled")
            bundle = self._bundles.get(req_id)
            if bundle is not None:
                # First demand ran during this step's scheduling pass, so
                # any unfinished bundle missed its overlap window.
                self._late(bundle)

        self._estimator.on_first_scheduled(
            now,
            scheduled_count,
            queue_remains_nonempty=bool(self._admitted_unscheduled),
        )

        for req_id in context.preempted_req_ids:
            bundle = self._bundles.get(req_id)
            self._release_owner(req_id, reason="preempted")
            if bundle is not None:
                self._cancel(bundle, "cancelled_preempted")

        if (
            self.config.jit_activation
            and self._owner_req_id is not None
            and self._owner_deadline is not None
            and now >= self._owner_deadline
        ):
            owner_req_id = self._owner_req_id
            bundle = self._bundles.get(owner_req_id)
            self._release_owner(owner_req_id, reason="expired")
            if bundle is not None:
                self._late(bundle)

        activated_req_id = None
        if self.config.jit_activation and self._owner_req_id is None:
            activated_req_id = self._activate_next_bundle(now)

        # Earliest deadline first keeps JIT residency close to demand and avoids
        # retaining a farther-future request while an earlier one waits.
        drivable = [
            bundle
            for bundle in (self._bundles.get(req_id) for req_id in self._active)
            if bundle is not None
            and bundle.state not in (BundleState.QUEUED, BundleState.SUBMITTED)
            and bundle.req_id != activated_req_id
            and (not self.config.jit_activation or bundle.req_id == self._owner_req_id)
        ]
        drivable.sort(key=_bundle_deadline)

        step_key_budget = self.config.max_promotions_per_step
        for bundle in drivable:
            step_key_budget = self._drive_bundle(bundle, now, step_key_budget)
        self._stats.set_gauge(
            AdmissionPrefetchMetrics.ACTIVE_OWNER,
            1.0 if self._owner_req_id is not None else 0.0,
            labelvalues=self._tier_label,
        )

    def _activate_next_bundle(self, now: float) -> str | None:
        queued = [
            bundle
            for bundle in self._bundles.values()
            if bundle.state is BundleState.QUEUED
        ]
        if not queued:
            return None
        if not self.host.prefetch_speculation_idle(self.config.tier_idx):
            self._stats.increase_counter(
                AdmissionPrefetchMetrics.ACTIVATION_DEFERRED,
                labelvalues=(self._tier_label[0], "speculation_busy"),
            )
            return None
        if self.config.demand_idle_only and not self.host.prefetch_demand_idle(
            self.config.tier_idx
        ):
            self._stats.increase_counter(
                AdmissionPrefetchMetrics.ACTIVATION_DEFERRED,
                labelvalues=(self._tier_label[0], "demand_busy"),
            )
            return None

        bundle = min(queued, key=_bundle_deadline)
        h_remaining_ms = (bundle.deadline - now) * 1000.0
        if h_remaining_ms <= 0:
            self._late(bundle)
            return None
        prefetch_latency_ms = self.host.prefetch_transfer_cost_ms(
            bundle.tier_idx, len(bundle.keys)
        )
        if h_remaining_ms <= prefetch_latency_ms:
            self._stats.observe_histogram(
                AdmissionPrefetchMetrics.DEADLINE_MARGIN,
                (h_remaining_ms - prefetch_latency_ms) / 1000.0,
            )
            self._gate_reject(bundle, "gate_reject_jit_deadline")
            return None

        self._owner_req_id = bundle.req_id
        self._owner_deadline = bundle.deadline
        self._stats.set_gauge(
            AdmissionPrefetchMetrics.ACTIVE_OWNER,
            1.0,
            labelvalues=self._tier_label,
        )
        self._set_state(bundle, BundleState.PENDING_LOOKUP)
        results = []
        for key in bundle.keys:
            results.append(
                self.host.prefetch_secondary_lookup(
                    bundle.tier_idx, key, bundle.req_context
                )
            )
        for result in results:
            if result is LookupResult.HIT:
                bundle.resolved_run += 1
            elif result is LookupResult.MISS:
                self._finalize_keys(AdmissionPrefetchMetrics.SECONDARY_ABSENT, 1)
                bundle.absent_found = True
                break
            else:
                break
        return bundle.req_id

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

        # The per-bundle ceiling was already applied at admission, which is
        # also where its excess was counted -- the bundle can never hold more
        # keys than the ceiling, so there is nothing left to trim here.
        #
        # Step budget is different: the shortfall is only what earlier bundles
        # already spent this step, and the next step restores it. Submit what
        # fits and carry the rest, rather than discarding a verified-resident
        # prefix. The deadline keeps running, so a bundle that never drains
        # resolves as LATE.
        remaining = bundle.resolved_run - bundle.cursor
        if key_budget <= 0:
            return key_budget
        if self.config.jit_activation and bundle.outstanding:
            return key_budget
        if (
            self.config.jit_activation
            and self.config.demand_idle_only
            and not self.host.prefetch_demand_idle(bundle.tier_idx)
        ):
            return key_budget
        bundle_len = min(remaining, key_budget)
        # Cost is measured from real promotions on this tier, so a busy tier
        # reports longer transfers and the deadline tightens on its own. That
        # feedback is what protects the active workload; there is no separate
        # utility term until a contention cost exists that is worth measuring.
        # Gate on the whole remaining run, not just this step's slice: the
        # question is whether the rest of this prefix can still be hidden, so
        # a bundle that would only dribble is cut off honestly.
        prefetch_latency_ms = self.host.prefetch_transfer_cost_ms(
            bundle.tier_idx, remaining
        )
        if not bundle.gate_observed:
            # Once per bundle, or multi-step bundles inflate the histogram.
            self._stats.observe_histogram(
                AdmissionPrefetchMetrics.BUNDLE_SIZE, bundle.resolved_run
            )
            bundle.gate_observed = True
        # Emitted for accepted and rejected decisions alike: the margin
        # distribution is what shows whether lead time is the binding
        # constraint or there is room to spare.
        self._stats.observe_histogram(
            AdmissionPrefetchMetrics.DEADLINE_MARGIN,
            (h_remaining_ms - prefetch_latency_ms) / 1000.0,
        )
        if h_remaining_ms <= prefetch_latency_ms:
            self._gate_reject(bundle, "gate_reject_deadline")
            return key_budget

        logger.debug(
            "admission prefetch decision: req_id=%s slice=%d remaining=%d "
            "H_remaining_ms=%.3f L_prefetch_ms=%.3f shadow=%s",
            bundle.req_id,
            bundle_len,
            remaining,
            h_remaining_ms,
            prefetch_latency_ms,
            self.config.shadow_mode,
        )

        if self.config.shadow_mode:
            # Shadow exists to predict live, so it must drain on the same
            # schedule: one slice per step, re-gated each time. Disposing of
            # the whole run at once made shadow optimistic -- it never
            # re-checked the deadline for slices live would still be
            # submitting several steps later, and it charged the step budget
            # for I/O that had not happened yet.
            self._finalize_keys(AdmissionPrefetchMetrics.SHADOW_SUBMIT, bundle_len)
            bundle.cursor += bundle_len
            if bundle.cursor >= bundle.resolved_run:
                self._bundle_outcome("shadow_submit")
                self._set_state(bundle, BundleState.SHADOW_SUBMITTED)
            return key_budget - bundle_len

        start = bundle.cursor
        submit_result = self.host.prefetch_submit(
            bundle.tier_idx, bundle.keys[start : start + bundle_len], bundle.req_context
        )
        self._finalize_keys(
            AdmissionPrefetchMetrics.PRIMARY_REDUNDANT,
            len(submit_result.primary_redundant),
        )
        self._finalize_keys(
            AdmissionPrefetchMetrics.ALLOC_REFUSED,
            len(submit_result.capacity_skipped),
        )
        self._finalize_keys(
            AdmissionPrefetchMetrics.SUBMITTED, len(submit_result.submitted)
        )
        bundle.cursor += bundle_len
        if submit_result.submitted:
            bundle.outstanding |= set(submit_result.submitted)
            for key in submit_result.submitted:
                self._submitted_key_owner[key] = bundle.req_id

        if submit_result.capacity_skipped:
            # prefetch_submit stops at the first refusal; anything past it
            # would leave a hole the demand scan cannot cross.
            self._close_resolved(bundle, AdmissionPrefetchMetrics.ALLOC_REFUSED)

        if bundle.cursor >= bundle.resolved_run:
            if bundle.outstanding:
                self._set_state(bundle, BundleState.SUBMITTED)
            elif submit_result.capacity_skipped:
                self._bundle_outcome("alloc_refused")
                self._set_state(bundle, BundleState.CAPACITY_SKIPPED)
            else:
                self._bundle_outcome("redundant")
                self._set_state(bundle, BundleState.REDUNDANT)
        else:
            self._set_state(bundle, BundleState.SUBMITTING)
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
            touched.add(req_id)
        for req_id in touched:
            bundle = self._bundles.get(req_id)
            if bundle is None or bundle.outstanding:
                continue
            if bundle.cursor < bundle.resolved_run:
                # A slice landed but the bundle still has run left to submit;
                # it stays drivable rather than terminalizing early.
                self._set_state(bundle, BundleState.SUBMITTING)
                continue
            if bundle.abandoned_outcome is not None:
                # Cancelled or late while keys were still moving; its per-key
                # accounting closed then, only the outcome was owed.
                self._bundle_outcome(bundle.abandoned_outcome)
                self._set_state(bundle, BundleState.CANCELLED)
            elif bundle.any_load_failed:
                self._bundle_outcome("failed")
                self._release_owner(bundle.req_id, reason="promotion_failed")
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
        self._release_owner(req_id, reason="finished")
        if bundle is not None:
            # Removing the bundle guarantees no further lookups are issued
            # for this request, so the tier's AsyncLookupManager cleanup
            # cannot be repopulated after it runs.
            self._cancel(bundle, "cancelled_finished")

    def has_pending_work(self) -> bool:
        return bool(self._active) or self._owner_req_id is not None

    def reset(self) -> None:
        assert not self._submitted_key_owner, (
            "reset() requires all submitted promotions to be drained first"
        )
        if self._owner_req_id is not None:
            self._release_owner(self._owner_req_id, reason="reset")
        for req_id in list(self._active):
            bundle = self._bundles.get(req_id)
            if bundle is not None:
                self._cancel(bundle, "cancelled_reset")
        self._admitted_unscheduled.clear()
        self._estimator.reset()
