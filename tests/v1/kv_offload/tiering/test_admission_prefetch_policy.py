# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2.1 admission prefetch policy.

These exercise the policy in isolation through a fake host, so every state
transition, deadline race, and accounting path is testable without a real
tiering manager or storage backend.
"""

import random

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.prefetch.admission import (
    AdmissionPrefetchPolicy,
    BundleState,
)
from vllm.v1.kv_offload.tiering.prefetch.base import (
    AdmissionPrefetchMetrics,
    AdmissionSubmitResult,
)
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig
from vllm.v1.kv_offload.tiering.prefetch.estimators import (
    LeadTimeEstimator,
    TransferCostModel,
)

TIER_LABEL = ("1:example",)


def to_keys(count, prefix="k"):
    return [make_offload_key(f"{prefix}{i}".encode(), 0) for i in range(count)]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += ms / 1000.0


class FakeHost:
    """Records every host call and serves scripted lookup verdicts."""

    def __init__(self):
        self.primary = {}
        self.secondary = {}
        self.tier_allowed = True
        self.primary_lookups = []
        self.secondary_lookups = []
        self.submits = []
        # Keys the next submit should report as redundant/capacity-skipped.
        self.submit_redundant = set()
        self.submit_capacity_skipped = set()
        # Transfer cost the host reports; the manager measures this from real
        # promotions, so tests set it directly.
        self.transfer_base_ms = 0.5
        self.transfer_per_chunk_ms = 2.2

    def prefetch_primary_lookup(self, key, req_context):
        self.primary_lookups.append(key)
        return self.primary.get(key, LookupResult.MISS)

    def prefetch_secondary_lookup(self, tier_idx, key, req_context):
        self.secondary_lookups.append(key)
        return self.secondary.get(key, LookupResult.RETRY)

    def prefetch_tier_allowed(self, tier_idx, req_context):
        return self.tier_allowed

    def prefetch_tier_label(self, tier_idx):
        return TIER_LABEL

    def prefetch_transfer_cost_ms(self, tier_idx, n_chunks):
        return self.transfer_base_ms + self.transfer_per_chunk_ms * n_chunks

    def prefetch_submit(self, tier_idx, keys, req_context):
        keys = list(keys)
        self.submits.append(keys)
        result = AdmissionSubmitResult()
        for key in keys:
            if key in self.submit_redundant:
                result.primary_redundant.append(key)
            elif key in self.submit_capacity_skipped:
                result.capacity_skipped.append(key)
            else:
                result.submitted.append(key)
        return result


def make_policy(host=None, clock=None, **config_kwargs):
    config_kwargs.setdefault("enabled", True)
    config_kwargs.setdefault("shadow_mode", False)
    # Generous default lead time so tests opt in to deadline pressure.
    config_kwargs.setdefault("initial_admission_interval_ms", 10_000.0)
    config = PrefetchConfig(**config_kwargs)
    host = host or FakeHost()
    clock = clock or FakeClock()
    return AdmissionPrefetchPolicy(config, host, clock), host, clock


def counter(policy, name, labelvalues=TIER_LABEL):
    return policy._stats._values.get(name, {}).get(labelvalues, 0)


def counter_any_label(policy, name):
    return sum(policy._stats._values.get(name, {}).values())


def assert_partition(policy):
    """The per-key terminal partition must always close."""
    considered = counter(policy, AdmissionPrefetchMetrics.CONSIDERED)
    terminal = sum(
        counter(policy, name) for name in AdmissionPrefetchMetrics.TERMINAL_COUNTERS
    )
    assert considered == terminal, (
        f"partition violated: considered={considered} terminal={terminal}"
    )


def fill_queue(policy, count, tag="filler"):
    """Admit keyless requests to create queue depth.

    Lead time is predicted as queue_position * mean admission interval, so a
    request admitted to an empty queue predicts H = 0 and is correctly never
    prefetched. Tests that want a submission must first establish depth.
    Keyless requests add depth without creating bundles.
    """
    for i in range(count):
        ctx = ReqContext(req_id=f"{tag}{i}", kv_transfer_params=None)
        policy.on_request_enqueued(ctx)


def admit(policy, host, req_id, keys, queue_ahead=3):
    fill_queue(policy, queue_ahead, tag=f"{req_id}-ahead")
    ctx = ReqContext(req_id=req_id, kv_transfer_params=None)
    policy.on_request_enqueued(ctx)
    policy.on_request_admitted(ctx, keys)
    return ctx


def resolve_all(host, keys, result=LookupResult.HIT):
    for key in keys:
        host.secondary[key] = result


def empty_step():
    return ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


class TestSelection:
    def test_frontier_skips_primary_resident_prefix(self):
        policy, host, _ = make_policy()
        keys = to_keys(5)
        host.primary[keys[0]] = LookupResult.HIT
        host.primary[keys[1]] = LookupResult.HIT_PENDING
        admit(policy, host, "r0", keys)

        assert counter(policy, AdmissionPrefetchMetrics.PRIMARY_REDUNDANT) == 2
        # Only post-frontier keys become candidates.
        assert host.secondary_lookups == keys[2:]
        assert_partition(policy)

    def test_all_primary_resident_creates_no_bundle(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        for key in keys:
            host.primary[key] = LookupResult.HIT
        admit(policy, host, "r0", keys)

        assert not policy.has_pending_work()
        assert host.secondary_lookups == []
        assert counter(policy, AdmissionPrefetchMetrics.PRIMARY_REDUNDANT) == 3
        assert_partition(policy)

    def test_bundle_stops_at_first_absent_key(self):
        policy, host, _ = make_policy()
        keys = to_keys(5)
        resolve_all(host, keys[:2], LookupResult.HIT)
        host.secondary[keys[2]] = LookupResult.MISS
        resolve_all(host, keys[3:], LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        # Contiguity: only the run before the absent key is submitted, and
        # keys after it are never counted at all.
        assert host.submits == [keys[:2]]
        assert counter(policy, AdmissionPrefetchMetrics.SECONDARY_ABSENT) == 1
        assert counter(policy, AdmissionPrefetchMetrics.SUBMITTED) == 2
        assert counter(policy, AdmissionPrefetchMetrics.CONSIDERED) == 3
        assert_partition(policy)

    def test_absent_at_frontier_yields_absent_bundle(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        host.secondary[keys[0]] = LookupResult.MISS
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        assert host.submits == []
        assert counter(policy, AdmissionPrefetchMetrics.SECONDARY_ABSENT) == 1
        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_admission_scan_capped_by_max_candidate_chunks(self):
        policy, host, _ = make_policy(max_candidate_chunks=4)
        keys = to_keys(50)
        admit(policy, host, "r0", keys)

        # The candidate window bounds the scan regardless of prompt length.
        assert len(host.secondary_lookups) == 4
        assert policy._bundles["r0"].keys == keys[:4]

    def test_frontier_scan_stops_at_first_non_resident_key(self):
        policy, host, _ = make_policy()
        keys = to_keys(20)
        host.primary[keys[0]] = LookupResult.HIT
        admit(policy, host, "r0", keys)

        # Scanning past the frontier would be wasted work: the demand path
        # breaks there too.
        assert host.primary_lookups == keys[:2]

    def test_disallowed_tier_creates_no_bundle(self):
        policy, host, _ = make_policy()
        host.tier_allowed = False
        admit(policy, host, "r0", to_keys(3))

        assert not policy.has_pending_work()
        assert host.secondary_lookups == []


class TestAsyncResidency:
    def test_retry_keeps_bundle_pending_not_absent(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        # Pending lookups must never be counted as absence.
        assert counter(policy, AdmissionPrefetchMetrics.SECONDARY_ABSENT) == 0
        assert host.submits == []
        assert policy.has_pending_work()
        assert policy._bundles["r0"].state is BundleState.PENDING_LOOKUP

    def test_multi_step_resolution_then_submit(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        host.secondary[keys[0]] = LookupResult.HIT
        policy.step(empty_step())
        assert host.submits == []

        resolve_all(host, keys[1:], LookupResult.HIT)
        policy.step(empty_step())

        assert host.submits == [keys]
        assert counter(policy, AdmissionPrefetchMetrics.SUBMITTED) == 3
        assert_partition(policy)

    def test_no_lookup_reissued_after_finish(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        admit(policy, host, "r0", keys)
        policy.on_request_finished("r0")
        before = len(host.secondary_lookups)
        policy.step(empty_step())
        policy.step(empty_step())

        # Re-issuing lookups after finish would repopulate the tier's
        # AsyncLookupManager state after its cleanup() already ran.
        assert len(host.secondary_lookups) == before
        assert not policy.has_pending_work()
        assert_partition(policy)


class TestDeadlines:
    def test_expiry_while_pending_is_late_and_never_submitted(self):
        policy, host, clock = make_policy(initial_admission_interval_ms=100.0)
        keys = to_keys(2)
        # Queue position 0 => H = 0 => no lead time to hide a transfer.
        admit(policy, host, "r0", keys, queue_ahead=0)
        policy.step(empty_step())

        assert host.submits == []
        assert counter_any_label(policy, AdmissionPrefetchMetrics.BUNDLE_OUTCOMES)
        assert counter(policy, AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED) == 2
        assert not policy.has_pending_work()
        assert_partition(policy)


class TestLeadTimeEstimator:
    def test_batch_is_one_throughput_sample(self):
        estimator = LeadTimeEstimator(
            PrefetchConfig(
                initial_admission_interval_ms=50.0,
                admission_interval_ewma_alpha=1.0,
            )
        )

        estimator.on_first_scheduled(0.0, 32, queue_remains_nonempty=True)
        estimator.on_first_scheduled(0.1, 32, queue_remains_nonempty=True)

        assert estimator.predict_ms(32) == pytest.approx(100.0)

    def test_idle_time_does_not_inflate_next_sample(self):
        estimator = LeadTimeEstimator(
            PrefetchConfig(
                initial_admission_interval_ms=50.0,
                admission_interval_ewma_alpha=1.0,
            )
        )
        estimator.on_first_scheduled(0.0, 32, queue_remains_nonempty=True)
        estimator.on_first_scheduled(0.1, 32, queue_remains_nonempty=True)
        estimator.on_queue_idle()

        estimator.on_first_scheduled(3600.0, 1, queue_remains_nonempty=True)

        assert estimator.predict_ms(32) == pytest.approx(100.0)

    def test_reset_restores_initial_interval(self):
        estimator = LeadTimeEstimator(
            PrefetchConfig(
                initial_admission_interval_ms=50.0,
                admission_interval_ewma_alpha=1.0,
            )
        )
        estimator.on_first_scheduled(0.0, 1, queue_remains_nonempty=True)
        estimator.on_first_scheduled(1.0, 1, queue_remains_nonempty=True)

        estimator.reset()

        assert estimator.predict_ms(2) == pytest.approx(100.0)


class TestQueueObservation:
    def test_non_candidate_requests_contribute_to_queue_position(self):
        policy, host, _ = make_policy(initial_admission_interval_ms=10.0)
        fill_queue(policy, 50)
        keys = to_keys(1)

        admit(policy, host, "marked", keys, queue_ahead=0)

        assert policy._bundles["marked"].lead_time_ms == pytest.approx(500.0)

    def test_actual_lead_time_is_observed_at_first_schedule(self):
        policy, host, clock = make_policy()
        keys = to_keys(1)
        admit(policy, host, "r0", keys)
        clock.advance_ms(25.0)

        policy.step(ScheduleEndContext(new_req_ids=("r0",), preempted_req_ids=()))

        observations = policy._stats._values[AdmissionPrefetchMetrics.ACTUAL_LEAD_TIME][
            ()
        ]
        assert observations == pytest.approx([0.025])


class TestDeadlineGates:
    def test_resolution_and_expiry_same_step_is_late(self):
        policy, host, clock = make_policy(initial_admission_interval_ms=100.0)
        keys = to_keys(2)
        admit(policy, host, "r0", keys, queue_ahead=1)  # H = 100ms
        resolve_all(host, keys, LookupResult.HIT)
        clock.advance_ms(150)
        policy.step(empty_step())

        # Deadline is evaluated before the gate, so an expired bundle is
        # LATE even though its residency resolved in the same step.
        assert host.submits == []
        assert_partition(policy)

    def test_first_schedule_marks_unsubmitted_bundle_late(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        admit(policy, host, "r0", keys)
        policy.step(ScheduleEndContext(new_req_ids=("r0",), preempted_req_ids=()))

        assert host.submits == []
        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_gate_rejects_when_transfer_exceeds_lead_time(self):
        policy, host, _ = make_policy(initial_admission_interval_ms=1.0)
        host.transfer_base_ms = 1000.0
        keys = to_keys(2)
        # H = 1ms of lead time cannot hide a 1000ms transfer.
        admit(policy, host, "r0", keys, queue_ahead=1)
        resolve_all(host, keys, LookupResult.HIT)
        policy.step(empty_step())

        assert host.submits == []
        assert counter(policy, AdmissionPrefetchMetrics.GATE_REJECT) == 2
        assert_partition(policy)

    def test_rising_measured_cost_tightens_the_gate(self):
        """A busy tier reports slower transfers, so the gate closes itself."""
        policy, host, _ = make_policy(initial_admission_interval_ms=10.0)
        keys_fast = to_keys(2, prefix="fast")
        resolve_all(host, keys_fast, LookupResult.HIT)
        admit(policy, host, "fast", keys_fast, queue_ahead=1)
        policy.step(empty_step())
        assert host.submits == [keys_fast]

        # Same lead time, but the tier is now measurably slower.
        host.transfer_per_chunk_ms = 500.0
        keys_slow = to_keys(2, prefix="slow")
        resolve_all(host, keys_slow, LookupResult.HIT)
        admit(policy, host, "slow", keys_slow, queue_ahead=1)
        policy.step(empty_step())

        assert host.submits == [keys_fast]
        assert counter(policy, AdmissionPrefetchMetrics.GATE_REJECT) == 2
        assert_partition(policy)


class TestShadowMode:
    def test_shadow_never_calls_submit(self):
        policy, host, _ = make_policy(shadow_mode=True)
        keys = to_keys(3)
        admit(policy, host, "r0", keys)
        resolve_all(host, keys, LookupResult.HIT)
        policy.step(empty_step())

        assert host.submits == []
        assert counter(policy, AdmissionPrefetchMetrics.SHADOW_SUBMIT) == 3
        assert counter(policy, AdmissionPrefetchMetrics.SUBMITTED) == 0
        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_shadow_is_default(self):
        assert PrefetchConfig().shadow_mode is True

    def test_shadow_drains_on_the_same_schedule_as_live(self):
        # Shadow is the predictor for the live cell, so it has to spend the
        # step budget the way live does. Disposing of a whole run in one step
        # made shadow optimistic: it never re-gated the slices live would
        # still be submitting several steps later.
        kwargs = dict(max_promotions_per_step=2, max_bundle_chunks=8)
        keys = to_keys(5)

        live, live_host, _ = make_policy(shadow_mode=False, **kwargs)
        resolve_all(live_host, keys, LookupResult.HIT)
        admit(live, live_host, "r0", keys)

        shadow, shadow_host, _ = make_policy(shadow_mode=True, **kwargs)
        resolve_all(shadow_host, keys, LookupResult.HIT)
        admit(shadow, shadow_host, "r0", keys)

        # After one step both have disposed of exactly one slice.
        live.step(empty_step())
        shadow.step(empty_step())
        assert counter(live, AdmissionPrefetchMetrics.SUBMITTED) == 2
        assert counter(shadow, AdmissionPrefetchMetrics.SHADOW_SUBMIT) == 2
        assert shadow_host.submits == []
        assert shadow.has_pending_work()

        # And both take the same number of steps to finish.
        for _ in range(2):
            live.step(empty_step())
            shadow.step(empty_step())
        assert counter(live, AdmissionPrefetchMetrics.SUBMITTED) == 5
        assert counter(shadow, AdmissionPrefetchMetrics.SHADOW_SUBMIT) == 5
        # Terminal bundles are reaped, and the outcome is emitted once for the
        # whole bundle rather than once per slice.
        assert not shadow.has_pending_work()
        assert (
            counter(
                shadow,
                AdmissionPrefetchMetrics.BUNDLE_OUTCOMES,
                TIER_LABEL + ("shadow_submit",),
            )
            == 1
        )
        assert_partition(shadow)


class TestCancellation:
    def test_preemption_cancels_unsubmitted_bundle(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        admit(policy, host, "r0", keys)
        policy.step(ScheduleEndContext(new_req_ids=(), preempted_req_ids=("r0",)))

        assert host.submits == []
        assert counter(policy, AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED) == 3
        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_finish_cancels_unsubmitted_bundle(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        resolve_all(host, keys[:1], LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        policy.on_request_finished("r0")

        assert not policy.has_pending_work()
        assert counter(policy, AdmissionPrefetchMetrics.CANCELLED) == 1
        assert_partition(policy)

    def test_preemption_leaves_submitted_bundle_alone(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        assert policy._bundles["r0"].state is BundleState.SUBMITTED

        policy.step(ScheduleEndContext(new_req_ids=(), preempted_req_ids=("r0",)))
        assert policy._bundles["r0"].state is BundleState.SUBMITTED

        policy.on_promotion_finished(keys, success=True)
        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_finish_while_submitted_completes_cleanly(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        policy.on_request_finished("r0")

        # The promotion is in flight and must be allowed to complete.
        assert policy.has_pending_work()
        policy.on_promotion_finished(keys, success=True)
        assert not policy.has_pending_work()
        assert_partition(policy)


class TestCompletion:
    def test_successful_promotion_marks_ready(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        policy.on_promotion_finished(keys, success=True)

        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_failed_promotion_marks_failed(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        policy.on_promotion_finished(keys, success=False)

        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_partial_completion_keeps_bundle_pending(self):
        policy, host, _ = make_policy()
        keys = to_keys(3)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())
        policy.on_promotion_finished(keys[:1], success=True)

        assert policy.has_pending_work()
        policy.on_promotion_finished(keys[1:], success=True)
        assert not policy.has_pending_work()

    def test_unknown_completion_keys_are_ignored(self):
        policy, host, _ = make_policy()
        policy.on_promotion_finished(to_keys(2, prefix="ghost"), success=True)
        assert not policy.has_pending_work()


class TestCapacityAndBudget:
    def test_max_pending_bundles_skips_new_bundles(self):
        policy, host, _ = make_policy(max_pending_bundles=1)
        admit(policy, host, "r0", to_keys(2, prefix="a"))
        before = len(host.secondary_lookups)
        admit(policy, host, "r1", to_keys(2, prefix="b"))

        assert len(host.secondary_lookups) == before
        assert len(policy._active) == 1

    def test_bundle_larger_than_step_budget_submits_across_steps(self):
        # The step budget is temporary, so a long resolved run is carried
        # rather than discarded. Discarding it threw away 94% of the
        # verified-resident prefix in the first benchmark.
        policy, host, _ = make_policy(max_promotions_per_step=2, max_bundle_chunks=8)
        keys = to_keys(5)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)

        policy.step(empty_step())
        assert host.submits == [keys[0:2]]
        assert policy._bundles["r0"].state is BundleState.SUBMITTING
        assert counter(policy, AdmissionPrefetchMetrics.BUNDLE_TRIM) == 0

        policy.step(empty_step())
        policy.step(empty_step())
        assert host.submits == [keys[0:2], keys[2:4], keys[4:5]]
        assert counter(policy, AdmissionPrefetchMetrics.SUBMITTED) == 5
        assert_partition(policy)

    def test_candidates_beyond_the_bundle_ceiling_are_counted_as_trim(self):
        # max_bundle_chunks is the one limit no later step relaxes, and it
        # bounds the probe as well as the bundle. The keys past it are never
        # looked up, so they must be counted at admission or they vanish from
        # the partition and nothing reports that the ceiling was binding.
        policy, host, _ = make_policy(max_bundle_chunks=2)
        keys = to_keys(5)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        assert host.submits == [keys[:2]]
        assert counter(policy, AdmissionPrefetchMetrics.SUBMITTED) == 2
        assert counter(policy, AdmissionPrefetchMetrics.BUNDLE_TRIM) == 3
        # Every candidate is accounted for, not just the probed prefix.
        assert counter(policy, AdmissionPrefetchMetrics.CONSIDERED) == 5
        assert_partition(policy)

    def test_budget_exhaustion_defers_second_bundle_to_next_step(self):
        policy, host, _ = make_policy(max_promotions_per_step=2)
        keys_a = to_keys(2, prefix="a")
        keys_b = to_keys(2, prefix="b")
        resolve_all(host, keys_a + keys_b, LookupResult.HIT)
        admit(policy, host, "r0", keys_a)
        admit(policy, host, "r1", keys_b)
        policy.step(empty_step())

        # Step contention is temporary, so the second bundle keeps its chance
        # instead of losing it to whichever request resolved first.
        assert host.submits == [keys_a]
        assert policy._bundles["r1"].state is BundleState.RESIDENT
        assert counter(policy, AdmissionPrefetchMetrics.ALLOC_REFUSED) == 0

        policy.step(empty_step())
        assert host.submits == [keys_a, keys_b]
        assert_partition(policy)

    def test_probe_window_bounded_by_bundle_ceiling(self):
        # Probing keys that could never be submitted only lengthens the
        # tier's lookup batch and pushes results past bundle deadlines.
        policy, host, _ = make_policy(max_candidate_chunks=1024, max_bundle_chunks=16)
        admit(policy, host, "r0", to_keys(500))

        assert len(host.secondary_lookups) == 16

    def test_earliest_deadline_wins_the_step_budget(self):
        policy, host, _ = make_policy(
            max_promotions_per_step=2, initial_admission_interval_ms=100.0
        )
        slack = to_keys(2, prefix="slack")
        urgent = to_keys(2, prefix="urgent")

        # Admitted first, behind a deep queue, so it has the most lead time.
        admit(policy, host, "slack", slack, queue_ahead=20)
        # Drain that queue so the next admission sees a shallow one and is
        # therefore closer to its own demand.
        policy.step(
            ScheduleEndContext(
                new_req_ids=tuple(f"slack-ahead{i}" for i in range(20)),
                preempted_req_ids=(),
            )
        )
        admit(policy, host, "urgent", urgent, queue_ahead=0)
        assert (
            policy._bundles["urgent"].lead_time_ms
            < policy._bundles["slack"].lead_time_ms
        )

        resolve_all(host, slack + urgent, LookupResult.HIT)
        policy.step(empty_step())

        # Admission order would have served slack first. The bundle closest to
        # its demand wins the budget; the one that can still be hidden defers.
        assert host.submits == [urgent]
        policy.step(empty_step())
        assert host.submits == [urgent, slack]
        assert_partition(policy)

    def test_pending_bundle_overflow_accounts_candidate_keys(self):
        policy, host, _ = make_policy(max_pending_bundles=1)
        admit(policy, host, "r0", to_keys(1, prefix="a"))
        rejected = to_keys(3, prefix="b")

        admit(policy, host, "r1", rejected)

        assert counter(policy, AdmissionPrefetchMetrics.BUNDLE_OVERFLOW) == 3
        assert_partition(policy)

    def test_submit_alloc_refusal_terminalizes_bundle(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        host.submit_capacity_skipped = set(keys)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        assert counter(policy, AdmissionPrefetchMetrics.ALLOC_REFUSED) == 2
        assert not policy.has_pending_work()
        assert_partition(policy)

    def test_submit_redundant_keys_are_counted(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        host.submit_redundant = {keys[0]}
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        assert counter(policy, AdmissionPrefetchMetrics.PRIMARY_REDUNDANT) == 1
        assert counter(policy, AdmissionPrefetchMetrics.SUBMITTED) == 1
        assert_partition(policy)


class TestBoundedWork:
    def test_step_touches_only_active_bundles(self):
        policy, host, _ = make_policy()
        for i in range(5):
            keys = to_keys(2, prefix=f"done{i}")
            resolve_all(host, keys, LookupResult.HIT)
            admit(policy, host, f"done{i}", keys)
            policy.step(empty_step())
            policy.on_promotion_finished(keys, success=True)

        assert not policy.has_pending_work()
        before = len(host.secondary_lookups)
        policy.step(empty_step())
        # Terminal bundles are dropped, so idle steps do no per-bundle work.
        assert len(host.secondary_lookups) == before

    def test_has_pending_work_tracks_active_bundles(self):
        policy, host, _ = make_policy()
        assert not policy.has_pending_work()
        keys = to_keys(2)
        admit(policy, host, "r0", keys)
        assert policy.has_pending_work()
        policy.on_request_finished("r0")
        assert not policy.has_pending_work()


class TestReset:
    def test_reset_cancels_pending_bundles(self):
        policy, host, _ = make_policy()
        admit(policy, host, "r0", to_keys(3))
        policy.reset()

        assert not policy.has_pending_work()
        assert counter(policy, AdmissionPrefetchMetrics.LOOKUP_UNRESOLVED) == 3
        assert_partition(policy)

    def test_reset_requires_drained_submissions(self):
        policy, host, _ = make_policy()
        keys = to_keys(2)
        resolve_all(host, keys, LookupResult.HIT)
        admit(policy, host, "r0", keys)
        policy.step(empty_step())

        with pytest.raises(AssertionError):
            policy.reset()

        policy.on_promotion_finished(keys, success=True)
        policy.reset()


class TestPartitionInvariant:
    @pytest.mark.parametrize("seed", range(12))
    def test_terminal_partition_invariant_randomized(self, seed):
        rng = random.Random(seed)
        policy, host, clock = make_policy(
            max_pending_bundles=4,
            max_promotions_per_step=6,
            initial_admission_interval_ms=rng.choice([0.0, 50.0, 500.0]),
        )
        live = []

        for round_idx in range(40):
            if rng.random() < 0.5:
                req_id = f"r{round_idx}"
                keys = to_keys(rng.randint(1, 5), prefix=req_id)
                for key in keys:
                    roll = rng.random()
                    if roll < 0.5:
                        host.secondary[key] = LookupResult.HIT
                    elif roll < 0.7:
                        host.secondary[key] = LookupResult.MISS
                    if rng.random() < 0.2:
                        host.primary[key] = LookupResult.HIT
                admit(policy, host, req_id, keys)
                live.append((req_id, keys))

            if rng.random() < 0.3:
                clock.advance_ms(rng.choice([0.0, 10.0, 200.0]))

            new_ids: tuple[str, ...] = ()
            preempted: tuple[str, ...] = ()
            if live and rng.random() < 0.2:
                new_ids = (rng.choice(live)[0],)
            if live and rng.random() < 0.15:
                preempted = (rng.choice(live)[0],)
            policy.step(
                ScheduleEndContext(new_req_ids=new_ids, preempted_req_ids=preempted)
            )

            if live and rng.random() < 0.3:
                req_id, keys = rng.choice(live)
                policy.on_promotion_finished(keys, success=rng.random() < 0.8)
            if live and rng.random() < 0.25:
                req_id, keys = live.pop(rng.randrange(len(live)))
                policy.on_request_finished(req_id)

            assert_partition(policy)

        # reset_cache() drains in-flight jobs before resetting the policy;
        # mirror that ordering here.
        policy.on_promotion_finished(list(policy._submitted_key_owner), success=True)
        for req_id, _ in live:
            policy.on_request_finished(req_id)
        policy.reset()
        assert not policy.has_pending_work()
        assert_partition(policy)


class TestTransferCostModel:
    def test_uses_seeds_until_enough_samples(self):
        model = TransferCostModel(
            PrefetchConfig(transfer_base_ms=1.0, transfer_per_chunk_ms=3.0)
        )
        assert model.measured() is None
        assert model.predict_ms(10) == pytest.approx(31.0)

    def test_fits_base_and_slope_from_observations(self):
        model = TransferCostModel(PrefetchConfig())
        # Ground truth: 4ms fixed + 1.5ms per chunk.
        for _ in range(15):
            for n in (1, 4, 16):
                model.observe(n, 4.0 + 1.5 * n)

        fit = model.measured()
        assert fit is not None
        base, per_chunk = fit
        assert base == pytest.approx(4.0, abs=0.2)
        assert per_chunk == pytest.approx(1.5, abs=0.05)
        assert model.predict_ms(8) == pytest.approx(16.0, abs=0.3)

    def test_uniform_batch_size_is_not_identifiable(self):
        # Slope and intercept cannot be separated without spread, so the
        # model must keep reporting seeds rather than invent a fit.
        model = TransferCostModel(PrefetchConfig())
        for _ in range(100):
            model.observe(8, 20.0)
        assert model.measured() is None

    def test_tracks_a_tier_that_gets_slower(self):
        model = TransferCostModel(PrefetchConfig())
        for _ in range(200):
            for n in (1, 8):
                model.observe(n, 1.0 * n)
        fast = model.predict_ms(8)
        assert fast == pytest.approx(8.0, abs=0.5)

        for _ in range(400):
            for n in (1, 8):
                model.observe(n, 10.0 * n)
        assert model.predict_ms(8) > fast * 3

    def test_ignores_degenerate_samples(self):
        model = TransferCostModel(PrefetchConfig())
        model.observe(0, 5.0)
        model.observe(-1, 5.0)
        model.observe(4, -1.0)
        assert model.measured() is None
