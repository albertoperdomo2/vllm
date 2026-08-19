# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager-level tests for the V2.1 admission prefetch policy.

These drive a real TieringOffloadingManager so the policy is exercised
against the actual promotion, allocation, and accounting machinery. The
stub tier resolves residency asynchronously (RETRY until the test says
otherwise), which the synchronous ExampleSecondaryTierManager cannot do.
"""

from collections.abc import Iterable
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
    TieringOffloadingMetrics,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.prefetch.base import AdmissionPrefetchMetrics
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig

_MOCK_OFFLOADING_SPEC = MagicMock()


def to_keys(int_hashes):
    return [make_offload_key(str(i).encode(), 0) for i in int_hashes]


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    region = MagicMock()
    backing = torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy()
    region.create_kv_memoryview.return_value = memoryview(backing)
    return region


class AsyncStubSecondaryTier(SecondaryTierManager):
    """Secondary tier with test-controlled lookup and completion timing."""

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(self, offloading_spec, primary_kv_view, tier_type="stub"):
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.blocks: dict[OffloadKey, bool] = {}
        # Keys whose residency verdict is known; others answer RETRY.
        self.resolved: set[OffloadKey] = set()
        self.completed_jobs: list[JobResult] = []
        self.pending_jobs: list[JobMetadata] = []
        self.autocomplete = True
        self.fail_loads = False
        self.lookup_calls: list[OffloadKey] = []

    def resolve(self, keys, present=True):
        for key in keys:
            self.resolved.add(key)
            if present:
                self.blocks[key] = True

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        self.lookup_calls.append(key)
        if key not in self.resolved:
            return LookupResult.RETRY
        return LookupResult.HIT if key in self.blocks else LookupResult.MISS

    def submit_store(self, job_metadata: JobMetadata) -> None:
        for key in job_metadata.keys:
            self.blocks[key] = True
        self.completed_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))

    def submit_load(self, job_metadata: JobMetadata) -> None:
        if self.autocomplete:
            self.completed_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=not self.fail_loads)
            )
        else:
            self.pending_jobs.append(job_metadata)

    def complete_pending(self, success=True):
        for job in self.pending_jobs:
            self.completed_jobs.append(JobResult(job_id=job.job_id, success=success))
        self.pending_jobs = []

    def get_finished_jobs(self) -> Iterable[JobResult]:
        result = self.completed_jobs
        self.completed_jobs = []
        return result

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def drain_jobs(self) -> None:
        self.complete_pending()

    def has_pending_work(self) -> bool:
        return bool(self.pending_jobs)


class Harness:
    """A real tiering manager plus the stub tier, driven step by step."""

    def __init__(self, num_blocks=8, **prefetch_kwargs):
        prefetch_kwargs.setdefault("enabled", True)
        prefetch_kwargs.setdefault("shadow_mode", False)
        prefetch_kwargs.setdefault("initial_admission_interval_ms", 10_000.0)
        region = _mock_mmap_region(num_blocks)
        self.primary = CPUPrimaryTierOffloadingManager(
            num_blocks=num_blocks, mmap_region=region
        )
        self.tier = AsyncStubSecondaryTier(
            _MOCK_OFFLOADING_SPEC, region.create_kv_memoryview()
        )
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary,
            secondary_tiers=[self.tier],
            prefetch_config=PrefetchConfig(**prefetch_kwargs),
        )
        self.policy = self.manager._prefetch_policy

    def admit(self, req_id, keys, queue_ahead=3):
        for i in range(queue_ahead):
            filler = ReqContext(req_id=f"{req_id}-ahead{i}", kv_transfer_params=None)
            self.manager.on_new_request(filler)
            self.manager.prefetch_on_admission([], filler)
        ctx = ReqContext(req_id=req_id, kv_transfer_params=None)
        self.manager.on_new_request(ctx)
        self.manager.prefetch_on_admission(keys, ctx)
        return ctx

    def step(self, new_req_ids=(), preempted_req_ids=()):
        self.manager.on_schedule_end(
            ScheduleEndContext(
                new_req_ids=new_req_ids, preempted_req_ids=preempted_req_ids
            )
        )

    def counter(self, name, labelvalues=("1:stub",)):
        stats = self.policy._stats._values.get(name, {})
        return stats.get(labelvalues, 0)

    def manager_counter(self, name, labelvalues=("1:stub",)):
        return self.manager._stats._values.get(name, {}).get(labelvalues, 0)

    def assert_partition(self):
        considered = self.counter(AdmissionPrefetchMetrics.CONSIDERED)
        terminal = sum(
            self.counter(name) for name in AdmissionPrefetchMetrics.TERMINAL_COUNTERS
        )
        assert considered == terminal


class TestSubmission:
    def test_resident_bundle_is_submitted_and_promoted(self):
        h = Harness()
        keys = to_keys([1, 2, 3])
        h.tier.resolve(keys)
        h.admit("r0", keys)
        h.step()

        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 3
        # V1 counters stay populated so bench cells remain comparable.
        assert h.manager_counter(TieringOffloadingMetrics.PREFETCH_ATTEMPTED) == 3
        assert h.manager_counter(TieringOffloadingMetrics.PREFETCH_PROMOTED) == 3
        assert len(h.manager._prefetched) == 3
        h.assert_partition()

    def test_submission_allocates_each_key_exactly_once(self, monkeypatch):
        h = Harness()
        keys = to_keys([1, 2, 3])
        h.tier.resolve(keys)
        calls = []
        original = h.primary.prepare_store

        def spy(store_keys, req_context, **kwargs):
            calls.append(list(store_keys))
            return original(store_keys, req_context, **kwargs)

        monkeypatch.setattr(h.primary, "prepare_store", spy)
        monkeypatch.setattr(h.primary, "prepare_write", spy)

        h.admit("r0", keys)
        h.step()

        # One allocation per key: no reserve-then-consume double spend.
        allocated = [k for call in calls for k in call]
        assert sorted(allocated) == sorted(keys)

    def test_absent_keys_are_never_submitted(self):
        h = Harness()
        keys = to_keys([1, 2, 3])
        h.tier.resolve(keys[:1], present=True)
        h.tier.resolve(keys[1:], present=False)
        h.admit("r0", keys)
        h.step()

        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 1
        assert h.counter(AdmissionPrefetchMetrics.SECONDARY_ABSENT) == 1
        h.assert_partition()

    def test_pending_lookup_defers_submission_across_steps(self):
        h = Harness()
        keys = to_keys([1, 2])
        h.admit("r0", keys)
        h.step()
        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 0
        assert h.manager.has_pending_work()

        h.tier.resolve(keys)
        h.step()
        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 2
        h.assert_partition()


class TestNonEviction:
    def test_speculative_submission_never_evicts_resident_blocks(self):
        h = Harness(num_blocks=2)
        resident = to_keys([90, 91])
        ctx = ReqContext(req_id="demand", kv_transfer_params=None)
        assert h.primary.prepare_store(resident, ctx) is not None
        h.primary.complete_store(resident, ctx)

        keys = to_keys([1, 2])
        h.tier.resolve(keys)
        h.admit("r0", keys)
        h.step()

        # The tier is full of evictable blocks; the demand path would evict
        # them, but speculative work must not.
        for key in resident:
            assert h.primary.lookup(key, ctx) is LookupResult.HIT
        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 0
        assert h.counter(AdmissionPrefetchMetrics.CAPACITY_SKIP) == 2
        h.assert_partition()

    def test_partial_capacity_submits_contiguous_prefix(self):
        h = Harness(num_blocks=3)
        resident = to_keys([90, 91])
        ctx = ReqContext(req_id="demand", kv_transfer_params=None)
        h.primary.prepare_store(resident, ctx)
        h.primary.complete_store(resident, ctx)

        keys = to_keys([1, 2, 3])
        h.tier.resolve(keys)
        h.admit("r0", keys)
        h.step()

        # Only one free block remains, so the contiguous prefix of length 1
        # is submitted and the rest is capacity-skipped.
        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 1
        assert h.counter(AdmissionPrefetchMetrics.CAPACITY_SKIP) == 2
        h.assert_partition()


class TestShadowMode:
    def test_shadow_mode_moves_no_data(self):
        h = Harness(shadow_mode=True)
        keys = to_keys([1, 2, 3])
        h.tier.resolve(keys)
        h.admit("r0", keys)
        h.step()

        assert h.counter(AdmissionPrefetchMetrics.SHADOW_SUBMIT) == 3
        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 0
        # No allocation, no promotion, no V1 counter movement.
        assert h.primary._get_num_free_blocks() == 8
        assert h.manager_counter(TieringOffloadingMetrics.PREFETCH_PROMOTED) == 0
        assert not h.manager._prefetched
        assert not h.manager.has_pending_work()
        h.assert_partition()


class TestCompletionAndFailure:
    def test_successful_promotion_resolves_bundle(self):
        h = Harness()
        keys = to_keys([1, 2])
        h.tier.resolve(keys)
        h.admit("r0", keys)
        h.step()
        h.step()

        assert not h.policy.has_pending_work()
        h.assert_partition()

    def test_failed_promotion_is_load_failed_and_frees_blocks(self):
        h = Harness()
        keys = to_keys([1, 2])
        h.tier.resolve(keys)
        h.tier.fail_loads = True
        h.admit("r0", keys)
        free_before = h.primary._get_num_free_blocks()
        h.step()
        h.step()

        assert h.manager_counter(TieringOffloadingMetrics.PREFETCH_LOAD_FAILED) == 2
        assert not h.manager._prefetched
        assert h.primary._get_num_free_blocks() == free_before
        assert not h.policy.has_pending_work()
        h.assert_partition()

    def test_useful_outcome_recorded_on_demand_hit(self):
        h = Harness()
        keys = to_keys([1, 2])
        h.tier.resolve(keys)
        ctx = h.admit("r0", keys)
        h.step()
        h.step()

        for key in keys:
            h.manager.lookup(key, ctx)

        assert h.manager_counter(TieringOffloadingMetrics.PREFETCH_USEFUL) == 2


class TestCancellation:
    def test_request_finish_cancels_pending_bundle(self):
        h = Harness()
        keys = to_keys([1, 2])
        ctx = h.admit("r0", keys)
        h.step()
        lookups_before = len(h.tier.lookup_calls)

        h.manager.on_request_finished(ctx)
        h.step()
        h.step()

        # No lookups may be re-issued after finish: doing so would repopulate
        # the tier's async lookup state after its cleanup ran.
        assert len(h.tier.lookup_calls) == lookups_before
        assert not h.policy.has_pending_work()
        h.assert_partition()

    def test_preemption_cancels_pending_bundle(self):
        h = Harness()
        keys = to_keys([1, 2])
        h.admit("r0", keys)
        h.step(preempted_req_ids=("r0",))

        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 0
        assert not h.policy.has_pending_work()
        h.assert_partition()

    def test_finish_while_submitted_completes_cleanly(self):
        h = Harness()
        keys = to_keys([1, 2])
        h.tier.resolve(keys)
        h.tier.autocomplete = False
        ctx = h.admit("r0", keys)
        h.step()
        assert h.counter(AdmissionPrefetchMetrics.SUBMITTED) == 2

        h.manager.on_request_finished(ctx)
        h.tier.complete_pending(success=True)
        h.step()

        assert not h.policy.has_pending_work()
        h.assert_partition()


class TestLiveness:
    def test_has_pending_work_covers_policy_state(self):
        h = Harness()
        assert not h.manager.has_pending_work()

        h.admit("r0", to_keys([1, 2]))
        # A bundle awaiting residency needs more steps, so the engine must
        # keep stepping even with nothing scheduled.
        assert h.manager.has_pending_work()

        h.tier.resolve(to_keys([1, 2]))
        h.step()
        h.step()
        assert not h.manager.has_pending_work()


class TestResetCache:
    def test_reset_cache_with_pending_and_submitted_bundles(self):
        h = Harness()
        submitted = to_keys([1, 2])
        h.tier.resolve(submitted)
        h.tier.autocomplete = False
        h.admit("r0", submitted)
        h.step()
        h.admit("r1", to_keys([5, 6]))

        h.manager.reset_cache()

        assert not h.manager._prefetch_job_keys
        assert not h.policy.has_pending_work()
        h.assert_partition()


class TestV1Coexistence:
    def test_v1_and_v2_are_mutually_exclusive(self):
        region = _mock_mmap_region(4)
        primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=region)
        tier = AsyncStubSecondaryTier(
            _MOCK_OFFLOADING_SPEC, region.create_kv_memoryview()
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            TieringOffloadingManager(
                primary_tier=primary,
                secondary_tiers=[tier],
                admission_prefetch_chunks=100,
                prefetch_config=PrefetchConfig(enabled=True),
            )

    def test_prefetch_requires_secondary_tier(self):
        region = _mock_mmap_region(4)
        primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=region)
        with pytest.raises(ValueError, match="at least one secondary tier"):
            TieringOffloadingManager(
                primary_tier=primary,
                secondary_tiers=[],
                prefetch_config=PrefetchConfig(enabled=True),
            )

    def test_tier_idx_out_of_range_is_rejected(self):
        region = _mock_mmap_region(4)
        primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=region)
        tier = AsyncStubSecondaryTier(
            _MOCK_OFFLOADING_SPEC, region.create_kv_memoryview()
        )
        with pytest.raises(ValueError, match="out of range"):
            TieringOffloadingManager(
                primary_tier=primary,
                secondary_tiers=[tier],
                prefetch_config=PrefetchConfig(enabled=True, tier_idx=3),
            )

    def test_disabled_config_leaves_manager_inert(self):
        h_region = _mock_mmap_region(4)
        primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=h_region)
        tier = AsyncStubSecondaryTier(
            _MOCK_OFFLOADING_SPEC, h_region.create_kv_memoryview()
        )
        manager = TieringOffloadingManager(
            primary_tier=primary,
            secondary_tiers=[tier],
            prefetch_config=PrefetchConfig(enabled=False),
        )
        assert manager._prefetch_policy is None
        assert not manager.prefetch_policy_enabled
