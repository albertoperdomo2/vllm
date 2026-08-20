# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TieringOffloadingManager and ExampleSecondaryTierManager.

These tests verify:
1. Basic tiered offloading operations (store, load, lookup)
2. Cascade behavior (blocks stored to all secondary tiers)
3. Promotion behavior (blocks loaded from secondary to primary to GPU)
4. ref_cnt management (blocks protected during async transfers)
5. Eviction coordination between tiers
"""

from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    _parse_tier_filter,
)
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingHistogramMetadata,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
    TieringOffloadingMetrics,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.prefetch.base import AdmissionPrefetchMetrics
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

_CTX = ReqContext(req_id="test")
_MOCK_OFFLOADING_SPEC = MagicMock()


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    """Create a mock SharedOffloadRegion for testing."""
    mock = MagicMock()
    view = memoryview(torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy())
    mock.create_kv_memoryview.return_value = view
    return mock


def _make_tiering_spec(
    extra_config: dict[str, object] | None = None,
) -> TieringOffloadingSpec:
    config = OffloadingConfig(
        groups=(OffloadingGroupConfig(16, ("layer",)),),
        worker_kv_bytes_per_block=8,
        enable_kv_cache_events=False,
        extra_config={
            "cpu_bytes_to_use": 65536,
            **(extra_config or {}),
        },
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )
    return TieringOffloadingSpec(config)


def to_keys(int_ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def counter_total(stats: OffloadingConnectorStats, name: str) -> float:
    """Sum a counter across all its tier label values."""
    return sum(stats.data["data"].get(name, {}).values())


def assert_prefetch_accounting(stats: OffloadingConnectorStats, manager) -> None:
    """Every prefetch promotion is resolved, failed, or still tracked.

    This must hold unconditionally — including when tracking is disabled —
    otherwise promotions vanish from the accounting and the useful/wasted
    ratio silently loses its denominator.
    """
    promoted = counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED)
    accounted = (
        counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL)
        + counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED)
        + counter_total(stats, TieringOffloadingMetrics.PREFETCH_UNTRACKED)
        + counter_total(stats, TieringOffloadingMetrics.PREFETCH_LOAD_FAILED)
        + len(manager._prefetched)
        + len(manager._pending_untracked_prefetches)
    )
    assert accounted == promoted


def count_hits(manager, keys: list[OffloadKey]) -> int | None:
    """Count consecutive lookup hits from the start of keys.

    Returns the count of leading HIT results, or None if any lookup
    returns HIT_PENDING or RETRY.
    """
    count = 0
    for key in keys:
        result = manager.lookup(key, _CTX)
        if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
            return None
        if result is not LookupResult.HIT:
            break
        count += 1
    return count


class MetricsSecondaryTierManager(SecondaryTierManager):
    """Test-only secondary tier that declares and emits one labeled metric."""

    MY_TIER_METRIC = "my_tier_metric"

    @classmethod
    def build_metric_definitions(cls, extra_config):
        return {
            cls.MY_TIER_METRIC: OffloadingCounterMetadata(
                documentation="Number of bytes served by the test tier.",
                labelnames=("tier",),
            )
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats: OffloadingConnectorStats | None = None

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return False

    def submit_store(self, job_metadata: JobMetadata) -> None:
        return

    def submit_load(self, job_metadata: JobMetadata) -> None:
        return

    def get_finished_jobs(self) -> Iterable[JobResult]:
        return ()

    def drain_jobs(self) -> None:
        return

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self.stats
        self.stats = None
        return stats


def test_tiering_spec_collects_secondary_metric_definitions(monkeypatch):
    monkeypatch.setitem(
        SecondaryTierFactory._registry,
        "test_metrics",
        lambda: MetricsSecondaryTierManager,
    )

    metrics = TieringOffloadingSpec.build_metric_definitions(
        {"secondary_tiers": [{"type": "test_metrics"}]}
    )

    metadata = metrics[MetricsSecondaryTierManager.MY_TIER_METRIC]
    assert metadata.documentation == "Number of bytes served by the test tier."
    assert metadata.labelnames == ("tier",)


def test_tiering_spec_defines_prefetch_outcome_metrics():
    """The outcome counters must reach Prometheus with a single tier label.

    OffloadPromMetrics builds its counters straight from these definitions, and
    _get_prometheus_metric asserts the emitted label arity matches labelnames.
    """
    metrics = TieringOffloadingSpec.build_metric_definitions({})

    for name in (
        TieringOffloadingMetrics.PREFETCH_USEFUL,
        TieringOffloadingMetrics.PREFETCH_WASTED,
        TieringOffloadingMetrics.PREFETCH_REDUNDANT,
        TieringOffloadingMetrics.PREFETCH_UNTRACKED,
        TieringOffloadingMetrics.PREFETCH_LOAD_FAILED,
        TieringOffloadingMetrics.PREFETCH_LATE,
    ):
        metadata = metrics[name]
        assert isinstance(metadata, OffloadingCounterMetadata)
        assert metadata.labelnames == ("tier",)


def test_tiering_spec_defaults_admission_prefetch_chunks_to_zero():
    spec = _make_tiering_spec()

    assert spec.admission_prefetch_chunks == 0


def test_tiering_manager_accepts_positive_admission_prefetch_chunks():
    mock_region = _mock_mmap_region(5)
    primary_tier = CPUPrimaryTierOffloadingManager(
        num_blocks=5, mmap_region=mock_region
    )
    secondary_tier = ExampleSecondaryTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_region.create_kv_memoryview(),
        tier_type="example",
    )

    manager = TieringOffloadingManager(
        primary_tier=primary_tier,
        secondary_tiers=[secondary_tier],
        admission_prefetch_chunks=100,
    )

    assert manager.admission_prefetch_chunks == 100


@pytest.mark.parametrize("value", [-1, -100])
def test_tiering_spec_rejects_negative_admission_prefetch_chunks(value):
    with pytest.raises(
        ValueError,
        match="admission_prefetch_chunks must be a non-negative int",
    ):
        _make_tiering_spec({"admission_prefetch_chunks": value})


@pytest.mark.parametrize("value", [False, True])
def test_tiering_spec_rejects_boolean_admission_prefetch_chunks(value):
    with pytest.raises(
        ValueError,
        match="admission_prefetch_chunks must be a non-negative int",
    ):
        _make_tiering_spec({"admission_prefetch_chunks": value})


def test_enabled_admission_prefetch_requires_secondary_tier():
    primary_tier = CPUPrimaryTierOffloadingManager(
        num_blocks=5, mmap_region=_mock_mmap_region(5)
    )

    with pytest.raises(
        ValueError,
        match="admission_prefetch_chunks requires at least one secondary tier",
    ):
        TieringOffloadingManager(
            primary_tier=primary_tier,
            admission_prefetch_chunks=1,
        )


def test_tiering_spec_defines_admission_prefetch_metrics():
    """V2 policy counters need declarable metadata with exact label arity.

    build_metric_definitions runs before any manager exists, so the
    definitions must come from the raw extra_config alone.
    """
    metrics = TieringOffloadingSpec.build_metric_definitions({})

    for name in (
        AdmissionPrefetchMetrics.CONSIDERED,
        *AdmissionPrefetchMetrics.TERMINAL_COUNTERS,
    ):
        metadata = metrics[name]
        assert isinstance(metadata, OffloadingCounterMetadata)
        assert metadata.labelnames == ("tier",)

    assert metrics[AdmissionPrefetchMetrics.BUNDLE_OUTCOMES].labelnames == (
        "tier",
        "outcome",
    )
    assert metrics[AdmissionPrefetchMetrics.TRANSITIONS].labelnames == ("transition",)
    for name in (
        AdmissionPrefetchMetrics.LEAD_TIME,
        AdmissionPrefetchMetrics.ACTUAL_LEAD_TIME,
    ):
        assert isinstance(metrics[name], OffloadingHistogramMetadata)


def test_tiering_spec_defaults_prefetch_config_to_none():
    assert _make_tiering_spec().prefetch_config is None


def test_tiering_spec_prefetch_defaults_to_shadow_mode():
    spec = _make_tiering_spec(
        {
            "prefetch": {"enabled": True},
            "secondary_tiers": [{"type": "example"}],
        }
    )

    # Live submission must be an explicit opt-in until V2.0 calibration.
    assert spec.prefetch_config.shadow_mode is True
    assert spec.prefetch_config.policy == "admission"


def test_tiering_spec_rejects_prefetch_with_v1_chunks():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_tiering_spec(
            {
                "prefetch": {"enabled": True},
                "admission_prefetch_chunks": 100,
                "secondary_tiers": [{"type": "example"}],
            }
        )


def test_tiering_spec_rejects_prefetch_without_secondary_tier():
    with pytest.raises(ValueError, match="at least one secondary tier"):
        _make_tiering_spec({"prefetch": {"enabled": True}})


def test_tiering_spec_rejects_prefetch_tier_idx_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _make_tiering_spec(
            {
                "prefetch": {"enabled": True, "tier_idx": 4},
                "secondary_tiers": [{"type": "example"}],
            }
        )


def test_tiering_spec_rejects_unknown_prefetch_policy():
    with pytest.raises(ValueError, match="Unknown prefetch policy"):
        _make_tiering_spec(
            {
                "prefetch": {"enabled": True, "policy": "nope"},
                "secondary_tiers": [{"type": "example"}],
            }
        )


def test_tiering_spec_rejects_unknown_prefetch_keys():
    with pytest.raises(ValueError, match="Unknown prefetch config keys"):
        _make_tiering_spec({"prefetch": {"enabled": True, "shadow": True}})


@pytest.mark.parametrize(
    "prefetch_config,message",
    [
        ({"enabled": 1}, "must be a boolean"),
        ({"shadow_mode": "yes"}, "must be a boolean"),
        ({"max_pending_bundles": 0}, "must be >= 1"),
        ({"max_pending_bundles": True}, "must be an integer"),
        ({"max_promotions_per_step": -1}, "must be >= 1"),
        ({"speculative_reserve_blocks": -1}, "must be >= 0"),
        ({"max_bundle_chunks": 0}, "must be >= 1"),
        ({"speculative_max_bytes": 0}, "Unknown prefetch config keys"),
        ({"transfer_base_ms": -1.0}, "must be finite and >= 0"),
        ({"admission_interval_ewma_alpha": 0.0}, "must be in \\(0, 1\\]"),
        # Retired cost-model knobs are rejected rather than silently ignored.
        ({"p_use": 0.9}, "Unknown prefetch config keys"),
        ({"demand_load_per_chunk_ms": 2.0}, "Unknown prefetch config keys"),
        ({"policy": ""}, "must be a non-empty string"),
    ],
)
def test_tiering_spec_rejects_invalid_prefetch_values(prefetch_config, message):
    with pytest.raises(ValueError, match=message):
        _make_tiering_spec({"prefetch": prefetch_config})


def test_tiering_manager_aggregates_secondary_stats():
    mock_region = _mock_mmap_region(5)
    primary_tier = CPUPrimaryTierOffloadingManager(
        num_blocks=5, mmap_region=mock_region
    )
    secondary_tier = MetricsSecondaryTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_region.create_kv_memoryview(),
        tier_type="test_metrics",
    )
    secondary_stats = OffloadingConnectorStats()
    secondary_stats.increase_counter(
        MetricsSecondaryTierManager.MY_TIER_METRIC, 7, ("test_metrics",)
    )
    secondary_tier.stats = secondary_stats
    manager = TieringOffloadingManager(
        primary_tier=primary_tier,
        secondary_tiers=[secondary_tier],
    )

    stats = manager.get_stats()

    assert stats is not None
    assert (
        stats.data["data"][MetricsSecondaryTierManager.MY_TIER_METRIC][
            ("test_metrics",)
        ]
        == 7
    )

    # The primary tier's cache-usage gauge is always reported, so get_stats()
    # never returns None, but the secondary tier has nothing new to report
    # once its stats have been consumed.
    second_stats = manager.get_stats()
    assert second_stats is not None
    assert MetricsSecondaryTierManager.MY_TIER_METRIC not in second_stats.data["data"]


class TestExampleSecondaryTierManager:
    """Tests for ExampleSecondaryTierManager implementation."""

    def test_basic_store_and_lookup(self):
        """Test basic store and lookup operations."""
        mock_view = memoryview(torch.zeros((10, 16), dtype=torch.int8).numpy())
        tier = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
            custom_param=67,
        )

        # Initially empty
        blocks = to_keys(range(3))
        assert tier.lookup(blocks[0], _CTX) is LookupResult.MISS

        # Store blocks (simulate with direct insertion for testing)
        tier.blocks[blocks[0]] = True
        tier.blocks[blocks[1]] = True

        # Lookup should find first two blocks
        assert tier.lookup(blocks[0], _CTX) is LookupResult.HIT
        assert tier.lookup(blocks[1], _CTX) is LookupResult.HIT

        # Third block not present
        assert tier.lookup(blocks[2], _CTX) is LookupResult.MISS


class TestTieringOffloadingManager:
    """Tests for TieringOffloadingManager."""

    @pytest.fixture
    def manager_setup(self):
        # Create primary tier (CPU-based)
        mock_region = _mock_mmap_region(5)
        self.primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=5, mmap_region=mock_region
        )

        mock_view = mock_region.create_kv_memoryview()

        # Create secondary tiers with the primary view
        self.secondary_tier1 = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )
        self.secondary_tier2 = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )

        # Create tiered manager
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary_tier,
            secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
        )

    def _simulate_on_schedule_end(self, new_req_ids: list[str] | None = None):
        """Simulate end of scheduler step: lifecycle flush + drain events."""
        ctx = ScheduleEndContext(new_req_ids=new_req_ids or [], preempted_req_ids=())
        self.manager.on_schedule_end(ctx)
        list(self.manager.take_events())

    def _start_request(self, req_context: ReqContext = _CTX):
        if req_context.req_id not in self.manager._req_state:
            self.manager.on_new_request(req_context)

    def test_take_events_aggregates_tier_owned_events(self, manager_setup):
        primary_event = OffloadingEvent(to_keys([1]), Medium.CPU, removed=False)
        secondary_event1 = OffloadingEvent(to_keys([2]), Medium.STORAGE, removed=False)
        secondary_event2 = OffloadingEvent(to_keys([3]), Medium.STORAGE, removed=True)

        self.primary_tier.take_events = MagicMock(return_value=[primary_event])
        self.secondary_tier1.take_events = MagicMock(return_value=[secondary_event1])
        self.secondary_tier2.take_events = MagicMock(return_value=[secondary_event2])

        assert list(self.manager.take_events()) == [
            primary_event,
            secondary_event1,
            secondary_event2,
        ]
        self.primary_tier.take_events.assert_called_once_with()
        self.secondary_tier1.take_events.assert_called_once_with()
        self.secondary_tier2.take_events.assert_called_once_with()

    def test_basic_store_to_primary(self, manager_setup):
        """Test basic store operation to primary tier."""
        blocks = to_keys(range(3))

        # Prepare store
        self._start_request()
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 3

        # Complete store
        self.manager.complete_store(blocks, _CTX, success=True)

        # Blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

    def test_cascade_to_all_secondary_tiers(self, manager_setup):
        """Test that blocks are cascaded to ALL secondary tiers."""
        blocks = to_keys(range(3))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Store to primary
        self._start_request()
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store (triggers cascade via submit_store on each tier)
        self.manager.complete_store(blocks, _CTX, success=True)

        # submit_store was called once per secondary tier
        self.secondary_tier1.submit_store.assert_called_once()
        self.secondary_tier2.submit_store.assert_called_once()

        # Blocks should be in both secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3

        # Verify blocks are present
        assert all(
            self.secondary_tier1.lookup(b, _CTX) is LookupResult.HIT for b in blocks
        )
        assert all(
            self.secondary_tier2.lookup(b, _CTX) is LookupResult.HIT for b in blocks
        )

    def test_ref_cnt_protection_during_cascade(self, manager_setup):
        """Test that ref_cnt protects blocks during cascade."""
        blocks = to_keys(range(3))

        # Store to primary
        self._start_request()
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)

        # After complete_store, blocks should have ref_cnt > 0
        # (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            # ref_cnt should be 2 (one for each secondary tier)
            assert block.ref_cnt == 2

        # End of step 1: _maybe_process_finished_jobs() was already called by
        # prepare_store() above (setting the per-step flag), so on_schedule_end()
        # does NOT poll get_finished_jobs() again — cascade completions remain
        # unprocessed until the next step.
        self._simulate_on_schedule_end()

        # ref_cnt still held: cascade jobs finished (sync tier) but haven't
        # been polled yet because the per-step guard skipped the second call.
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 2

        # Secondary tiers have completed jobs waiting to be drained
        assert len(self.secondary_tier1.completed_jobs) > 0
        assert len(self.secondary_tier2.completed_jobs) > 0

        # End of step 2: flag was reset, so _maybe_process_finished_jobs()
        # runs and processes the cascade completions (complete_read → ref_cnt--)
        self._simulate_on_schedule_end()

        # After cascade completes, ref_cnt should be 0
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 0

        # All completed jobs have been drained
        assert len(self.secondary_tier1.completed_jobs) == 0
        assert len(self.secondary_tier2.completed_jobs) == 0

    def test_lookup_from_primary(self, manager_setup):
        """Test lookup when blocks are in primary tier."""
        blocks = to_keys(range(3))

        # Store blocks
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)

        # Lookup should find all blocks in primary
        assert count_hits(self.manager, blocks) == 3

    def test_promotion_from_secondary(self, manager_setup):
        """Test promotion of blocks from secondary to primary tier."""
        blocks = to_keys(range(3))

        # Manually add blocks to secondary tier (simulate previous cascade)
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        # Lookup each block to initiate promotion for all of them
        for block in blocks:
            result = self.manager.lookup(block, _CTX)
            assert result is LookupResult.RETRY  # promotion initiated

        # End of step 1: flushes deferred submit_load() calls
        self._simulate_on_schedule_end()

        # End of step 2: processes the completed promotion jobs
        self._simulate_on_schedule_end()

        # Now blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

        # Next lookup should succeed
        assert count_hits(self.manager, blocks) == 3

    def test_prefetch_assume_resident_promotes_without_secondary_lookup(
        self, manager_setup
    ):
        """Blind prefetch batches assumed-resident keys under the real tier."""
        self._start_request()
        blocks = to_keys(range(3))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True
        self.secondary_tier1.lookup = MagicMock(wraps=self.secondary_tier1.lookup)
        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        initiated = self.manager.prefetch_assume_resident(blocks, _CTX, tier_idx=0)

        assert initiated == len(blocks)
        self.secondary_tier1.lookup.assert_not_called()
        self.secondary_tier1.submit_load.assert_not_called()

        self._simulate_on_schedule_end()
        self.secondary_tier1.submit_load.assert_called_once()
        job_metadata = self.secondary_tier1.submit_load.call_args.args[0]
        assert job_metadata.keys == blocks

        self._simulate_on_schedule_end()
        assert all(
            self.primary_tier.lookup(block, _CTX) is LookupResult.HIT
            for block in blocks
        )

        stats = self.manager.get_stats()
        assert stats is not None
        tier_label = ("1:example",)
        assert stats.data["data"][TieringOffloadingMetrics.PREFETCH_ATTEMPTED] == {
            tier_label: len(blocks)
        }
        assert stats.data["data"][TieringOffloadingMetrics.PREFETCH_PROMOTED] == {
            tier_label: len(blocks)
        }

    def test_prefetch_batches_one_hundred_keys_in_one_load_job(self):
        mock_region = _mock_mmap_region(100)
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=100, mmap_region=mock_region
        )
        secondary_tier = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_region.create_kv_memoryview(),
            tier_type="example",
        )
        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[secondary_tier],
        )
        manager.on_new_request(_CTX)
        blocks = to_keys(range(100))
        secondary_tier.blocks.update(dict.fromkeys(blocks, True))
        secondary_tier.submit_load = MagicMock(wraps=secondary_tier.submit_load)

        assert manager.prefetch_assume_resident(blocks, _CTX) == 100
        manager.on_schedule_end(
            ScheduleEndContext(new_req_ids=(), preempted_req_ids=())
        )

        secondary_tier.submit_load.assert_called_once()
        job_metadata = secondary_tier.submit_load.call_args.args[0]
        assert job_metadata.keys == blocks
        assert len(job_metadata.block_ids) == 100

    def test_prefetch_primary_hit_is_redundant(self, manager_setup):
        self._start_request()
        block = to_keys([1])[0]
        write_result = self.primary_tier.prepare_write([block], _CTX)
        assert write_result is not None
        self.primary_tier.complete_write([block], _CTX, success=True)

        assert self.manager.prefetch_assume_resident([block], _CTX) == 0

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_REDUNDANT) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED) == 0

    def test_prefetch_primary_hit_pending_is_redundant(self, manager_setup):
        self._start_request()
        block = to_keys([1])[0]
        self.secondary_tier1.blocks[block] = True

        assert self.manager.prefetch_assume_resident([block], _CTX) == 1
        assert self.primary_tier.lookup(block, _CTX) is LookupResult.HIT_PENDING
        assert self.manager.prefetch_assume_resident([block], _CTX) == 0

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_REDUNDANT) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED) == 1

    def test_prefetch_primary_capacity_exhaustion_is_skipped(self, manager_setup):
        self._start_request()
        pending_blocks = to_keys(range(5))
        assert self.primary_tier.prepare_write(pending_blocks, _CTX) is not None
        candidate = to_keys([10])[0]
        self.secondary_tier1.blocks[candidate] = True

        assert self.manager.prefetch_assume_resident([candidate], _CTX) == 0

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_SKIPPED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED) == 0

    def test_prefetch_filtered_source_tier_is_skipped(self, manager_setup):
        candidate = to_keys([1])[0]
        self.secondary_tier1.blocks[candidate] = True
        self.secondary_tier1.lookup = MagicMock(wraps=self.secondary_tier1.lookup)
        ctx = ReqContext(
            req_id="filtered-prefetch",
            load_tier_filter=TierFilter(matchers=(TierMatcher(medium=Medium.STORAGE),)),
        )
        self._start_request(ctx)

        assert self.manager.prefetch_assume_resident([candidate], ctx) == 0
        self.secondary_tier1.lookup.assert_not_called()

        stats = self.manager.get_stats()
        assert stats is not None
        tier_label = ("1:example",)
        assert stats.data["data"][TieringOffloadingMetrics.PREFETCH_SKIPPED] == {
            tier_label: 1
        }

    def test_prefetch_attempt_accounting_partitions_outcomes(self, manager_setup):
        self._start_request()
        redundant, promoted, skipped = to_keys(range(3))

        write_result = self.primary_tier.prepare_write([redundant], _CTX)
        assert write_result is not None
        self.primary_tier.complete_write([redundant], _CTX, success=True)
        assert self.primary_tier.prepare_read([redundant], _CTX) is not None
        assert self.primary_tier.prepare_write(to_keys(range(10, 13)), _CTX)

        self.secondary_tier1.blocks[promoted] = True
        self.secondary_tier1.blocks[skipped] = True
        assert (
            self.manager.prefetch_assume_resident([redundant, promoted, skipped], _CTX)
            == 1
        )

        stats = self.manager.get_stats()
        assert stats is not None
        attempted = counter_total(stats, TieringOffloadingMetrics.PREFETCH_ATTEMPTED)
        partitioned = (
            counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED)
            + counter_total(stats, TieringOffloadingMetrics.PREFETCH_REDUNDANT)
            + counter_total(stats, TieringOffloadingMetrics.PREFETCH_SKIPPED)
        )
        assert attempted == 3
        assert attempted == partitioned

    def _queue_tracked_promotion(self, block: OffloadKey) -> None:
        self.secondary_tier1.blocks[block] = True
        assert self.manager.prefetch_assume_resident([block], _CTX) == 1

    def _complete_tracked_promotion(self, block: OffloadKey) -> None:
        self._queue_tracked_promotion(block)
        # Step 1 flushes the deferred submit_load(), step 2 processes the job.
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

    def test_prefetch_useful_when_demand_lookup_hits(self, manager_setup):
        """A demand lookup reaching a prefetched block counts it as useful."""
        self._start_request()
        block = to_keys([1])[0]
        self._complete_tracked_promotion(block)

        assert self.manager.lookup(block, _CTX) is LookupResult.HIT

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        # resolved keys stop being tracked, so they cannot be counted twice
        assert block not in self.manager._prefetched
        assert_prefetch_accounting(stats, self.manager)

    def test_prefetch_wasted_when_evicted_before_demand_lookup(self, manager_setup):
        """A prefetched block evicted before the demand scan counts as wasted."""
        self._start_request()
        block = to_keys([1])[0]
        self._complete_tracked_promotion(block)

        # Fill the 5-block primary tier, forcing the promoted block out.
        filler = to_keys(range(10, 15))
        result = self.manager.prepare_store(filler, _CTX)
        assert result is not None
        assert block in result.evicted_keys

        assert self.manager.lookup(block, _CTX) is LookupResult.MISS

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 1
        assert block not in self.manager._prefetched
        assert_prefetch_accounting(stats, self.manager)

    def test_prefetch_in_flight_promotion_stays_tracked(self, manager_setup):
        """The first HIT_PENDING is late once and can still become useful."""
        self._start_request()
        block = to_keys([1])[0]
        self._queue_tracked_promotion(block)

        # The promotion is still in flight: primary holds the slot at ref_cnt=-1.
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT_PENDING
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT_PENDING

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_LATE) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        assert block in self.manager._prefetched
        assert block in self.manager._late_prefetches
        assert_prefetch_accounting(stats, self.manager)

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_LATE) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 1
        assert block not in self.manager._prefetched
        assert block not in self.manager._late_prefetches

    def test_failed_prefetch_is_load_failed_not_later_wasted(self, manager_setup):
        """A failed assume-resident load has one terminal failure outcome."""
        self._start_request()
        missing = to_keys([1])[0]

        assert self.manager.prefetch_assume_resident([missing], _CTX) == 1
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        assert self.manager.lookup(missing, _CTX) is LookupResult.MISS

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_LOAD_FAILED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert missing not in self.manager._prefetched
        assert missing not in self.manager._late_prefetches
        assert self.manager._prefetch_job_keys == {}
        assert_prefetch_accounting(stats, self.manager)

    def test_failed_prefetch_recovers_through_reactive_lookup(self, manager_setup):
        """Demand lookup can retry a key after its proactive load failed."""
        self._start_request()
        block = to_keys([1])[0]

        assert self.manager.prefetch_assume_resident([block], _CTX) == 1
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()
        assert self.primary_tier.lookup(block, _CTX) is LookupResult.MISS

        self.secondary_tier1.blocks[block] = True
        self.secondary_tier1.lookup = MagicMock(wraps=self.secondary_tier1.lookup)

        assert self.manager.lookup(block, _CTX) is LookupResult.RETRY
        self.secondary_tier1.lookup.assert_called_once_with(block, _CTX)
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_LOAD_FAILED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert block not in self.manager._prefetched
        assert block not in self.manager._late_prefetches
        assert_prefetch_accounting(stats, self.manager)

    def test_mixed_promotion_failure_counts_only_prefetch_keys(self, manager_setup):
        """A mixed failed batch attributes failure only to proactive keys."""
        self._start_request()
        demand, missing_prefetch = to_keys([1, 2])
        self.secondary_tier1.blocks[demand] = True
        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        assert self.manager.lookup(demand, _CTX) is LookupResult.RETRY
        assert self.manager.prefetch_assume_resident([missing_prefetch], _CTX) == 1

        self._simulate_on_schedule_end()
        self.secondary_tier1.submit_load.assert_called_once()
        job_metadata = self.secondary_tier1.submit_load.call_args.args[0]
        assert job_metadata.keys == [demand, missing_prefetch]
        assert self.manager._prefetch_job_keys[job_metadata.job_id] == (
            0,
            (missing_prefetch,),
        )

        self._simulate_on_schedule_end()

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_LOAD_FAILED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        assert missing_prefetch not in self.manager._prefetched
        assert self.manager._prefetch_job_keys == {}
        assert_prefetch_accounting(stats, self.manager)

    def test_prefetch_tracking_overflow_is_untracked_not_wasted(self, manager_setup):
        """Losing a tracking slot is not an outcome — the block may still be used.

        Counting overflow as wasted would make the success rate a function of
        prefetch_track_capacity and workload burst size rather than of prefetch
        quality.
        """
        self._start_request()
        self.manager._prefetch_track_capacity = 1
        first, second = to_keys([1, 2])
        self._queue_tracked_promotion(first)
        assert self.manager.lookup(first, _CTX) is LookupResult.HIT_PENDING
        assert first in self.manager._late_prefetches
        self._queue_tracked_promotion(second)

        # Capacity is 1, so tracking the second key displaces the first.
        assert first not in self.manager._prefetched
        assert first not in self.manager._late_prefetches
        assert second in self.manager._prefetched

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED) == 2
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_UNTRACKED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert_prefetch_accounting(stats, self.manager)

    def test_tracking_overflow_and_load_failure_are_disjoint(self, manager_setup):
        """A failed overflowed promotion must not also count as untracked."""
        self._start_request()
        self.manager._prefetch_track_capacity = 1
        first, second = to_keys([1, 2])

        assert self.manager.prefetch_assume_resident([first, second], _CTX) == 2
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED) == 2
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_LOAD_FAILED) == 2
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_UNTRACKED) == 0
        assert_prefetch_accounting(stats, self.manager)

    def test_repeated_prefetch_resolves_previous_record_as_wasted(self, manager_setup):
        """Re-promoting a tracked key must not silently drop its first record.

        A second promotion can only start once the block has left the primary
        tier, so the record being replaced resolved as wasted. Without this,
        two promotions would yield at most one outcome.
        """
        self._start_request()
        block = to_keys([1])[0]
        self._queue_tracked_promotion(block)
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT_PENDING
        assert block in self.manager._late_prefetches
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        # Evict the promoted copy, then release the filler's ref_cnt so the
        # primary tier has a slot free for a second promotion.
        filler = to_keys(range(10, 15))
        result = self.manager.prepare_store(filler, _CTX)
        assert result is not None
        assert block in result.evicted_keys
        self.manager.complete_store(filler, _CTX, success=True)
        # Two steps: the first flushes the cascades, the second drains them so
        # the filler stops holding ref_cnt and the primary tier has a free slot.
        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        # Queue the same key again before any demand lookup resolves its record.
        self._queue_tracked_promotion(block)
        assert block not in self.manager._late_prefetches

        stats = self.manager.get_stats()
        assert stats is not None
        promoted = counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED)
        useful = counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL)
        wasted = counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED)
        untracked = counter_total(stats, TieringOffloadingMetrics.PREFETCH_UNTRACKED)

        assert promoted == 2
        # The first record resolved as wasted; the second is still tracked.
        assert wasted == 1
        assert useful == 0
        assert untracked == 0
        assert block in self.manager._prefetched
        # Every promotion is accounted for: one resolved, one still in flight.
        assert_prefetch_accounting(stats, self.manager)

    def test_prefetch_outcome_tracking_disabled_reports_untracked(self, manager_setup):
        """Capacity 0 must still account for promotions, not drop them silently.

        Otherwise the accounting invariant breaks and a dashboard cannot tell a
        disabled tracker from a prefetch that never promotes anything.
        """
        self._start_request()
        self.manager._prefetch_track_capacity = 0
        block = to_keys([1])[0]
        self._queue_tracked_promotion(block)
        assert not self.manager._prefetched

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        stats = self.manager.get_stats()
        assert stats is not None
        promoted = counter_total(stats, TieringOffloadingMetrics.PREFETCH_PROMOTED)
        untracked = counter_total(stats, TieringOffloadingMetrics.PREFETCH_UNTRACKED)
        assert promoted == 1
        # Tracking off: untracked == promoted signals "ratio has no samples".
        assert untracked == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 0
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert_prefetch_accounting(stats, self.manager)

    def test_lookup_reports_sync_delay_for_resolved_lookups(self, manager_setup):
        """Resolved lookups report one sync delay sample on allocation."""
        self._start_request()
        blocks = to_keys(range(2))

        # No tier has these blocks: they resolve immediately as misses.
        for block in blocks:
            assert self.manager.lookup(block, _CTX) is LookupResult.MISS

        stats = self.manager.get_stats()
        if stats is not None:
            assert f"{TieringOffloadingMetrics.LOOKUP_SYNC_DELAY}_count" not in (
                stats.reduce()
            )

        self._simulate_on_schedule_end(new_req_ids=[_CTX.req_id])

        stats = self.manager.get_stats()
        assert stats is not None
        reduced = stats.reduce()
        assert reduced[f"{TieringOffloadingMetrics.LOOKUP_SYNC_DELAY}_count"] == 1
        assert f"{TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY}_count" not in reduced

    def test_lookup_reports_async_delay_across_promotion(self, manager_setup):
        """A new request reports async delay at schedule end."""
        self._start_request()
        block = to_keys(range(1))[0]
        self.secondary_tier1.blocks[block] = True

        # First lookup finds the block in a secondary tier and defers.
        assert self.manager.lookup(block, _CTX) is LookupResult.RETRY

        # The first scheduler step reports async delay for the new request.
        self._simulate_on_schedule_end(new_req_ids=[_CTX.req_id])
        stats = self.manager.get_stats()
        assert stats is not None
        reduced = stats.reduce()
        assert reduced[f"{TieringOffloadingMetrics.LOOKUP_SYNC_DELAY}_count"] == 1
        assert reduced[f"{TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY}_count"] == 1

        # Promotion completes on the next scheduler step.
        self._simulate_on_schedule_end()

        # Next lookup resolves via the now-promoted primary-tier block.
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT

        stats = self.manager.get_stats()
        if stats is not None:
            assert f"{TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY}_count" not in (
                stats.reduce()
            )

    def test_lookup_reports_async_delay_on_request_finish(self, manager_setup):
        """Never-allocated lookup delays flush at teardown."""
        ctx = ReqContext(req_id="req_lookup_finish")
        self._start_request(ctx)
        block = to_keys(range(1))[0]
        self.secondary_tier1.blocks[block] = True

        # Lookup finds the block in a secondary tier and defers.
        assert self.manager.lookup(block, ctx) is LookupResult.RETRY

        self._simulate_on_schedule_end()
        stats = self.manager.get_stats()
        if stats is not None:
            assert f"{TieringOffloadingMetrics.LOOKUP_SYNC_DELAY}_count" not in (
                stats.reduce()
            )
            assert f"{TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY}_count" not in (
                stats.reduce()
            )

        # Request finishes before the deferred lookup is ever resolved.
        self.manager.on_request_finished(ctx)

        stats = self.manager.get_stats()
        assert stats is not None
        reduced = stats.reduce()
        assert reduced[f"{TieringOffloadingMetrics.LOOKUP_SYNC_DELAY}_count"] == 1
        assert reduced[f"{TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY}_count"] == 1

    def test_partial_lookup(self, manager_setup):
        """Test lookup with partial hits."""
        blocks = to_keys(range(5))

        # Store first 3 blocks to primary
        self._start_request()
        self.manager.prepare_store(blocks[:3], _CTX)
        self.manager.complete_store(blocks[:3], _CTX, success=True)

        # Lookup all 5 blocks should return 3 (first 3 found)
        assert count_hits(self.manager, blocks) == 3

    def test_eviction_in_primary_tier(self, manager_setup):
        """Test eviction in primary tier when capacity is exceeded."""
        # Primary tier has capacity of 5 blocks
        # First, fill the primary tier
        blocks = to_keys(range(5))
        self._start_request()
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 5
        self.manager.complete_store(blocks, _CTX, success=True)

        # End of step: release ref_cnt from cascade
        self._simulate_on_schedule_end()

        # Now try to store 2 more blocks (should trigger eviction)
        more_blocks = to_keys(range(5, 7))
        result = self.manager.prepare_store(more_blocks, _CTX)

        # Should evict 2 blocks from primary tier
        assert result is not None
        assert len(result.evicted_keys) == 2
        assert len(result.keys_to_store) == 2

    def test_touch_propagates_to_all_tiers(self, manager_setup):
        """Test that touch() propagates to all tiers."""
        blocks = to_keys(range(3))

        # Store blocks
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        self._simulate_on_schedule_end()
        # for secondary tiers to drain jobs, so primary tier's blocks are evictable.
        self._simulate_on_schedule_end()

        self.secondary_tier1.touch = MagicMock(wraps=self.secondary_tier1.touch)
        self.secondary_tier2.touch = MagicMock(wraps=self.secondary_tier2.touch)

        # Touch blocks
        self.manager.touch(blocks, _CTX)

        # Verify touch was called on primary tier (check LRU order)
        primary_keys = list(self.primary_tier._policy.evictable_blocks.keys())
        assert primary_keys[-3:] == list(reversed(blocks))

        # Verify touch was propagated to all secondary tiers
        self.secondary_tier1.touch.assert_called_once_with(blocks, _CTX)
        self.secondary_tier2.touch.assert_called_once_with(blocks, _CTX)

    def test_failed_store_no_cascade(self, manager_setup):
        """Test that failed GPU→primary store doesn't cascade."""
        blocks = to_keys(range(3))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Prepare store
        self._start_request()
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store with failure — cascade must not happen
        self.manager.complete_store(blocks, _CTX, success=False)

        # submit_store was never called on either secondary tier
        self.secondary_tier1.submit_store.assert_not_called()
        self.secondary_tier2.submit_store.assert_not_called()

    def test_lookup_batches_submit_load_per_request(self, manager_setup):
        """lookup() defers submit_load until on_schedule_end(), one per request.

        Blocks from different requests each get their own submit_load call, each
        carrying the correct req_context.
        """
        blocks = to_keys(range(4))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        ctx_a = ReqContext(req_id="req_a")
        ctx_b = ReqContext(req_id="req_b")

        # All lookups return RETRY: secondary hit triggers promotion
        assert self.manager.lookup(blocks[0], ctx_a) is LookupResult.RETRY
        assert self.manager.lookup(blocks[1], ctx_a) is LookupResult.RETRY
        assert self.manager.lookup(blocks[2], ctx_b) is LookupResult.RETRY
        assert self.manager.lookup(blocks[3], ctx_b) is LookupResult.RETRY

        # submit_load must not fire during lookup - only at end of step
        self.secondary_tier1.submit_load.assert_not_called()

        # simulate end of step
        self._simulate_on_schedule_end()

        assert self.secondary_tier1.submit_load.call_count == 2
        calls = self.secondary_tier1.submit_load.call_args_list
        jm_a = calls[0].args[0]
        jm_b = calls[1].args[0]
        assert set(jm_a.keys) == {blocks[0], blocks[1]}
        assert jm_a.req_context is ctx_a
        assert set(jm_b.keys) == {blocks[2], blocks[3]}
        assert jm_b.req_context is ctx_b

    def test_lookup_shared_block_no_duplicate_promotion(self, manager_setup):
        """A block looked up by two requests in the same step is promoted once.

        The first lookup initiates promotion (returns None via secondary hit).
        The second lookup sees ref_cnt=-1 on the primary slot and returns None
        via the primary in-flight path — without triggering a second promotion.
        """
        shared_block = to_keys([0])[0]
        self.secondary_tier1.blocks[shared_block] = True

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        ctx_a = ReqContext(req_id="req_a")
        ctx_b = ReqContext(req_id="req_b")

        result_a = self.manager.lookup(shared_block, ctx_a)
        result_b = self.manager.lookup(shared_block, ctx_b)

        # First lookup triggers promotion (RETRY), second finds block
        # already in primary with write in-flight (HIT_PENDING).
        assert result_a is LookupResult.RETRY
        assert result_b is LookupResult.HIT_PENDING

        self._simulate_on_schedule_end()

        # Only one submit_load call despite two lookups
        self.secondary_tier1.submit_load.assert_called_once()
        job_metadata = self.secondary_tier1.submit_load.call_args.args[0]
        assert list(job_metadata.keys) == [shared_block]
        assert job_metadata.req_context is ctx_a

    def test_complete_store_forwards_req_context_to_submit_store(self, manager_setup):
        """complete_store cascades to secondary tiers with the correct req_context."""
        blocks = to_keys(range(2))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )

        ctx = ReqContext(req_id="req_ctx", kv_transfer_params={"key": "value"})

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.complete_store(blocks, ctx, success=True)

        assert self.secondary_tier1.submit_store.call_count == 1
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert job_metadata.req_context is ctx

    def test_on_request_finished_delays_secondary_until_store_submitted(
        self, manager_setup
    ):
        """Manager hook is eager; secondary hooks wait for cascade submission."""
        blocks = to_keys(range(2))
        ctx = ReqContext(req_id="req_delayed_secondary")
        calls: list[tuple[str, str]] = []

        self.primary_tier.on_request_finished = MagicMock(
            side_effect=lambda req_context: calls.append(
                ("primary_finish", req_context.req_id)
            )
        )

        original_submit_store1 = self.secondary_tier1.submit_store
        original_submit_store2 = self.secondary_tier2.submit_store

        def submit_store1(job_metadata):
            calls.append(("submit_store_1", job_metadata.req_context.req_id))
            return original_submit_store1(job_metadata)

        def submit_store2(job_metadata):
            calls.append(("submit_store_2", job_metadata.req_context.req_id))
            return original_submit_store2(job_metadata)

        self.secondary_tier1.submit_store = MagicMock(side_effect=submit_store1)
        self.secondary_tier2.submit_store = MagicMock(side_effect=submit_store2)
        self.secondary_tier1.on_request_finished = MagicMock(
            side_effect=lambda req_context: calls.append(
                ("secondary_finish_1", req_context.req_id)
            )
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            side_effect=lambda req_context: calls.append(
                ("secondary_finish_2", req_context.req_id)
            )
        )

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.on_request_finished(ctx)

        assert calls == [("primary_finish", ctx.req_id)]
        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.complete_store(blocks, ctx, success=True)

        assert calls == [
            ("primary_finish", ctx.req_id),
            ("submit_store_1", ctx.req_id),
            ("submit_store_2", ctx.req_id),
            ("secondary_finish_1", ctx.req_id),
            ("secondary_finish_2", ctx.req_id),
        ]

    def test_failed_store_finalizes_finished_request(self, manager_setup):
        """Failed primary stores still unblock secondary finalization."""
        blocks = to_keys(range(2))
        ctx = ReqContext(req_id="req_failed_store_finalize")

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )
        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.complete_store(blocks, ctx, success=False)

        self.secondary_tier1.submit_store.assert_not_called()
        self.secondary_tier2.submit_store.assert_not_called()
        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_zero_store_request_finalizes_immediately(self, manager_setup):
        """Requests with no pending stores finalize secondary tiers immediately."""
        ctx = ReqContext(req_id="req_zero_store_finalize")

        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_reset_cache_finalizes_delayed_secondary_request(self, manager_setup):
        """reset_cache abandons pending primary stores and finalizes secondaries."""
        blocks = to_keys(range(2))
        ctx = ReqContext(req_id="req_reset_finalize_secondary")

        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.prepare_store(blocks, ctx)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.reset_cache()

        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert self.manager._req_state == {}

    def test_reset_cache_clears_pending_primary_stores_for_active_request(
        self, manager_setup
    ):
        """reset_cache drops active pending stores so resumed requests finalize."""
        initial_blocks = to_keys(range(2))
        resumed_blocks = to_keys(range(2, 4))
        ctx = ReqContext(req_id="req_reset_resume")

        self.secondary_tier1.on_request_finished = MagicMock(
            wraps=self.secondary_tier1.on_request_finished
        )
        self.secondary_tier2.on_request_finished = MagicMock(
            wraps=self.secondary_tier2.on_request_finished
        )

        self._start_request(ctx)
        self.manager.prepare_store(initial_blocks, ctx)
        assert self.manager._req_state[ctx.req_id].pending_primary_stores == 1

        self.manager.reset_cache()

        assert ctx.req_id in self.manager._req_state
        assert self.manager._req_state[ctx.req_id].pending_primary_stores == 0
        self.secondary_tier1.on_request_finished.assert_not_called()
        self.secondary_tier2.on_request_finished.assert_not_called()

        self.manager.prepare_store(resumed_blocks, ctx)
        self.manager.complete_store(resumed_blocks, ctx, success=True)
        self.manager.on_request_finished(ctx)

        self.secondary_tier1.on_request_finished.assert_called_once_with(ctx)
        self.secondary_tier2.on_request_finished.assert_called_once_with(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_on_new_request_lifecycle(self, manager_setup):
        """Policy defaults to BLOCK_LEVEL, escalates when a tier requests it,
        and is cleaned up on on_request_finished."""
        # Default: all tiers return BLOCK_LEVEL
        ctx = ReqContext(req_id="req_policy_lifecycle")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.BLOCK_LEVEL
        assert self.manager._req_state[ctx.req_id].request_level_tiers is None
        self.manager.on_request_finished(ctx)
        assert ctx.req_id not in self.manager._req_state

        # Escalate: tier1 requests REQUEST_LEVEL
        self.secondary_tier1.on_new_request = lambda req_context: (
            RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        )

        ctx = ReqContext(req_id="req_policy_lifecycle_2")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.REQUEST_LEVEL
        assert self.manager._req_state[ctx.req_id].request_level_tiers == {
            self.secondary_tier1
        }

        # Cleanup
        self.manager.on_request_finished(ctx)
        assert ctx.req_id not in self.manager._req_state

    def test_prepare_store_cascades_existing_blocks_to_request_level_tiers(
        self, manager_setup
    ):
        """prepare_store cascades hit blocks to request-level tiers only."""
        # Store some blocks to primary first
        existing_blocks = to_keys(range(3))
        self._start_request()
        result = self.manager.prepare_store(existing_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(existing_blocks, _CTX, success=True)
        # Drain cascade completions
        self._simulate_on_schedule_end()

        # Make tier1 request-level, tier2 stays block-level
        self.secondary_tier1.on_new_request = lambda req_context: (
            RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        )

        ctx = ReqContext(req_id="req_cascade")
        self.manager.on_new_request(ctx)

        # Spy on submit_store
        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Call prepare_store with existing + new blocks
        new_blocks = to_keys(range(3, 5))
        all_blocks = existing_blocks + new_blocks
        result = self.manager.prepare_store(all_blocks, ctx)
        assert result is not None
        assert set(result.keys_to_store) == set(new_blocks)

        # Only tier1 (request-level) should get existing blocks cascaded now.
        # New blocks are cascaded to ALL tiers later via complete_store().
        self.secondary_tier1.submit_store.assert_called_once()
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert set(job_metadata.keys) == set(existing_blocks)

        # tier2 (block-level) does not get existing blocks here.
        self.secondary_tier2.submit_store.assert_not_called()

    def test_reset_cache_clears_orchestrator_state(self, manager_setup):
        """reset_cache wipes every kind of orchestrator state and resets
        primary tier; pending submissions are dropped without being sent
        to the secondary tier. Active request state is retained."""
        # Cascade — populates primary blocks and leaves cascade jobs
        # in _transfer_jobs (the synchronous example tier has already
        # queued completions); reset_cache's drain loop will pick them up.
        blocks = to_keys(range(3))
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        assert self.manager._transfer_jobs

        # Pending promotion submission (deferred — no on_schedule_end after
        # the lookup that staged it).
        promo_block = to_keys([99])[0]
        self.secondary_tier1.blocks[promo_block] = True
        assert (
            self.manager.lookup(promo_block, ReqContext(req_id="pending"))
            is LookupResult.RETRY
        )
        assert self.manager._pending_load_submissions

        # Request-level tier registration.
        self.secondary_tier1.on_new_request = lambda req_context: (
            RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        )
        rl_ctx = ReqContext(req_id="rl")
        self.manager.on_new_request(rl_ctx)
        assert self.manager._req_state[rl_ctx.req_id].request_level_tiers == {
            self.secondary_tier1
        }

        # Mark this step as already polled (reset_cache must clear it).
        self.manager._processed_jobs_this_step = True

        # Spy: pending submission must NOT reach the tier.
        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        self.manager.reset_cache()

        # Orchestrator state cleared.
        assert self.manager._transfer_jobs == {}
        assert self.manager._pending_load_submissions == {}
        assert set(self.manager._req_state) == {_CTX.req_id, rl_ctx.req_id}
        assert self.manager._processed_jobs_this_step is False

        # Primary tier reset to a fresh state.
        assert self.primary_tier._num_allocated_blocks == 0
        assert self.primary_tier._free_list == []
        for block in blocks:
            assert self.primary_tier.lookup(block, _CTX) is LookupResult.MISS

        # Pending submission was dropped, not submitted.
        self.secondary_tier1.submit_load.assert_not_called()

    def test_reset_cache_resolves_and_clears_prefetch_attribution(self, manager_setup):
        self._start_request()
        block = to_keys([1])[0]
        self._queue_tracked_promotion(block)
        assert self.manager.lookup(block, _CTX) is LookupResult.HIT_PENDING

        # Flush the promotion. The synchronous example tier has completed it,
        # but the manager has not consumed the completion or its provenance.
        self._simulate_on_schedule_end()
        assert self.manager._prefetch_job_keys
        assert block in self.manager._prefetched
        assert block in self.manager._late_prefetches

        self.manager.reset_cache()

        stats = self.manager.get_stats()
        assert stats is not None
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_WASTED) == 1
        assert counter_total(stats, TieringOffloadingMetrics.PREFETCH_USEFUL) == 0
        assert self.manager._prefetched == {}
        assert self.manager._late_prefetches == set()
        assert self.manager._prefetch_job_keys == {}
        assert self.manager._transfer_jobs == {}
        assert self.manager._pending_load_submissions == {}
        assert self.primary_tier.lookup(block, _CTX) is LookupResult.MISS

    def test_reset_cache_drains_all_tiers(self, manager_setup):
        """reset_cache must drain each secondary tier before resetting
        the primary tier so no tier I/O is touching primary memory.
        Without the drain, an in-flight transfer could write into, or
        read junk from, a primary slot that the post-reset path has
        reallocated.
        """
        self.secondary_tier1.drain_jobs = MagicMock(
            wraps=self.secondary_tier1.drain_jobs
        )
        self.secondary_tier2.drain_jobs = MagicMock(
            wraps=self.secondary_tier2.drain_jobs
        )

        # Drive a cascade so a job lands in _transfer_jobs.
        blocks = to_keys(range(3))
        self._start_request()
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        assert self.manager._transfer_jobs

        self.manager.reset_cache()

        self.secondary_tier1.drain_jobs.assert_called_once()
        self.secondary_tier2.drain_jobs.assert_called_once()
        assert self.manager._transfer_jobs == {}

    @pytest.mark.parametrize(
        "load_tier_filter",
        [
            TierFilter(matchers=(TierMatcher(medium=Medium.STORAGE),)),
            TierFilter(matchers=()),
        ],
        ids=["non_matching_medium", "empty_no_load"],
    )
    def test_tier_filter_skips_filtered_secondary(
        self, manager_setup, load_tier_filter
    ):
        """Filter excluding secondary medium returns MISS from secondaries
        even when they hold the block; primary is unaffected."""
        blocks = to_keys(range(2))
        # Put one block in primary, one only in secondary
        self._start_request()
        self.manager.prepare_store(blocks[:1], _CTX)
        self.manager.complete_store(blocks[:1], _CTX, success=True)
        self.secondary_tier1.blocks[blocks[1]] = True

        # Secondaries have medium=CPU, so load_tier_filter skips them.
        self.secondary_tier1.lookup = MagicMock(wraps=self.secondary_tier1.lookup)

        ctx = ReqContext(req_id="r1", load_tier_filter=load_tier_filter)
        assert self.manager.lookup(blocks[0], ctx) is LookupResult.HIT
        assert self.manager.lookup(blocks[1], ctx) is LookupResult.MISS
        self.secondary_tier1.lookup.assert_not_called()

    @pytest.mark.parametrize(
        "load_tier_filter",
        [
            TierFilter.ALL,
            TierFilter(matchers=(TierMatcher(medium=Medium.CPU),)),
            TierFilter(matchers=(TierMatcher(),)),
        ],
        ids=["all", "explicit_cpu", "unconstrained_matcher"],
    )
    def test_tier_filter_allows_matching_secondary(
        self, manager_setup, load_tier_filter
    ):
        """Filter that matches the secondary's medium allows lookup."""
        blocks = to_keys(range(1))
        self.secondary_tier1.blocks[blocks[0]] = True

        self.secondary_tier1.lookup = MagicMock(wraps=self.secondary_tier1.lookup)

        ctx = ReqContext(req_id="r2", load_tier_filter=load_tier_filter)
        assert self.manager.lookup(blocks[0], ctx) is LookupResult.RETRY
        self.secondary_tier1.lookup.assert_called()


class TestTieringOffloadingWithoutSecondaryTiers:
    """Test TieringOffloadingManager with no secondary tiers (backward compat)."""

    def test_works_without_secondary_tiers(self):
        """Test that manager works with empty secondary_tiers list."""
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=5, mmap_region=_mock_mmap_region(5)
        )

        # Create manager with no secondary tiers
        manager = TieringOffloadingManager(
            primary_tier=primary_tier, secondary_tiers=[]
        )

        blocks = to_keys(range(3))

        # Should work like a regular OffloadingManager
        manager.on_new_request(_CTX)
        result = manager.prepare_store(blocks, _CTX)
        assert result is not None
        manager.complete_store(blocks, _CTX, success=True)

        assert count_hits(manager, blocks) == 3


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            [{"medium": "storage"}],
            TierFilter(matchers=(TierMatcher(medium=Medium.STORAGE),)),
        ),
        (
            [{"medium": "CPU"}],
            TierFilter(matchers=(TierMatcher(medium=Medium.CPU),)),
        ),
        (
            [{}],
            TierFilter(matchers=(TierMatcher(),)),
        ),
        (
            [{"medium": "storage", "locality": "local"}],
            TierFilter(
                matchers=(TierMatcher(medium=Medium.STORAGE, locality=Locality.LOCAL),)
            ),
        ),
        (
            [{"medium": "cpu"}, {"medium": "storage"}],
            TierFilter(
                matchers=(
                    TierMatcher(medium=Medium.CPU),
                    TierMatcher(medium=Medium.STORAGE),
                )
            ),
        ),
        (
            [],
            TierFilter(matchers=()),
        ),
    ],
    ids=[
        "medium_storage",
        "medium_cpu_uppercase",
        "unconstrained",
        "with_locality",
        "multiple_matchers",
        "empty_list_deny_all",
    ],
)
def test_parse_tier_filter_valid(raw, expected):
    assert _parse_tier_filter(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "not a list",
        [{"medium": "unknown"}],
        [{"locality": "nowhere"}],
    ],
    ids=["non_list", "invalid_medium", "invalid_locality"],
)
def test_parse_tier_filter_invalid_returns_all(raw):
    assert _parse_tier_filter(raw) is TierFilter.ALL


def test_parse_tier_filter_skips_bad_entries():
    result = _parse_tier_filter(
        [
            {"medium": "storage"},
            "not a dict",
            {"medium": "bogus"},
            {"medium": "cpu"},
        ]
    )
    assert result.matchers == (
        TierMatcher(medium=Medium.STORAGE),
        TierMatcher(medium=Medium.CPU),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
