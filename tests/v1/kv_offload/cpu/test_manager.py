# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.manager import AllocationMode, CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy


def make_req_context(
    req_id: str = "", kv_transfer_params: dict | None = None
) -> ReqContext:
    """Create a ReqContext as production code would, from a request's params."""
    return ReqContext(req_id=req_id, kv_transfer_params=kv_transfer_params)


_EMPTY_REQ_CTX = make_req_context()


def make_cpu_manager(
    num_blocks: int = 4,
    cache_policy: str = "lru",
    cache_policy_module_path: str | None = None,
    enable_events: bool = False,
    store_threshold: int = 0,
    max_tracker_size: int = 64_000,
    speculative_reserve_blocks: int = 0,
) -> CPUOffloadingManager:
    return CPUOffloadingManager(
        num_blocks=num_blocks,
        cache_policy=cache_policy,
        cache_policy_module_path=cache_policy_module_path,
        enable_events=enable_events,
        store_threshold=store_threshold,
        max_tracker_size=max_tracker_size,
        speculative_reserve_blocks=speculative_reserve_blocks,
    )


@dataclass
class ExpectedPrepareStoreOutput:
    keys_to_store: list[int]
    store_block_ids: list[int]
    evicted_keys: list[int]


def to_key(int_hash: int) -> OffloadKey:
    return make_offload_key(str(int_hash).encode(), 0)


def to_keys(int_hashes: list[int]) -> list[OffloadKey]:
    return [to_key(i) for i in int_hashes]


def verify_store_output(
    prepare_store_output: PrepareStoreOutput | None,
    expected_prepare_store_output: ExpectedPrepareStoreOutput,
):
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == to_keys(
        expected_prepare_store_output.keys_to_store
    )
    assert prepare_store_output.evicted_keys == to_keys(
        expected_prepare_store_output.evicted_keys
    )
    store_spec = prepare_store_output.store_spec
    assert isinstance(store_spec, CPULoadStoreSpec)
    expected_array = np.array(
        expected_prepare_store_output.store_block_ids, dtype=np.int64
    )
    assert np.array_equal(expected_array, store_spec.block_ids)


def verify_load_output(
    prepare_load_output: LoadStoreSpec, expected_prepare_load_output: list[int]
):
    assert isinstance(prepare_load_output, CPULoadStoreSpec)
    expected_array = np.array(expected_prepare_load_output, dtype=np.int64)
    assert np.array_equal(expected_array, prepare_load_output.block_ids)


def check_split_usage_stats(
    manager: CPUOffloadingManager, write: float, read: float, total: float
):
    stats = manager.get_stats()
    assert stats is not None
    reduced = stats.reduce()
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC] == pytest.approx(
        write
    )
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC] == pytest.approx(
        read
    )
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC] == pytest.approx(total)


def verify_events(
    events: Iterable[OffloadingEvent],
    expected_stores: tuple[set[int], ...] = (),
    expected_evictions: tuple[set[int], ...] = (),
):
    stores: list[set[OffloadKey]] = []
    evictions: list[set[OffloadKey]] = []
    for event in events:
        assert event.medium == Medium.CPU
        if event.removed:
            evictions.append(set(event.keys))
        else:
            stores.append(set(event.keys))

    def to_key_sets(
        int_sets: tuple[set[int], ...],
    ) -> tuple[set[OffloadKey], ...]:
        return tuple([set(to_keys(list(int_set))) for int_set in int_sets])

    assert tuple(evictions) == to_key_sets(expected_evictions)
    assert tuple(stores) == to_key_sets(expected_stores)


def test_cpu_eviction_removed_precedes_stored():
    """An eviction is announced before the store that reuses its capacity."""
    manager = make_cpu_manager(num_blocks=2, enable_events=True)

    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    list(manager.take_events())

    manager.prepare_store(to_keys([3]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([3]), _EMPTY_REQ_CTX)

    events = list(manager.take_events())
    removed_idx = [i for i, event in enumerate(events) if event.removed]
    stored_idx = [i for i, event in enumerate(events) if not event.removed]
    assert removed_idx and stored_idx, events
    assert max(removed_idx) < min(stored_idx)
    assert all(event.medium == manager.medium for event in events)


@pytest.mark.parametrize("eviction_policy", ["lru", "arc"])
def test_already_stored_block_not_evicted_during_prepare_store(eviction_policy):
    """
    Regression test: a block that is already stored must not be evicted
    by prepare_store() when it needs to make room for new blocks.
    Applies to both lru and arc policies.

    Scenario:
        - Store blocks [1, 2] and complete.
        - touch([1]) makes block 2 the LRU candidate.
        - prepare_store([2, 3, 4, 5]):
            * block 2 is filtered out as "already stored"
            * but without the fix, block 2 would be evicted as the LRU
              candidate to make room for [3, 4, 5]
        - After complete_store([2, 3, 4, 5]), block 2 must still be present.
    """
    manager = make_cpu_manager(
        num_blocks=4,
        cache_policy=eviction_policy,
        enable_events=True,
    )

    # store [1, 2] and complete
    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

    # touch [1] to make block 2 the LRU candidate
    manager.touch(to_keys([1]), _EMPTY_REQ_CTX)

    # prepare_store([2, 3, 4, 5]):
    #   - block 2 is already stored -> filtered out of keys_to_store
    #   - block 2 must NOT be evicted even though it is the LRU candidate
    #   - block 1 (ID 0) is evicted instead; new blocks [3,4,5] get IDs 2,3,0
    prepare_store_output = manager.prepare_store(to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            evicted_keys=[1],  # block 1 evicted, not block 2
        ),
    )

    # complete_store must not silently drop block 2
    manager.complete_store(to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX)

    # block 2 must still be present in the cache
    assert manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT


def test_filter_reused_manager_reports_stores_skipped_counter():
    manager = make_cpu_manager(
        num_blocks=4,
        cache_policy="lru",
        store_threshold=2,
    )

    prepare_store_output = manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)

    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[],
            store_block_ids=[],
            evicted_keys=[],
        ),
    )
    stats = manager.get_stats()
    assert stats is not None
    assert stats.reduce()[CPUOffloadingMetrics.STORES_SKIPPED] == 3
    stats = manager.get_stats()
    assert stats is not None
    assert stats.reduce()[CPUOffloadingMetrics.STORES_SKIPPED] == 0


def test_cpu_manager_reports_cache_usage_gauge():
    def check_usage_stats(manager: CPUOffloadingManager, value: float):
        stats = manager.get_stats()
        assert stats is not None
        assert stats.reduce()[
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC
        ] == pytest.approx(value)

    # Zero-capacity manager always reports 0.0
    manager = make_cpu_manager(num_blocks=0)
    check_usage_stats(manager, 0.0)

    # Empty manager (4 blocks, none allocated): usage = 0.0
    manager = make_cpu_manager(num_blocks=4)
    check_usage_stats(manager, 0.0)

    # After allocating 2 of 4 blocks: usage = 0.5
    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    check_usage_stats(manager, 0.5)

    # After filling all 4 blocks: usage = 1.0
    manager.prepare_store(to_keys([3, 4]), _EMPTY_REQ_CTX)
    check_usage_stats(manager, 1.0)

    # After completing store, the blocks becomes evictable as it is not actively used
    # and usage drops.
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    check_usage_stats(manager, 0.5)

    # After completing store, the blocks becomes evictable as it is not actively used
    # and usage drops.
    manager.complete_store(to_keys([3, 4]), _EMPTY_REQ_CTX)
    check_usage_stats(manager, 0.0)


def test_cpu_manager_reports_allocation_size_histogram():
    manager = make_cpu_manager(num_blocks=4, cache_policy="lru")

    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)

    stats = manager.get_stats()

    assert stats is not None
    reduced = stats.reduce()
    assert reduced[f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_count"] == 2
    assert reduced[f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_sum"] == 3

    # The cache-usage gauge is always reported, so get_stats() never returns
    # None, but the histogram has nothing new once its samples are consumed.
    second_stats = manager.get_stats()
    assert second_stats is not None
    assert f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_count" not in (
        second_stats.reduce()
    )


def test_cpu_manager_reports_allocation_size_on_allocation_failure(monkeypatch):
    manager = make_cpu_manager(num_blocks=4, cache_policy="lru")

    def fail_allocate_blocks(keys):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(manager, "_allocate_blocks", fail_allocate_blocks)

    with pytest.raises(RuntimeError, match="allocation failed"):
        manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)

    stats = manager.get_stats()

    assert stats is not None
    reduced = stats.reduce()
    assert reduced[f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_count"] == 1
    assert reduced[f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_sum"] == 3


def test_cpu_manager_reports_allocation_size_on_eviction_failure():
    manager = make_cpu_manager(num_blocks=1, cache_policy="lru")

    manager.prepare_store(to_keys([1]), _EMPTY_REQ_CTX)
    manager.get_stats()

    assert manager.prepare_store(to_keys([2]), _EMPTY_REQ_CTX) is None

    stats = manager.get_stats()

    assert stats is not None
    reduced = stats.reduce()
    assert reduced[f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_count"] == 1
    assert reduced[f"{CPUOffloadingMetrics.CPU_ALLOCATION_SIZE}_sum"] == 1


def test_cpu_manager_reports_cache_write_and_read_usage_gauges():
    manager = make_cpu_manager(num_blocks=4)

    # Store path: pins write usage until complete_store.
    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    check_split_usage_stats(manager, write=0.5, read=0.0, total=0.5)

    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    check_split_usage_stats(manager, write=0.0, read=0.0, total=0.0)

    # Load path: pins read usage until complete_load.
    assert manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.HIT
    manager.prepare_load(to_keys([1]), _EMPTY_REQ_CTX)
    check_split_usage_stats(manager, write=0.0, read=0.25, total=0.25)

    manager.complete_load(to_keys([1]), _EMPTY_REQ_CTX)
    check_split_usage_stats(manager, write=0.0, read=0.0, total=0.0)

    # Concurrent write + read pins are both reflected and additive.
    manager.prepare_store(to_keys([3, 4]), _EMPTY_REQ_CTX)
    manager.prepare_load(to_keys([2]), _EMPTY_REQ_CTX)
    check_split_usage_stats(manager, write=0.5, read=0.25, total=0.75)


def test_cpu_manager_clears_write_usage_after_failed_store():
    manager = make_cpu_manager(num_blocks=4)

    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    check_split_usage_stats(manager, write=0.5, read=0.0, total=0.5)

    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX, success=False)
    check_split_usage_stats(manager, write=0.0, read=0.0, total=0.0)


def test_cpu_manager():
    """
    Tests CPUOffloadingManager with lru policy.
    """
    # initialize a CPU manager with a capacity of 4 blocks
    cpu_manager = make_cpu_manager(num_blocks=4, cache_policy="lru", enable_events=True)

    # prepare store [1, 2]
    prepare_store_output = cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[1, 2],
            store_block_ids=[0, 1],
            evicted_keys=[],
        ),
    )

    # lookup [1, 2] -> write in-flight, not yet ready
    assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.HIT_PENDING
    assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT_PENDING

    # no events so far
    assert list(cpu_manager.take_events()) == []

    # complete store [1, 2]
    cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    verify_events(cpu_manager.take_events(), expected_stores=({1, 2},))

    # lookup [1, 2]
    assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is LookupResult.MISS

    # prepare store [2, 3, 4, 5] -> evicts [1]
    prepare_store_output = cpu_manager.prepare_store(
        to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX
    )
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            evicted_keys=[1],
        ),
    )

    # verify eviction event
    verify_events(cpu_manager.take_events(), expected_evictions=({1},))

    # prepare store with no space
    assert cpu_manager.prepare_store(to_keys([1, 6]), _EMPTY_REQ_CTX) is None

    # complete store [2, 3, 4, 5]
    cpu_manager.complete_store(to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX)

    # lookup (now that we have [2, 3, 4, 5])
    assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.MISS
    assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(4), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(5), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(0), _EMPTY_REQ_CTX) is LookupResult.MISS

    # prepare load [2, 3]
    prepare_load_output = cpu_manager.prepare_load(to_keys([2, 3]), _EMPTY_REQ_CTX)
    verify_load_output(prepare_load_output, [1, 2])

    # prepare store with no space ([2, 3] is being loaded)
    assert cpu_manager.prepare_store(to_keys([6, 7, 8]), _EMPTY_REQ_CTX) is None

    # complete load [2, 3]. Load changes the eviction list, making 2, 3 recent.
    cpu_manager.complete_load(to_keys([2, 3]), _EMPTY_REQ_CTX)

    # prepare store [6, 7, 8] -> evicts [4, 5, 2] (oldest)
    prepare_store_output = cpu_manager.prepare_store(to_keys([6, 7, 8]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[6, 7, 8],
            store_block_ids=[1, 0, 3],
            evicted_keys=[4, 5, 2],
        ),
    )

    # complete store [6, 7, 8]
    cpu_manager.complete_store(to_keys([6, 7, 8]), _EMPTY_REQ_CTX)

    # touch [3, 6, 7] (move to end of LRU order)
    cpu_manager.touch(to_keys([3, 6, 7]), _EMPTY_REQ_CTX)

    # prepare store [7, 9] -> evicts [8] (oldest following previous touch)
    prepare_store_output = cpu_manager.prepare_store(to_keys([9]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[9],
            store_block_ids=[3],
            evicted_keys=[8],
        ),
    )

    # complete store [7, 9] with failure
    cpu_manager.complete_store(to_keys([7, 9]), _EMPTY_REQ_CTX, success=False)

    # assert [7] is still stored, but [9] is not
    assert cpu_manager.lookup(to_key(7), _EMPTY_REQ_CTX) is LookupResult.HIT
    assert cpu_manager.lookup(to_key(9), _EMPTY_REQ_CTX) is LookupResult.MISS

    verify_events(
        cpu_manager.take_events(),
        expected_stores=({3, 4, 5}, {6, 7, 8}),
        expected_evictions=({4, 5, 2}, {8}),
    )


def test_prepare_load_preserves_key_order():
    """block_ids[i] must correspond to keys[i] (co-indexed invariant)."""
    manager = make_cpu_manager(num_blocks=4, cache_policy="lru")

    key_a, key_b, key_c = to_key(0), to_key(1), to_key(2)

    # Store all three keys and learn their block ID assignments
    store_output = manager.prepare_store([key_a, key_b, key_c], _EMPTY_REQ_CTX)
    assert store_output is not None
    assert isinstance(store_output.store_spec, CPULoadStoreSpec)
    key_to_block_id = {
        k: int(bid)
        for k, bid in zip(store_output.keys_to_store, store_output.store_spec.block_ids)
    }
    manager.complete_store([key_a, key_b, key_c], _EMPTY_REQ_CTX)

    # Forward order: [a, b, c]
    spec_fwd = manager.prepare_load([key_a, key_b, key_c], _EMPTY_REQ_CTX)
    assert isinstance(spec_fwd, CPULoadStoreSpec)
    assert [int(x) for x in spec_fwd.block_ids] == [
        key_to_block_id[key_a],
        key_to_block_id[key_b],
        key_to_block_id[key_c],
    ]
    manager.complete_load([key_a, key_b, key_c], _EMPTY_REQ_CTX)  # order irrelevant

    # Arbitrary permutation: [b, c, a]
    spec_perm = manager.prepare_load([key_b, key_c, key_a], _EMPTY_REQ_CTX)
    assert isinstance(spec_perm, CPULoadStoreSpec)
    assert [int(x) for x in spec_perm.block_ids] == [
        key_to_block_id[key_b],
        key_to_block_id[key_c],
        key_to_block_id[key_a],
    ]
    manager.complete_load([key_a, key_b, key_c], _EMPTY_REQ_CTX)  # order irrelevant


class TestARCPolicy:
    """Unit tests for CPUOffloadingManager with ARC eviction policy."""

    def _make_manager(
        self, num_blocks: int = 4, enable_events: bool = True
    ) -> tuple[CPUOffloadingManager, ARCCachePolicy]:
        manager = make_cpu_manager(
            num_blocks=num_blocks,
            cache_policy="arc",
            enable_events=enable_events,
        )
        policy = manager._policy
        assert isinstance(policy, ARCCachePolicy)
        return manager, policy

    def test_basic(self):
        """
        Tests CPUOffloadingManager with arc policy.
        Verifies that ARC handles store, load, and lookup operations correctly.
        """
        cpu_manager, arc_policy = self._make_manager()

        # prepare store [1, 2]
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([1, 2]), _EMPTY_REQ_CTX
        )
        verify_store_output(
            prepare_store_output,
            ExpectedPrepareStoreOutput(
                keys_to_store=[1, 2],
                store_block_ids=[0, 1],
                evicted_keys=[],
            ),
        )

        # lookup [1, 2] -> write in-flight, not yet ready
        assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.HIT_PENDING
        assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT_PENDING

        # no events so far
        assert list(cpu_manager.take_events()) == []

        # complete store [1, 2]
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        verify_events(cpu_manager.take_events(), expected_stores=({1, 2},))

        # lookup [1, 2]
        assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.HIT
        assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT
        assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is LookupResult.MISS

        # blocks should be in T1 (recent)
        assert len(arc_policy.t1) == 2
        assert len(arc_policy.t2) == 0

    def test_t1_to_t2_promotion(self):
        """
        Tests that accessing a block in T1 promotes it to T2 (frequent).
        This is a key feature of ARC's adaptive behavior.
        """
        cpu_manager, arc_policy = self._make_manager(enable_events=False)

        # store and complete block 1
        cpu_manager.prepare_store(to_keys([1]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1]), _EMPTY_REQ_CTX)

        # block 1 starts in T1 (recent)
        assert to_keys([1])[0] in arc_policy.t1
        assert to_keys([1])[0] not in arc_policy.t2

        # touch block 1 (simulate second access)
        cpu_manager.touch(to_keys([1]), _EMPTY_REQ_CTX)

        # block 1 should now be in T2 (frequent)
        assert to_keys([1])[0] not in arc_policy.t1
        assert to_keys([1])[0] in arc_policy.t2

    def test_eviction_with_load(self):
        """
        Tests ARC eviction behavior similar to LRU test.
        Verifies that blocks being loaded (ref_cnt > 0) cannot be evicted.
        """
        cpu_manager, _ = self._make_manager()

        # prepare and complete store [1, 2, 3, 4]
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX
        )
        verify_store_output(
            prepare_store_output,
            ExpectedPrepareStoreOutput(
                keys_to_store=[1, 2, 3, 4],
                store_block_ids=[0, 1, 2, 3],
                evicted_keys=[],
            ),
        )
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # prepare load [2, 3] (increases ref_cnt)
        prepare_load_output = cpu_manager.prepare_load(to_keys([2, 3]), _EMPTY_REQ_CTX)
        verify_load_output(prepare_load_output, [1, 2])

        # prepare store [5, 6, 7] with [2, 3] being loaded
        # should fail because [2, 3] have ref_cnt > 0
        assert cpu_manager.prepare_store(to_keys([5, 6, 7]), _EMPTY_REQ_CTX) is None

        # complete load [2, 3]
        cpu_manager.complete_load(to_keys([2, 3]), _EMPTY_REQ_CTX)

        # now prepare store [5, 6, 7] should succeed
        # ARC will evict blocks one at a time from T1 as needed
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([5, 6, 7]), _EMPTY_REQ_CTX
        )
        assert prepare_store_output is not None
        # Should successfully evict enough blocks to make room (at least 1)
        assert len(prepare_store_output.evicted_keys) >= 1

    def test_adaptive_target(self):
        """
        Tests ARC's adaptive target adjustment via ghost lists.
        When a block in B1 (ghost list) is accessed, target_t1_size increases.
        When a block in B2 is accessed, target_t1_size decreases.
        """
        cpu_manager, arc_policy = self._make_manager(num_blocks=2, enable_events=False)

        # store blocks 1, 2 (fills cache)
        cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

        initial_target = arc_policy.target_t1_size

        # store block 3, evicting block 1 (moves to B1 ghost list)
        cpu_manager.prepare_store(to_keys([3]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([3]), _EMPTY_REQ_CTX)

        # block 1 should be in B1 (ghost list)
        assert to_keys([1])[0] in arc_policy.b1

        # touch block 1 (cache miss, but in B1)
        # this should increase target_t1_size (favor recency)
        cpu_manager.touch(to_keys([1]), _EMPTY_REQ_CTX)

        # target should have increased
        assert arc_policy.target_t1_size > initial_target

    def test_t1_t2_eviction_policy(self):
        """
        Tests that ARC evicts from T1 or T2 based on target_t1_size.
        If |T1| >= target_t1_size, evict from T1, otherwise from T2.
        """
        cpu_manager, arc_policy = self._make_manager(enable_events=False)

        # store blocks 1, 2, 3, 4
        cpu_manager.prepare_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # promote blocks 3, 4 to T2 by touching them
        cpu_manager.touch(to_keys([3, 4]), _EMPTY_REQ_CTX)

        # now: T1 = {1, 2}, T2 = {3, 4}
        assert len(arc_policy.t1) == 2
        assert len(arc_policy.t2) == 2

        # set target_t1_size to prefer evicting from T1
        # (when |T1| >= target, evict from T1)
        arc_policy.target_t1_size = 1

        # store block 5, should evict from T1 (block 1, LRU in T1)
        output = cpu_manager.prepare_store(to_keys([5]), _EMPTY_REQ_CTX)
        assert output is not None
        assert to_keys([1]) == output.evicted_keys

        cpu_manager.complete_store(to_keys([5]), _EMPTY_REQ_CTX)

        # block 1 should be in B1 (ghost list)
        assert to_keys([1])[0] in arc_policy.b1
        # block 5 should be in T1
        assert to_keys([5])[0] in arc_policy.t1

    def test_ghost_list_bounds(self):
        """
        Tests that ghost lists (B1, B2) don't grow unbounded.
        They should be capped at cache_capacity.
        """
        cpu_manager, arc_policy = self._make_manager(num_blocks=2, enable_events=False)

        # fill cache with blocks 1, 2
        cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

        # store many blocks to fill ghost lists
        for i in range(3, 20):
            cpu_manager.prepare_store(to_keys([i]), _EMPTY_REQ_CTX)
            cpu_manager.complete_store(to_keys([i]), _EMPTY_REQ_CTX)

        # ghost lists should not exceed cache_capacity
        assert len(arc_policy.b1) <= arc_policy.cache_capacity
        assert len(arc_policy.b2) <= arc_policy.cache_capacity

    def test_touch_ordering(self):
        """
        Tests that touch() correctly updates access patterns.
        Similar to LRU test but verifies T1/T2 ordering.
        """
        cpu_manager, arc_policy = self._make_manager()

        # store blocks 1, 2, 3, 4
        cpu_manager.prepare_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # promote 3, 4 to T2
        cpu_manager.touch(to_keys([3, 4]), _EMPTY_REQ_CTX)

        # T1 = {1, 2}, T2 = {3, 4}
        # touch [1, 3, 4] - should promote 1 to T2, and move 3,4 to end of T2
        cpu_manager.touch(to_keys([1, 3, 4]), _EMPTY_REQ_CTX)

        # T1 = {2}, T2 = {1, 3, 4} (in that order, with 4 most recent)
        assert len(arc_policy.t1) == 1
        assert len(arc_policy.t2) == 3

        # store block 5, should evict from T1 (block 2, only one in T1)
        prepare_store_output = cpu_manager.prepare_store(to_keys([5]), _EMPTY_REQ_CTX)
        verify_store_output(
            prepare_store_output,
            ExpectedPrepareStoreOutput(
                keys_to_store=[5],
                store_block_ids=[1],  # reuses block 2's storage
                evicted_keys=[2],
            ),
        )

    def test_failed_store(self):
        """
        Tests that failed store operations clean up correctly.
        Similar to LRU test but for ARC.
        """
        cpu_manager, arc_policy = self._make_manager()

        # store blocks 1, 2, 3, 4
        cpu_manager.prepare_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # prepare store block 5 (will evict block 1)
        prepare_store_output = cpu_manager.prepare_store(to_keys([5]), _EMPTY_REQ_CTX)
        assert prepare_store_output is not None
        assert len(prepare_store_output.evicted_keys) == 1

        # complete store with failure
        cpu_manager.complete_store(to_keys([5]), _EMPTY_REQ_CTX, success=False)

        # block 5 should not be in cache
        assert cpu_manager.lookup(to_key(5), _EMPTY_REQ_CTX) is LookupResult.MISS
        # block 5 should not be in T1 or T2
        assert to_keys([5])[0] not in arc_policy.t1
        assert to_keys([5])[0] not in arc_policy.t2

        # evicted block should still be gone (in B1 ghost list)
        evicted_hash = prepare_store_output.evicted_keys[0]
        assert evicted_hash in arc_policy.b1

    def test_full_scenario(self):
        """
        Comprehensive test covering multiple ARC operations in sequence.
        Similar to the full LRU test but adapted for ARC behavior.
        """
        cpu_manager, arc_policy = self._make_manager()

        # store [1, 2]
        cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

        # store [3, 4, 5] -> evicts [1]
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([3, 4, 5]), _EMPTY_REQ_CTX
        )
        assert prepare_store_output is not None
        assert len(prepare_store_output.evicted_keys) == 1
        cpu_manager.complete_store(to_keys([3, 4, 5]), _EMPTY_REQ_CTX)

        # promote some blocks to T2
        cpu_manager.touch(to_keys([2, 3]), _EMPTY_REQ_CTX)

        # T1 has {4, 5}, T2 has {2, 3}
        assert len(arc_policy.t1) == 2
        assert len(arc_policy.t2) == 2

        # store [6] -> should evict from T1 (4 is oldest in T1)
        prepare_store_output = cpu_manager.prepare_store(to_keys([6]), _EMPTY_REQ_CTX)
        assert prepare_store_output is not None
        cpu_manager.complete_store(to_keys([6]), _EMPTY_REQ_CTX)

        # verify blocks 2, 3 (in T2) are still present
        assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.HIT
        assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is LookupResult.HIT

        # verify events
        events = list(cpu_manager.take_events())
        assert len(events) > 0  # should have store and eviction events


def test_filter_reused_manager():
    """
    Tests CPUOffloadingManager reuse filtering (store_threshold=2).
    """
    manager = make_cpu_manager(
        num_blocks=4,
        cache_policy="lru",
        enable_events=True,
        store_threshold=2,
        max_tracker_size=3,
    )

    # Lookup [1, 2] -> 1st time, added to tracker but not eligible for store yet
    assert manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.MISS
    assert manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.MISS

    # prepare store [1, 2] -> should be filtered
    prepare_store_output = manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == []

    # Lookup [1] -> 2nd time, eligible now
    assert manager.lookup(to_key(1), _EMPTY_REQ_CTX) is LookupResult.MISS

    # prepare store [1, 2] -> [1] should be eligible, [2] should be filtered
    prepare_store_output = manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == to_keys([1])

    # Lookup [3, 4] -> 1st time
    # (evicts [2] from tracker since max_size is 3 and tracker has [1])
    assert manager.lookup(to_key(3), _EMPTY_REQ_CTX) is LookupResult.MISS
    assert manager.lookup(to_key(4), _EMPTY_REQ_CTX) is LookupResult.MISS
    # Verify [2] was evicted from the tracker (tracker now has: [1], [3], [4])
    assert to_keys([2])[0] not in manager.counts

    # Lookup [2] again -> (this adds [2] back to the tracker as 1st time)
    assert manager.lookup(to_key(2), _EMPTY_REQ_CTX) is LookupResult.MISS
    # Verify [2] was re-added with count=1 (not eligible yet)
    assert manager.counts.get(to_keys([2])[0]) == 1

    # prepare store [2] -> should still be filtered out since count was reset
    prepare_store_output = manager.prepare_store(to_keys([2]), _EMPTY_REQ_CTX)
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == []

    manager.complete_store(to_keys([1]), _EMPTY_REQ_CTX)


def test_evictable_cache_block_count():
    """
    Verifies _num_evictable_cache_blocks is maintained correctly through the
    full store/load lifecycle, eviction, failed stores, concurrent loads,
    reset_cache, and the early-exit fast path in prepare_store.
    """
    manager = make_cpu_manager(num_blocks=4, cache_policy="lru")

    # Initially no blocks allocated.
    assert manager._num_evictable_cache_blocks == 0

    # Initial cache state [x, x, x, x]

    # We get 3 blocks from the cache.
    manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    # cache state [1', 2', 3', x] <- 1', 2', 3' are actively being used.
    assert manager._num_evictable_cache_blocks == 0

    # Completing stores makes them idle.
    manager.complete_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    # cache state [1, 2, 3, x] <- 1, 2, 3 blocks are idle.
    assert manager._num_evictable_cache_blocks == 3

    # prepare_load pins a block: idle count decrements once even if the
    # same block is loaded by two concurrent callers.
    manager.prepare_load(to_keys([1]), _EMPTY_REQ_CTX)
    # cache state [1', 2, 3, x] <- 2, 3 blocks are idle.
    assert manager._num_evictable_cache_blocks == 2
    manager.prepare_load(to_keys([1]), _EMPTY_REQ_CTX)  # 2nd concurrent load
    # cache state [1', 2, 3, x] <- 2, 3 blocks are idle.
    assert manager._num_evictable_cache_blocks == 2  # no double-decrement

    # First complete_load does not restore idle (ref_cnt still 1).
    manager.complete_load(to_keys([1]), _EMPTY_REQ_CTX)
    # cache state [1', 2, 3, x] <- 2, 3 blocks are idle.
    assert manager._num_evictable_cache_blocks == 2
    # Second complete_load drops ref_cnt to 0 -> block becomes idle again.
    manager.complete_load(to_keys([1]), _EMPTY_REQ_CTX)
    # cache state [1, 2, 3, x] <- 1, 2, 3 blocks are idle.
    assert manager._num_evictable_cache_blocks == 3

    # Eviction decrements idle count.
    # Cache has 3 stored blocks and 1 free slot. Storing 3 new keys needs 2 eviction.
    manager.prepare_store(to_keys([4, 5, 6]), _EMPTY_REQ_CTX)
    # cache state [1, 4', 5', 6'] <- block 1 is idle
    assert manager._num_evictable_cache_blocks == 1

    # Failed store does not increment idle count (block discarded from cache).
    manager.complete_store(to_keys([4, 5, 6]), _EMPTY_REQ_CTX, success=False)
    # cache state [1, x, x, x] <- block 1 is idle. Other returned to cache.
    assert manager._num_evictable_cache_blocks == 1

    # reset_cache zeroes the count unconditionally.
    manager.reset_cache()
    # cache state [x, x, x, x]
    assert manager._num_evictable_cache_blocks == 0

    # setup 3 blocks with loads so idle count drops to 0.
    manager.prepare_store(to_keys([10, 11, 12]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([10, 11, 12]), _EMPTY_REQ_CTX)
    manager.prepare_load(to_keys([10, 11, 12]), _EMPTY_REQ_CTX)
    # cache state [10', 11', 12', x]
    assert manager._num_evictable_cache_blocks == 0

    # prepare_store requiring eviction must return None immediately (fast exit).
    # Spy on policy.evict to confirm the fast path short-circuits before calling it.
    evict_called = False
    original_evict = manager._policy.evict

    def spy_evict(*args, **kwargs):
        nonlocal evict_called
        evict_called = True
        return original_evict(*args, **kwargs)

    manager._policy.evict = spy_evict  # type: ignore[method-assign]
    # cache state [10', 11', 12', x] <- cannot evict anything
    assert manager.prepare_store(to_keys([14, 15]), _EMPTY_REQ_CTX) is None
    assert not evict_called, (
        "_num_evictable_cache_blocks==0 should short-circuit before evict()"
    )

    # After releasing the loads, eviction becomes possible again.
    manager.complete_load(to_keys([10, 11, 12]), _EMPTY_REQ_CTX)
    # cache state [10, 11, 12, x] <- 10, 11, 12 are idle
    assert manager._num_evictable_cache_blocks == 3
    assert manager.prepare_store(to_keys([14, 15]), _EMPTY_REQ_CTX) is not None
    # cache state [10, 11, 14', 15'] <- 10, 11 are idle
    assert manager._num_evictable_cache_blocks == 2
    manager.complete_store(to_keys([14, 15]), _EMPTY_REQ_CTX)
    # cache state [10, 11, 14, 15] <- all blocks idle
    assert manager._num_evictable_cache_blocks == 4


def test_touch_forwards_req_context_to_policy(monkeypatch):
    """Regression: CPUOffloadingManager.touch forwards ReqContext to policy."""
    manager = make_cpu_manager(num_blocks=4, cache_policy="lru")
    received = []

    def spy_touch(keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        received.append((list(keys), req_context))

    monkeypatch.setattr(manager._policy, "touch", spy_touch)

    keys = to_keys([1, 2])
    ctx = make_req_context(
        req_id="test-req",
        kv_transfer_params={"test_param": "test_value"},
    )

    manager.touch(keys, ctx)

    assert len(received) == 1
    assert received[0][0] == keys
    assert received[0][1] is ctx


def test_prepare_store_mode_none_refuses_instead_of_evicting():
    """Speculative allocation must never displace demand-useful blocks."""
    manager = make_cpu_manager(num_blocks=2)
    resident = to_keys([1, 2])
    output = manager.prepare_store(resident, _EMPTY_REQ_CTX)
    assert output is not None
    manager.complete_store(resident, _EMPTY_REQ_CTX)

    # The tier is full but everything in it is evictable, so the default
    # path would succeed by evicting. The speculative path must not.
    assert (
        manager.prepare_store(to_keys([3]), _EMPTY_REQ_CTX, mode=AllocationMode.NONE)
        is None
    )

    for key in resident:
        assert manager.lookup(key, _EMPTY_REQ_CTX) is LookupResult.HIT


def test_prepare_store_mode_none_allocates_from_free_blocks():
    manager = make_cpu_manager(num_blocks=4)
    output = manager.prepare_store(
        to_keys([1, 2]), _EMPTY_REQ_CTX, mode=AllocationMode.NONE
    )

    assert output is not None
    assert list(output.keys_to_store) == to_keys([1, 2])
    assert output.evicted_keys == []


def test_prepare_store_mode_none_allocates_each_key_once():
    """One allocation per key: no reserve-then-consume double spend."""
    manager = make_cpu_manager(num_blocks=4)
    keys = to_keys([1, 2])
    output = manager.prepare_store(keys, _EMPTY_REQ_CTX, mode=AllocationMode.NONE)
    assert output is not None
    manager.complete_store(keys, _EMPTY_REQ_CTX)

    free_after = manager._get_num_free_blocks()
    # Re-preparing the same keys is a no-op: they are already stored.
    repeat = manager.prepare_store(keys, _EMPTY_REQ_CTX, mode=AllocationMode.NONE)
    assert repeat is not None
    assert list(repeat.keys_to_store) == []
    assert manager._get_num_free_blocks() == free_after


def test_prepare_store_mode_none_release_frees_blocks():
    """A cancelled speculative allocation returns its blocks to the pool."""
    manager = make_cpu_manager(num_blocks=4)
    keys = to_keys([1, 2])
    before = manager._get_num_free_blocks()
    output = manager.prepare_store(keys, _EMPTY_REQ_CTX, mode=AllocationMode.NONE)
    assert output is not None
    assert manager._get_num_free_blocks() < before

    manager.complete_store(keys, _EMPTY_REQ_CTX, success=False)

    assert manager._get_num_free_blocks() == before
    for key in keys:
        assert manager.lookup(key, _EMPTY_REQ_CTX) is LookupResult.MISS


@pytest.mark.parametrize("eviction_policy", ["lru", "arc"])
def test_demand_reclaims_speculative_blocks_first(eviction_policy):
    manager = make_cpu_manager(num_blocks=3, cache_policy=eviction_policy)
    demand = to_keys([1, 2])
    speculative = to_keys([3])
    output = manager.prepare_store(demand, _EMPTY_REQ_CTX)
    assert output is not None
    manager.complete_store(demand, _EMPTY_REQ_CTX)
    output = manager.prepare_store(
        speculative, _EMPTY_REQ_CTX, mode=AllocationMode.NONE
    )
    assert output is not None
    manager.complete_store(speculative, _EMPTY_REQ_CTX)
    manager.mark_speculative(speculative)

    output = manager.prepare_store(to_keys([4, 5]), _EMPTY_REQ_CTX)

    assert output is not None
    assert speculative[0] in output.evicted_keys
    assert (
        sum(manager.lookup(key, _EMPTY_REQ_CTX) is LookupResult.HIT for key in demand)
        == 1
    )


def test_speculative_allocation_succeeds_against_a_warm_full_cache():
    """The regression test for the zero-submission bug.

    A warm cache has no *free* blocks -- everything is allocated and merely
    evictable -- so a speculative allocation that only accepts free blocks is
    refused forever. The reserve is what keeps headroom available.
    """

    def warm(manager):
        """Fill the cache the way a benchmark warmup does."""
        for batch in ([1, 2], [3, 4], [5, 6]):
            output = manager.prepare_store(to_keys(batch), _EMPTY_REQ_CTX)
            assert output is not None
            manager.complete_store(output.keys_to_store, _EMPTY_REQ_CTX)

    # Without a reserve the pool ends up fully allocated and merely evictable,
    # which is exactly the state the first live benchmark ran in.
    without = make_cpu_manager(num_blocks=4)
    warm(without)
    assert without._get_num_free_blocks() == 0
    assert (
        without.prepare_store(
            to_keys([9]), _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
        )
        is None
    ), "expected the un-reserved pool to reproduce the zero-submission refusal"

    # With a reserve, demand recycles its own blocks instead of consuming the
    # headroom, so speculative promotion still has somewhere to land.
    with_reserve = make_cpu_manager(num_blocks=4, speculative_reserve_blocks=2)
    warm(with_reserve)
    output = with_reserve.prepare_store(
        to_keys([9]), _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
    )

    assert output is not None, "speculative allocation refused on a warm cache"
    assert list(output.keys_to_store) == to_keys([9])


def test_speculative_never_evicts_demand_blocks():
    """Exhausting the reserve refuses rather than touching demand data."""
    manager = make_cpu_manager(num_blocks=4, speculative_reserve_blocks=1)
    demand = to_keys([1, 2, 3])
    manager.complete_store(
        manager.prepare_store(demand, _EMPTY_REQ_CTX).keys_to_store, _EMPTY_REQ_CTX
    )

    first = manager.prepare_store(
        to_keys([9]), _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
    )
    assert first is not None
    # Reserve is now fully held by an in-flight speculative block, which
    # cannot be recycled, so the next request must be refused.
    assert (
        manager.prepare_store(
            to_keys([10]), _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
        )
        is None
    )
    for key in demand:
        assert manager.lookup(key, _EMPTY_REQ_CTX) is LookupResult.HIT


def test_speculative_recycles_its_own_oldest_once_the_reserve_is_full():
    manager = make_cpu_manager(num_blocks=4, speculative_reserve_blocks=1)
    manager.complete_store(
        manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX).keys_to_store,
        _EMPTY_REQ_CTX,
    )

    spec_a = to_keys([9])
    manager.complete_store(
        manager.prepare_store(
            spec_a, _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
        ).keys_to_store,
        _EMPTY_REQ_CTX,
    )
    # Now ready and reclaimable, so a second speculative key recycles it.
    spec_b = to_keys([10])
    assert (
        manager.prepare_store(
            spec_b, _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
        )
        is not None
    )
    assert manager.lookup(spec_a[0], _EMPTY_REQ_CTX) is LookupResult.MISS


def test_demanded_speculative_block_stops_counting_against_the_reserve():
    manager = make_cpu_manager(num_blocks=4, speculative_reserve_blocks=1)
    spec = to_keys([9])
    manager.complete_store(
        manager.prepare_store(
            spec, _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
        ).keys_to_store,
        _EMPTY_REQ_CTX,
    )
    assert manager._reserve_unused() == 0

    # Demand consumes it: it is ordinary demand data now, and the reserve
    # is free for the next promotion.
    manager.prepare_load(spec, _EMPTY_REQ_CTX)
    assert manager._reserve_unused() == 1


@pytest.mark.parametrize(
    "mode", [AllocationMode.DEMAND_CACHE, AllocationMode.DEMAND_CRITICAL]
)
def test_zero_reserve_leaves_demand_arithmetic_unchanged(mode):
    # Guards the reactive cells: with prefetch off, neither demand contract
    # may differ from the raw free count.
    plain = make_cpu_manager(num_blocks=4)
    assert plain._reserve_unused() == 0
    assert plain._num_allocatable_blocks(mode) == plain._get_num_free_blocks()


def test_critical_demand_is_not_starved_by_the_reserve():
    """A request waiting on this store outranks speculative headroom."""
    manager = make_cpu_manager(num_blocks=4, speculative_reserve_blocks=3)
    # Warm through the critical path: on a 4-block pool with 3 reserved, the
    # strict cache-store contract would (correctly) decline this.
    manager.complete_store(
        manager.prepare_store(
            to_keys([1, 2, 3, 4]),
            _EMPTY_REQ_CTX,
            mode=AllocationMode.DEMAND_CRITICAL,
        ).keys_to_store,
        _EMPTY_REQ_CTX,
    )
    # The reserve leaves demand only one nominally allocatable block, but a
    # critical store larger than that must still succeed by evicting its own.
    output = manager.prepare_store(
        to_keys([5, 6, 7]), _EMPTY_REQ_CTX, mode=AllocationMode.DEMAND_CRITICAL
    )
    assert output is not None
    assert len(output.keys_to_store) == 3


def _fill_with_demand(manager, start, batches, mode=None, complete=True, batch_size=10):
    """Drive demand stores.

    With complete=False the stores are left in flight (ref_cnt -1), which is
    what makes blocks unevictable and forces the allocator into its fallback --
    the warmup condition at high concurrency.
    """
    ctx = _EMPTY_REQ_CTX
    in_flight = []
    key_id = start
    kwargs = {"mode": mode} if mode is not None else {}
    for _ in range(batches):
        batch = to_keys(list(range(key_id, key_id + batch_size)))
        key_id += batch_size
        output = manager.prepare_store(batch, ctx, **kwargs)
        if output is None:
            break
        if complete:
            manager.complete_store(output.keys_to_store, ctx)
        else:
            in_flight.append(output.keys_to_store)
    return key_id, in_flight


def test_borrowed_reserve_is_restored_by_later_demand():
    """The regression test for the second zero-submission failure.

    A transient borrow used to be permanent: the eviction target was clamped
    to exactly what demand needed, so eviction freed N blocks and the
    allocation consumed all N. Free stayed pinned wherever it fell, and the
    live run reported reserve=64/reserve_free=64 while zero blocks were free
    and every speculative allocation was refused.
    """
    manager = make_cpu_manager(num_blocks=100, speculative_reserve_blocks=16)
    key_id, _ = _fill_with_demand(manager, 0, batches=30)
    assert manager._get_num_free_blocks() == 16, (
        "reserve should be held at steady state"
    )

    # Pin the cache with in-flight stores so eviction cannot find candidates,
    # forcing a critical allocation to borrow. This is warmup at concurrency 64.
    key_id, in_flight = _fill_with_demand(
        manager, key_id, batches=9, mode=AllocationMode.DEMAND_CRITICAL, complete=False
    )
    borrowed = manager._get_num_free_blocks()
    assert borrowed < 16, "expected the critical path to dip into the reserve"
    assert manager.reserve_blocks_borrowed_in_current_batch > 0

    # Release the pressure and run ordinary traffic: the watermark must return.
    for keys in in_flight:
        manager.complete_store(keys, _EMPTY_REQ_CTX)
    _fill_with_demand(manager, key_id, batches=40)

    assert manager._get_num_free_blocks() == 16, (
        "reserve must be restored by demand eviction, not stay borrowed forever"
    )
    assert (
        manager.prepare_store(
            to_keys([9001, 9002]),
            _EMPTY_REQ_CTX,
            mode=AllocationMode.SPECULATIVE_ONLY,
        )
        is not None
    ), "speculative allocation must succeed once the reserve is backed again"


def test_cache_store_declines_rather_than_spend_the_reserve():
    """GPU->CPU persistence is optional; a running request's load is not.

    The connector counts an allocation failure and continues, so a cache
    store must never consume speculative headroom. Letting it is what emptied
    the reserve during warmup.
    """
    strict = make_cpu_manager(num_blocks=100, speculative_reserve_blocks=16)
    key_id, _ = _fill_with_demand(strict, 0, batches=30)
    _fill_with_demand(
        strict,
        key_id,
        batches=9,
        mode=AllocationMode.DEMAND_CACHE,
        complete=False,
    )
    assert strict._get_num_free_blocks() == 16, (
        "cache stores must not touch the reserve"
    )
    assert strict.reserve_blocks_borrowed_in_current_batch == 0

    critical = make_cpu_manager(num_blocks=100, speculative_reserve_blocks=16)
    key_id, _ = _fill_with_demand(critical, 0, batches=30)
    _fill_with_demand(
        critical,
        key_id,
        batches=9,
        mode=AllocationMode.DEMAND_CRITICAL,
        complete=False,
    )
    assert critical._get_num_free_blocks() < 16, "critical demand may borrow"
    assert critical.reserve_blocks_borrowed_in_current_batch > 0


def test_reserve_borrow_counter_measures_blocks_not_events():
    """Blocks, not events: the magnitude is the actionable part.

    An event count says a borrow happened; blocks say whether it was a
    one-block dip or ate the whole prefetch budget.
    """
    manager = make_cpu_manager(num_blocks=100, speculative_reserve_blocks=16)
    key_id, _ = _fill_with_demand(manager, 0, batches=30)
    free_before = manager._get_num_free_blocks()
    _fill_with_demand(
        manager,
        key_id,
        batches=9,
        mode=AllocationMode.DEMAND_CRITICAL,
        complete=False,
    )
    consumed = free_before - manager._get_num_free_blocks()

    counters = manager.get_stats()._values
    borrowed = counters[CPUOffloadingMetrics.CPU_CACHE_RESERVE_BORROWED_BLOCKS][()]
    assert borrowed == consumed, (
        "counter must equal the reserved blocks actually consumed, "
        f"not an event count ({borrowed} vs {consumed} blocks)"
    )
    assert borrowed > 1, "a single event here consumed multiple blocks"
    assert manager.reserve_blocks_borrowed_in_current_batch == 0


def test_capacity_gauges_distinguish_fill_from_pinned_occupancy():
    """The gauge confusion that hid the zero-submission failure.

    cpu_cache_usage_perc counts only pinned blocks, so a physically full but
    idle cache reports near zero. The fill gauge is what actually answers
    "is there room?".
    """
    manager = make_cpu_manager(num_blocks=4)
    keys = to_keys([1, 2, 3, 4])
    manager.complete_store(
        manager.prepare_store(keys, _EMPTY_REQ_CTX).keys_to_store, _EMPTY_REQ_CTX
    )

    gauges = manager.get_stats()._values
    fill = gauges[CPUOffloadingMetrics.CPU_CACHE_FILL_PERC][()]
    pinned = gauges[CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC][()]

    assert fill == 1.0, "cache is physically full"
    assert pinned == 0.0, "nothing is pinned; this is the misleading reading"
    assert gauges[CPUOffloadingMetrics.CPU_CACHE_FREE_BLOCKS][()] == 0
    assert gauges[CPUOffloadingMetrics.CPU_CACHE_EVICTABLE_BLOCKS][()] == 4


def _promote_and_lease(manager, key_ids, ctx=_EMPTY_REQ_CTX):
    """Land a ready speculative bundle and retain it, as a promotion does."""
    keys = to_keys(key_ids)
    output = manager.prepare_store(keys, ctx, mode=AllocationMode.SPECULATIVE_ONLY)
    assert output is not None
    manager.complete_store(output.keys_to_store, ctx)
    manager.mark_speculative(keys)
    manager.lease_speculative(keys)
    return keys


def test_lease_retains_promoted_blocks_against_cache_persistence():
    """The retention regression: a promoted copy must survive to demand.

    Speculative blocks are the *preferred* eviction victim, so without a
    lease ordinary GPU->CPU persistence takes a freshly promoted block before
    the queued request arrives -- the mechanism behind 96.5% wasted
    promotions in the live run.
    """
    for lease_budget, expected_survivors in ((0, 0), (2, 2)):
        manager = make_cpu_manager(num_blocks=8, speculative_reserve_blocks=2)
        manager.set_lease_budget(lease_budget)
        key_id, _ = _fill_with_demand(manager, 0, batches=4, batch_size=2)
        promoted = _promote_and_lease(manager, [9000, 9001])

        _fill_with_demand(manager, key_id, batches=4, batch_size=2)

        survivors = sum(
            manager.lookup(key, _EMPTY_REQ_CTX) is LookupResult.HIT for key in promoted
        )
        assert survivors == expected_survivors, (
            f"lease_budget={lease_budget}: {survivors} of 2 promoted blocks "
            "survived ordinary persistence"
        )


def test_critical_demand_may_break_a_lease_and_records_it():
    """Retention never outranks a request that is running now."""
    manager = make_cpu_manager(num_blocks=8, speculative_reserve_blocks=2)
    manager.set_lease_budget(2)
    key_id, _ = _fill_with_demand(manager, 0, batches=4, batch_size=2)
    promoted = _promote_and_lease(manager, [9000, 9001])

    _fill_with_demand(
        manager, key_id, batches=4, batch_size=2, mode=AllocationMode.DEMAND_CRITICAL
    )

    assert all(
        manager.lookup(key, _EMPTY_REQ_CTX) is LookupResult.MISS for key in promoted
    ), "critical demand must be able to reclaim a lease"
    counters = manager.get_stats()._values
    assert (
        counters[CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_LEASE_RECLAIMED_BLOCKS][()]
        == 2
    ), "breaking a lease is demand pressure and must be reported as such"


def test_unleased_speculative_is_spent_before_a_leased_block():
    manager = make_cpu_manager(num_blocks=8, speculative_reserve_blocks=4)
    manager.set_lease_budget(1)
    key_id, _ = _fill_with_demand(manager, 0, batches=2, batch_size=2)

    leased = _promote_and_lease(manager, [9000])
    # A second promotion that is never leased: the budget of 1 is already used.
    unleased = to_keys([9001])
    output = manager.prepare_store(
        unleased, _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
    )
    manager.complete_store(output.keys_to_store, _EMPTY_REQ_CTX)
    manager.mark_speculative(unleased)
    assert list(manager._leased) == leased

    _fill_with_demand(manager, key_id, batches=3, batch_size=2)

    assert manager.lookup(leased[0], _EMPTY_REQ_CTX) is LookupResult.HIT
    assert manager.lookup(unleased[0], _EMPTY_REQ_CTX) is LookupResult.MISS


def test_lease_budget_releases_the_oldest_bundle_first():
    manager = make_cpu_manager(num_blocks=16, speculative_reserve_blocks=4)
    manager.set_lease_budget(2)
    first = _promote_and_lease(manager, [9000, 9001])
    assert list(manager._leased) == first

    second = _promote_and_lease(manager, [9002, 9003])
    assert list(manager._leased) == second, (
        "a newer bundle must displace the older lease, not accumulate"
    )


def test_demand_releases_the_lease():
    manager = make_cpu_manager(num_blocks=8, speculative_reserve_blocks=2)
    manager.set_lease_budget(2)
    promoted = _promote_and_lease(manager, [9000, 9001])

    manager.touch(promoted, _EMPTY_REQ_CTX)

    assert not manager._leased, "retention is done once demand has arrived"
    assert not manager._speculative, "a demanded block is ordinary data"


def test_lease_budget_is_clamped_to_the_reserve():
    manager = make_cpu_manager(num_blocks=64, speculative_reserve_blocks=4)
    assert manager.set_lease_budget(100) == 4, (
        "a lease wider than the reserve could never be honoured"
    )


def test_reset_clears_lease_state():
    manager = make_cpu_manager(num_blocks=8, speculative_reserve_blocks=2)
    manager.set_lease_budget(2)
    _promote_and_lease(manager, [9000, 9001])

    manager.reset_cache()

    assert not manager._leased
    assert not manager._speculative


def test_lease_gauge_tracks_retained_blocks():
    manager = make_cpu_manager(num_blocks=8, speculative_reserve_blocks=2)
    manager.set_lease_budget(2)
    assert (
        manager.get_stats()._values[
            CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_LEASED_BLOCKS
        ][()]
        == 0
    )

    _promote_and_lease(manager, [9000, 9001])

    assert (
        manager.get_stats()._values[
            CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_LEASED_BLOCKS
        ][()]
        == 2
    )


def test_speculative_reserve_gauges_track_ownership():
    manager = make_cpu_manager(num_blocks=4, speculative_reserve_blocks=2)
    gauges = manager.get_stats()._values
    assert gauges[CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_RESERVE_BLOCKS][()] == 2
    assert (
        gauges[CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_RESERVE_FREE_BLOCKS][()] == 2
    )

    manager.prepare_store(
        to_keys([9]), _EMPTY_REQ_CTX, mode=AllocationMode.SPECULATIVE_ONLY
    )
    gauges = manager.get_stats()._values
    assert gauges[CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_BLOCKS][()] == 1
    assert (
        gauges[CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_RESERVE_FREE_BLOCKS][()] == 1
    )
