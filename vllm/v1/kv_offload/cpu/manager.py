# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from collections import OrderedDict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

logger = init_logger(__name__)


class AllocationMode(enum.Enum):
    """How far a caller may go to make room for its blocks.

    ANY is the demand path: reclaim speculative blocks first, then evict
    demand data through the cache policy. SPECULATIVE_ONLY may reclaim only
    blocks speculative work itself populated, so prefetch can never displace
    data the running workload owns. NONE takes free blocks or nothing.
    """

    ANY = enum.auto()
    SPECULATIVE_ONLY = enum.auto()
    NONE = enum.auto()


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        speculative_reserve_blocks: int = 0,
    ):
        self.medium: Medium = Medium.CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0
        # Blocks owned by speculative work, in allocation order. Populated at
        # ALLOCATION time, not completion: an in-flight promotion still
        # occupies a reserved block. Demand allocations reclaim these before
        # asking the configured policy to evict demand data, and every path
        # that ends speculative ownership pops from here, so this single
        # ledger cannot drift from reality.
        self._speculative: OrderedDict[OffloadKey, None] = OrderedDict()
        # Blocks held back from demand so speculative work always has headroom.
        # Without it a warm cache leaves zero free blocks and every speculative
        # allocation is refused forever -- the zero-submission bug.
        self._speculative_reserve: int = max(0, speculative_reserve_blocks)

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    # Ceiling on the speculative reserve as a fraction of the pool. Speculative
    # work is an optimization; letting it sequester a large share of the cache
    # would trade a certain loss of demand capacity for a possible gain.
    MAX_SPECULATIVE_RESERVE_FRACTION = 0.25

    def set_speculative_reserve(self, num_blocks: int) -> int:
        """Hold `num_blocks` back from demand for speculative promotion.

        Applied after config validation so a rejected prefetch configuration
        can never leave capacity sequestered.

        Returns:
            The reserve actually applied, which the pool-fraction ceiling may
            have reduced. Callers that sized a bundle against the requested
            value must re-check it against this one.
        """
        ceiling = int(self._num_blocks * self.MAX_SPECULATIVE_RESERVE_FRACTION)
        requested = max(0, num_blocks)
        if requested > ceiling:
            logger.warning(
                "Speculative prefetch reserve of %d blocks exceeds %.0f%% of the "
                "%d-block CPU pool; clamping to %d so demand capacity is not "
                "starved.",
                requested,
                self.MAX_SPECULATIVE_RESERVE_FRACTION * 100,
                self._num_blocks,
                ceiling,
            )
        self._speculative_reserve = min(requested, ceiling)
        return self._speculative_reserve

    def _reserve_unused(self) -> int:
        """Reserved blocks speculative work is not currently holding."""
        return max(0, self._speculative_reserve - len(self._speculative))

    def _num_allocatable_blocks(self, mode: "AllocationMode") -> int:
        """Free blocks this caller may take without reclaiming anything.

        Demand cannot see the unused part of the reserve, so it evicts its own
        LRU instead of consuming the headroom speculative work depends on.
        With a zero reserve both branches reduce to the raw free count, so the
        demand path is arithmetically unchanged when prefetch is off.
        """
        raw = self._get_num_free_blocks()
        if mode is AllocationMode.SPECULATIVE_ONLY:
            return min(raw, self._reserve_unused())
        return max(0, raw - self._reserve_unused())

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        self.mark_demanded(keys)
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self.mark_demanded(keys)
        self._policy.touch(keys, req_context)

    def mark_speculative(self, keys: Collection[OffloadKey]) -> None:
        """Mark ready blocks as reclaimable speculative capacity."""
        for key in keys:
            block = self._policy.get(key)
            if block is not None and block.is_ready:
                self._speculative[key] = None

    def mark_demanded(self, keys: Collection[OffloadKey]) -> None:
        """Remove demand-touched blocks from speculative provenance.

        A demanded block stops counting against the reserve: it has proven
        itself useful, so it becomes ordinary demand-owned data.
        """
        for key in keys:
            self._speculative.pop(key, None)

    def _evict_for_demand(
        self, count: int, protected: set[OffloadKey]
    ) -> tuple[
        list[tuple[OffloadKey, BlockStatus]],
        list[tuple[OffloadKey, BlockStatus]] | None,
    ]:
        """Select `count` victims for a demand allocation.

        Speculative blocks are reclaimed first so demand data survives longer.
        Returns (reclaimed_speculative, normal_victims); a None second element
        means the request cannot be satisfied and nothing has been mutated.
        """
        if count > self._num_evictable_cache_blocks:
            # Eviction will fail.
            return [], None
        # There is a still a chance for eviction failure as some of the
        # idle blocks might be in the protected list.
        reclaimed = self._speculative_eviction_candidates(count, protected)
        # Select normal victims atomically before removing speculative blocks.
        # Protect all speculative provenance so the cache policy cannot choose
        # a demand block in their place.
        normal = self._policy.evict(
            count - len(reclaimed), protected | set(self._speculative)
        )
        if normal is None:
            return [], None
        return reclaimed, normal

    def _speculative_eviction_candidates(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]]:
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key in self._speculative:
            if key in protected:
                continue
            block = self._policy.get(key)
            if block is not None and block.ref_cnt == 0:
                candidates.append((key, block))
                if len(candidates) == n:
                    break
        return candidates

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        mode: AllocationMode = AllocationMode.ANY,
    ) -> PrepareStoreOutput | None:
        """Allocate primary blocks for the given keys.

        Args:
            mode: how far this caller may go to make room. See AllocationMode.
                Speculative callers are bounded by the reserve and may reclaim
                only their own blocks, so prefetch cannot displace demand data.
        """
        speculative = mode is AllocationMode.SPECULATIVE_ONLY
        if not speculative:
            # A demand store claims ownership: these keys stop being
            # speculative and stop counting against the reserve.
            self.mark_demanded(keys)
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        # Blocks from the original input are excluded from eviction candidates:
        # a block that was already stored must remain in the cache after this call.
        protected = set(keys)

        if speculative:
            # Reclaim enough of our OWN oldest blocks to stay inside the
            # reserve and to find physical blocks. Speculative work never asks
            # the cache policy to evict, so demand data is untouchable.
            needed = max(
                0,
                len(self._speculative) + len(keys_to_store) - self._speculative_reserve,
                len(keys_to_store) - self._get_num_free_blocks(),
            )
            reclaimed = self._speculative_eviction_candidates(needed, protected)
            if len(reclaimed) < needed:
                return None
            evicted = reclaimed
        else:
            num_blocks_to_evict = len(keys_to_store) - self._num_allocatable_blocks(
                mode
            )
            if num_blocks_to_evict <= 0:
                reclaimed = []
                evicted = []
            elif mode is AllocationMode.NONE:
                return None
            else:
                reclaimed, normal_evicted = self._evict_for_demand(
                    num_blocks_to_evict, protected
                )
                if normal_evicted is None:
                    # Holding the reserve back must never starve demand: retry
                    # against raw free before refusing. A missed prefetch is a
                    # far cheaper failure than a refused demand store.
                    shortfall = len(keys_to_store) - self._get_num_free_blocks()
                    reclaimed, normal_evicted = (
                        self._evict_for_demand(shortfall, protected)
                        if shortfall > 0
                        else ([], [])
                    )
                    if normal_evicted is None:
                        return None
                evicted = reclaimed + normal_evicted

        to_evict: list[OffloadKey] = []
        if evicted:
            for key, _ in reclaimed:
                self._policy.remove(key)
                self._speculative.pop(key, None)

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        if speculative:
            # Claim ownership now, not on completion: an in-flight promotion
            # still occupies a reserved block, and mark_speculative() only
            # tags blocks once they are ready.
            for key in keys_to_store:
                self._speculative[key] = None

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._speculative.pop(key, None)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0
        self._speculative.clear()

        self._free_list.clear()
        self._num_allocated_blocks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        # Physical occupancy and its composition. usage above counts only
        # pinned blocks, so it cannot answer "is there room to allocate?".
        free_blocks = self._get_num_free_blocks()
        fill = (
            (self._num_blocks - free_blocks) / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_FILL_PERC, fill)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_FREE_BLOCKS, free_blocks)
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_EVICTABLE_BLOCKS,
            self._num_evictable_cache_blocks,
        )
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_WRITE_PENDING_BLOCKS,
            self._num_write_pending_blocks,
        )
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_BLOCKS, len(self._speculative)
        )
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_RESERVE_BLOCKS,
            self._speculative_reserve,
        )
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_RESERVE_FREE_BLOCKS,
            self._reserve_unused(),
        )

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
