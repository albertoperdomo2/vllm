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

    The two demand modes differ only in what they do when eviction cannot
    satisfy them, and that difference is what keeps the speculative reserve
    physically real:

    DEMAND_CACHE is GPU->CPU cache persistence. Refusing it is not free -- it
    sacrifices future reuse of that prefix -- but nothing currently running is
    waiting on it, so it declines rather than consume reserved blocks. This is
    the highest-volume caller; letting it borrow is how the reserve became an
    accounting fiction.

    DEMAND_CRITICAL is a reactive promotion serving a request that is running
    now. It may borrow the reserve as a last resort, because a missed prefetch
    is a far cheaper failure than a stalled request.

    The distinction is therefore urgency, not the value of the data.

    Both reclaim speculative blocks before evicting demand data through the
    cache policy. SPECULATIVE_ONLY may reclaim only blocks speculative work
    itself populated, so prefetch can never displace data the running workload
    owns. NONE takes free blocks or nothing.
    """

    DEMAND_CACHE = enum.auto()
    DEMAND_CRITICAL = enum.auto()
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
        # JIT mode assigns every speculative allocation to one request. The
        # active owner is released only when that request reaches demand,
        # finishes, expires, or fails; a newer request cannot replace it.
        self._speculative_owner_by_key: dict[OffloadKey, str] = {}
        self._active_speculative_owner: str | None = None
        # Ready speculative blocks held against ordinary cache persistence for
        # a bounded window, so demand gets a chance to consume them. Without
        # it a promoted copy is the *preferred* eviction victim and is gone
        # before the queued request arrives -- 96.5% of promotions were
        # wasted that way. A strict subset of _speculative, insertion-ordered
        # so the oldest lease is released first when the budget is exceeded.
        self._leased: OrderedDict[OffloadKey, None] = OrderedDict()
        self._lease_owner: str | None = None
        # Blocks the lease may cover. Zero disables retention entirely.
        self._lease_budget: int = 0
        # Blocks held back from demand so speculative work always has headroom.
        # Without it a warm cache leaves zero free blocks and every speculative
        # allocation is refused forever -- the zero-submission bug.
        self._speculative_reserve: int = max(0, speculative_reserve_blocks)

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []
        # Reserved blocks consumed by critical demand. Counted in blocks so
        # the rate distinguishes a harmless one-block dip from an erosion of
        # the whole prefetch budget; the free/reserve-free gauges show the
        # resulting shortfall but not who caused it.
        self.reserve_blocks_borrowed_in_current_batch: int = 0
        # Leased blocks broken by DEMAND_CRITICAL pressure. Distinct from a
        # prefetch outcome: retention was overridden, not mispredicted.
        self.lease_blocks_reclaimed_in_current_batch: int = 0

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

        For demand the result is deliberately allowed to go NEGATIVE when free
        blocks have fallen below the reserve. That negative value is what
        carries the shortfall into the caller's eviction target, making every
        demand allocation restore the watermark:

            evict(n - (free - reserve_unused)) then allocate(n)
                -> free == reserve_unused

        Clamping this at zero caps the target at exactly what demand needs, so
        eviction frees n blocks and the allocation immediately consumes all n.
        Free then stays pinned wherever it fell, permanently -- which is how a
        reserve of 64 came to report 64 unused blocks while zero were free.
        """
        raw = self._get_num_free_blocks()
        if mode is AllocationMode.SPECULATIVE_ONLY:
            return min(raw, self._reserve_unused())
        return raw - self._reserve_unused()

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

    def mark_speculative(
        self,
        keys: Collection[OffloadKey],
        owner_id: str | None = None,
    ) -> None:
        """Mark ready blocks as reclaimable speculative capacity."""
        if owner_id is not None and self._active_speculative_owner != owner_id:
            return
        for key in keys:
            block = self._policy.get(key)
            if block is not None and block.is_ready:
                self._speculative[key] = None
                if owner_id is not None:
                    self._speculative_owner_by_key[key] = owner_id

    def mark_demanded(self, keys: Collection[OffloadKey]) -> None:
        """Remove demand-touched blocks from speculative provenance.

        A demanded block stops counting against the reserve: it has proven
        itself useful, so it becomes ordinary demand-owned data. The lease is
        released first -- retention has served its purpose the moment demand
        arrives, and holding it longer would sequester capacity for nothing.
        """
        for key in keys:
            self._leased.pop(key, None)
            self._speculative.pop(key, None)
            self._speculative_owner_by_key.pop(key, None)

    # --- speculative retention lease ---

    def set_lease_budget(self, num_blocks: int) -> int:
        """Blocks the retention lease may cover; 0 disables retention.

        Bounded by the speculative reserve: a lease wider than the reserve
        could never be honoured, since the reserve is what caps speculative
        occupancy in the first place.

        Returns:
            The budget actually applied.
        """
        requested = max(0, num_blocks)
        if requested > self._speculative_reserve:
            logger.warning(
                "Speculative lease budget of %d blocks exceeds the %d-block "
                "speculative reserve; clamping to the reserve.",
                requested,
                self._speculative_reserve,
            )
        self._lease_budget = min(requested, self._speculative_reserve)
        self._enforce_lease_budget()
        return self._lease_budget

    def lease_speculative(
        self,
        keys: Collection[OffloadKey],
        owner_id: str | None = None,
    ) -> bool:
        """Retain ready speculative blocks against ordinary persistence.

        Only ready blocks that are still speculative are eligible: a block
        demand already claimed while the promotion was in flight is ordinary
        data and must not be re-marked.
        """
        if self._lease_budget <= 0:
            return False
        if owner_id is not None:
            if self._active_speculative_owner != owner_id:
                return False
            if self._lease_owner not in (None, owner_id):
                return False
            if self._lease_owner is None and self._leased:
                return False

        eligible: list[OffloadKey] = []
        for key in keys:
            if key not in self._speculative:
                continue
            if (
                owner_id is not None
                and self._speculative_owner_by_key.get(key) != owner_id
            ):
                continue
            block = self._policy.get(key)
            if block is not None and block.is_ready:
                eligible.append(key)

        if owner_id is not None:
            available = max(0, self._lease_budget - len(self._leased))
            eligible = eligible[:available]
            if eligible:
                self._lease_owner = owner_id

        for key in eligible:
            if key not in self._leased:
                self._leased[key] = None
        self._enforce_lease_budget()
        return bool(eligible)

    def release_speculative(self, keys: Collection[OffloadKey]) -> None:
        """Drop the lease, leaving the blocks ordinary speculative capacity."""
        for key in keys:
            self._leased.pop(key, None)
        if not self._leased:
            self._lease_owner = None

    def release_speculative_owner(
        self,
        owner_id: str,
        *,
        demanded: bool = False,
    ) -> tuple[OffloadKey, ...]:
        """Release one request's speculative ownership.

        Args:
            owner_id: Request that owns the speculative bundle.
            demanded: Convert the owner's blocks to ordinary demand-owned data.
                False leaves them unleased and reclaimable by later speculation.

        Returns:
            Keys whose ownership was released, including in-flight keys.
        """
        owned = tuple(
            key
            for key, key_owner in self._speculative_owner_by_key.items()
            if key_owner == owner_id
        )
        for key in owned:
            self._leased.pop(key, None)
            self._speculative_owner_by_key.pop(key, None)
            if demanded:
                self._speculative.pop(key, None)
        if self._lease_owner == owner_id:
            self._lease_owner = None
        if self._active_speculative_owner == owner_id:
            self._active_speculative_owner = None
        return owned

    def _enforce_lease_budget(self) -> None:
        """Release the oldest leases beyond the budget.

        Replacement is the expiry mechanism: a newer bundle displaces an older
        one rather than the lease growing without bound. Released blocks stay
        speculative, so they remain reclaimable -- just no longer protected.
        """
        while len(self._leased) > self._lease_budget:
            # A request-owned bundle keeps its earliest prefix blocks. Legacy
            # anonymous leases retain their historical replacement behavior.
            self._leased.popitem(last=self._lease_owner is not None)
        if not self._leased:
            self._lease_owner = None

    def _evict_for_demand(
        self,
        count: int,
        protected: set[OffloadKey],
        *,
        respect_lease: bool = True,
    ) -> tuple[
        list[tuple[OffloadKey, BlockStatus]],
        list[tuple[OffloadKey, BlockStatus]] | None,
    ]:
        """Select `count` victims for a demand allocation.

        Speculative blocks are reclaimed first so demand data survives longer.

        Args:
            respect_lease: skip leased speculative blocks. Demand allocations
                first pass True so ordinary and unleased victims are exhausted.
                A critical demand allocation may retry with False only after
                that protected attempt cannot make progress.

        Returns (reclaimed_speculative, normal_victims); a None second element
        means the request cannot be satisfied and nothing has been mutated.
        """
        if count > self._num_evictable_cache_blocks:
            # Eviction will fail.
            return [], None
        # There is a still a chance for eviction failure as some of the
        # idle blocks might be in the protected list.
        reclaimed = self._speculative_eviction_candidates(
            count, protected, respect_lease=respect_lease
        )
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
        self,
        n: int,
        protected: set[OffloadKey],
        *,
        respect_lease: bool = True,
    ) -> list[tuple[OffloadKey, BlockStatus]]:
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        if n <= 0:
            # The budget check below runs *after* appending, so without this
            # guard n=0 returns every eligible block instead of none. The
            # speculative path asks for 0 whenever it has room, which made
            # each promotion evict all previously promoted blocks.
            return candidates
        for key in self._speculative:
            if key in protected:
                continue
            if respect_lease and key in self._leased:
                # Unleased speculative blocks are spent first; a leased copy
                # is only a victim once its lease lapses.
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
        mode: AllocationMode = AllocationMode.DEMAND_CACHE,
        owner_id: str | None = None,
    ) -> PrepareStoreOutput | None:
        """Allocate primary blocks for the given keys.

        Args:
            mode: how far this caller may go to make room. See AllocationMode.
                Defaults to DEMAND_CACHE, the strict contract: an unqualified
                store is cache persistence, which declines rather than spend
                the speculative reserve. Speculative callers are bounded by
                the reserve and may reclaim only their own blocks, so prefetch
                cannot displace demand data.
        """
        speculative = mode is AllocationMode.SPECULATIVE_ONLY
        if owner_id is not None:
            if not speculative:
                raise ValueError("owner_id is only valid for speculative allocation")
            if self._active_speculative_owner not in (None, owner_id):
                return None
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
            # Speculative work displaces its own oldest, lease or not: the
            # lease protects a promoted copy from *demand persistence*, not
            # from the newer promotion that replaces it.
            reclaimed = self._speculative_eviction_candidates(
                needed, protected, respect_lease=False
            )
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
                    num_blocks_to_evict,
                    protected,
                    respect_lease=True,
                )
                if normal_evicted is None:
                    if mode is AllocationMode.DEMAND_CACHE:
                        # Nothing running is waiting on this store -- it buys
                        # future reuse, not progress on the current request --
                        # so it declines rather than spend reserved blocks.
                        # (Declining is not free: it sacrifices that reuse.
                        # It is simply cheaper than starving prefetch, which
                        # is what emptied the reserve during warmup.)
                        return None
                    # A request is waiting on this one, so holding the reserve
                    # back must never starve it: retry against raw free before
                    # refusing, borrowing reserved blocks if that is the only
                    # way. The watermark restores on later demand allocations.
                    free_before = self._get_num_free_blocks()
                    reserve_unused = self._reserve_unused()
                    shortfall = len(keys_to_store) - free_before
                    reclaimed, normal_evicted = (
                        # Only DEMAND_CRITICAL reaches here, and it has
                        # already been refused once -- the lease must not
                        # stand between a running request and progress.
                        self._evict_for_demand(
                            shortfall, protected, respect_lease=False
                        )
                        if shortfall > 0
                        else ([], [])
                    )
                    if normal_evicted is None:
                        return None
                    # Reserve actually consumed is the loss of *backed*
                    # headroom, min(free, reserve_unused), across this
                    # allocation -- not the whole reserve. A shortfall of zero
                    # or less evicts nothing and still spends free blocks.
                    free_after = (
                        free_before
                        + len(reclaimed)
                        + len(normal_evicted)
                        - len(keys_to_store)
                    )
                    self.reserve_blocks_borrowed_in_current_batch += min(
                        free_before, reserve_unused
                    ) - min(free_after, reserve_unused)
                evicted = reclaimed + normal_evicted

        to_evict: list[OffloadKey] = []
        if evicted:
            for key, _ in reclaimed:
                self._policy.remove(key)
                self._speculative.pop(key, None)
                self._speculative_owner_by_key.pop(key, None)
                # Membership, not the popped value: _leased maps to None, so
                # pop() returns None for a key that was present.
                if key in self._leased:
                    del self._leased[key]
                    if not speculative:
                        # Only DEMAND_CRITICAL reaches a leased block, since
                        # DEMAND_CACHE skips them. Counted separately because
                        # it is demand pressure breaking retention, not
                        # prefetch failing on its own terms.
                        self.lease_blocks_reclaimed_in_current_batch += 1

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
                if owner_id is not None:
                    self._speculative_owner_by_key[key] = owner_id
            if owner_id is not None:
                self._active_speculative_owner = owner_id

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
                    self._speculative_owner_by_key.pop(key, None)
                    self._leased.pop(key, None)
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
        self._speculative_owner_by_key.clear()
        self._active_speculative_owner = None
        self._leased.clear()
        self._lease_owner = None

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
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_LEASED_BLOCKS,
            len(self._leased),
        )
        if self.lease_blocks_reclaimed_in_current_batch:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_CACHE_SPECULATIVE_LEASE_RECLAIMED_BLOCKS,
                counter_increase_value=self.lease_blocks_reclaimed_in_current_batch,
            )
            self.lease_blocks_reclaimed_in_current_batch = 0
        if self.reserve_blocks_borrowed_in_current_batch:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_CACHE_RESERVE_BORROWED_BLOCKS,
                counter_increase_value=self.reserve_blocks_borrowed_in_current_batch,
            )
            self.reserve_blocks_borrowed_in_current_batch = 0

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
