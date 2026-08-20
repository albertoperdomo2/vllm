# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: built-in "lru"/
    "arc", or the name of a policy registered via CachePolicyFactory, or an
    out-of-tree CachePolicy class name paired with cache_policy_module_path
    (default: "lru")
  - cache_policy_module_path: (optional) Python import path to load
    eviction_policy from when it names an out-of-tree CachePolicy not
    registered via CachePolicyFactory
  - prefetch_track_capacity: (optional) max prefetch-promoted keys tracked to
    attribute the useful/wasted outcome counters; 0 disables outcome tracking,
    in which case every successful promotion is counted as PREFETCH_UNTRACKED
    (default: 8192)
  - admission_prefetch_chunks: (optional) blind first-N proactive promotion at
    request admission; requires a secondary tier and a per-request
    kv_transfer_params.abc_admission_prefetch=true opt-in. Mutually exclusive
    with "prefetch" (default: 0, disabled)
  - prefetch: (optional) residency- and deadline-gated prefetch policy. Selects
    ordered contiguous prefix bundles at admission, verifies secondary
    residency, and promotes only when predicted lead time can hide the
    calibrated transfer time. Keys:
      - enabled: (default false)
      - policy / policy_module_path: policy name registered with
        PrefetchPolicyFactory ("admission"), or an out-of-tree class name
        paired with a module path
      - shadow_mode: (default TRUE) evaluate and log gate decisions without
        moving any data. Keep enabled until the transfer/lead-time constants
        below are calibrated for the deployment
      - speculative_reserve_blocks: CPU blocks held back from demand to give
        speculative promotion bounded headroom. One block holds one chunk.
        Best-effort, not guaranteed: demand borrows the unused remainder
        rather than fail a store, since a missed prefetch is far cheaper than
        a refused demand store. A warm cache has no free blocks -- everything
        is allocated
        and merely evictable -- so without a reserve every non-evicting
        speculative allocation is refused. null auto-derives from the other
        bounds; 0 disables promotion. Clamped to 25% of the pool.
      - max_bundle_chunks: per-bundle ceiling, and the depth of the residency
        probe. A run longer than this is trimmed
      - max_promotions_per_step: global promotion I/O budget per scheduler
        step. A bundle exceeding it submits across successive steps rather
        than losing the remainder
      - tier_idx, max_pending_bundles, max_candidate_chunks: further bounds
        on policy work
      - initial_admission_interval_ms, admission_interval_ewma_alpha,
        transfer_base_ms, transfer_per_chunk_ms, p_use,
        demand_load_per_chunk_ms, delta_q_active_ms, c_failure_ms:
        UNCALIBRATED lead-time and utility constants
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "example", "storage", "network")
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "example",
            "custom_param": 67
        }
    ]
}
"""

from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.base import TieringOffloadingMetrics
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    DEFAULT_PREFETCH_TRACK_CAPACITY,
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.prefetch.base import (
    build_admission_prefetch_metric_definitions,
)
from vllm.v1.kv_offload.tiering.prefetch.config import PrefetchConfig
from vllm.v1.kv_offload.tiering.prefetch.factory import PrefetchPolicyFactory

logger = init_logger(__name__)


class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The CPU primary tier has direct GPU access and serves as the gateway for
    all GPU↔offload operations. Secondary tiers cannot directly access GPU
    memory and must transfer data through the primary tier.
    """

    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        metrics = super().build_metric_definitions(extra_config)
        metrics[TieringOffloadingMetrics.LOOKUP_SYNC_DELAY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of total blocking time spent querying secondary "
                    "tiers for a request, accumulated from first lookup until "
                    "the request is allocated or finishes, in seconds."
                ),
                buckets=(
                    0.00001,
                    0.00005,
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                ),
            )
        )
        metrics[TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of wall-clock time from a request's first deferred "
                    "secondary-tier lookup until the request is allocated or "
                    "finishes, in seconds."
                ),
                buckets=(
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                    5,
                    10,
                ),
            )
        )
        metrics[TieringOffloadingMetrics.PREFETCH_ATTEMPTED] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of KV cache chunks selected for proactive promotion, "
                    "labeled by source tier."
                ),
                labelnames=("tier",),
            )
        )
        metrics[TieringOffloadingMetrics.PREFETCH_PROMOTED] = OffloadingCounterMetadata(
            documentation=(
                "Number of prefetch chunks that initiated a secondary->primary "
                "promotion, labeled by tier. Subset of PREFETCH_ATTEMPTED."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.PREFETCH_SKIPPED] = OffloadingCounterMetadata(
            documentation=(
                "Number of selected prefetch chunks that did not initiate a "
                "promotion, labeled by source tier. Subset of PREFETCH_ATTEMPTED."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.PREFETCH_REDUNDANT] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of prefetch chunks that were already resident in the "
                    "primary tier, so no promotion was initiated. Subset of "
                    "PREFETCH_ATTEMPTED, excluded from PREFETCH_PROMOTED."
                ),
                labelnames=("tier",),
            )
        )
        metrics[TieringOffloadingMetrics.PREFETCH_USEFUL] = OffloadingCounterMetadata(
            documentation=(
                "Number of prefetch-promoted chunks that a later demand lookup "
                "found in the primary tier, labeled by tier. Subset of "
                "PREFETCH_PROMOTED."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.PREFETCH_WASTED] = OffloadingCounterMetadata(
            documentation=(
                "Number of prefetch-promoted chunks dropped from the primary tier "
                "before any demand lookup reached them, labeled by tier. Subset "
                "of PREFETCH_PROMOTED; the complement of PREFETCH_USEFUL."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.PREFETCH_EVICTED_BEFORE_DEMAND] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of prefetch-promoted chunks that ordinary cache "
                    "persistence reclaimed before any demand lookup reached "
                    "them, labeled by tier. A strict SUBSET of "
                    "PREFETCH_WASTED, which still counts these -- this only "
                    "attributes one cause. It is NOT every pre-demand "
                    "eviction: a promotion displaced by a later speculative "
                    "promotion is counted as wasted but not here. Read it "
                    "against the other two causes -- speculative "
                    "displacement, visible as "
                    "cpu_cache_speculative_reserve_free_blocks reaching zero, "
                    "and a lead time too short for demand to arrive at all. "
                    "The three have different fixes: retention, a larger "
                    "reserve, and the admission horizon respectively."
                ),
                labelnames=("tier",),
            )
        )
        metrics[TieringOffloadingMetrics.PREFETCH_UNTRACKED] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of prefetch-promoted chunks whose outcome was never "
                    "determined, because prefetch_track_capacity was exceeded or "
                    "is 0 (tracking disabled), labeled by tier. Not an outcome: "
                    "these chunks may still have been used. Equal to "
                    "PREFETCH_PROMOTED minus PREFETCH_LOAD_FAILED means tracking "
                    "is off and the "
                    "useful/wasted ratio has no samples; a smaller non-trivial "
                    "value means it is under-sampled and the capacity should be "
                    "raised."
                ),
                labelnames=("tier",),
            )
        )
        metrics[TieringOffloadingMetrics.PREFETCH_LOAD_FAILED] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Proactively selected chunks in an asynchronous promotion job that "
                    "completed unsuccessfully."
                ),
                labelnames=("tier",),
            )
        )
        metrics[TieringOffloadingMetrics.PREFETCH_LATE] = OffloadingCounterMetadata(
            documentation=(
                "Proactively promoted chunks whose first demand lookup saw HIT_PENDING."
            ),
            labelnames=("tier",),
        )

        prefetch_config = extra_config.get("prefetch") or {}
        if not isinstance(prefetch_config, dict):
            raise ValueError("prefetch must be a mapping")
        metrics.update(build_admission_prefetch_metric_definitions(prefetch_config))

        secondary_tier_configs = extra_config.get("secondary_tiers", [])
        if not isinstance(secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        for tier_config in secondary_tier_configs:
            assert isinstance(tier_config, dict)
            tier_cls = SecondaryTierFactory.get_tier_class(tier_config)
            metrics.update(tier_cls.build_metric_definitions(tier_config))
        return metrics

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        # Redeclare for mypy: parent sets this but `--follow-imports skip` hides it
        self._manager: OffloadingManager | None = None

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        admission_prefetch_chunks = self.extra_config.get(
            "admission_prefetch_chunks", 0
        )

        if (
            not isinstance(admission_prefetch_chunks, int)
            or isinstance(admission_prefetch_chunks, bool)
            or admission_prefetch_chunks < 0
        ):
            raise ValueError(
                "admission_prefetch_chunks must be a non-negative int, got "
                f"{admission_prefetch_chunks!r}"
            )
        self.admission_prefetch_chunks = admission_prefetch_chunks

        # Keys tracked for prefetch outcome metrics (useful vs wasted)
        prefetch_track_capacity = self.extra_config.get(
            "prefetch_track_capacity", DEFAULT_PREFETCH_TRACK_CAPACITY
        )
        if not isinstance(prefetch_track_capacity, int) or prefetch_track_capacity < 0:
            raise ValueError(
                "prefetch_track_capacity must be a non-negative int, got "
                f"{prefetch_track_capacity!r}"
            )
        self.prefetch_track_capacity = prefetch_track_capacity

        self.prefetch_config = PrefetchConfig.from_extra_config(self.extra_config)
        if self.prefetch_config is not None and self.prefetch_config.enabled:
            if admission_prefetch_chunks > 0:
                raise ValueError(
                    "prefetch.enabled and admission_prefetch_chunks are "
                    "mutually exclusive: the V2 policy replaces the V1 "
                    "blind first-N selection"
                )
            if not self.secondary_tier_configs:
                raise ValueError(
                    "prefetch.enabled requires at least one secondary tier"
                )
            if self.prefetch_config.tier_idx >= len(self.secondary_tier_configs):
                raise ValueError(
                    f"prefetch.tier_idx {self.prefetch_config.tier_idx} is out of "
                    f"range for {len(self.secondary_tier_configs)} secondary tier(s)"
                )
            # Resolve now so an unknown policy name fails at startup.
            PrefetchPolicyFactory.get_prefetch_policy_cls(
                self.prefetch_config.policy,
                self.prefetch_config.policy_module_path,
            )

        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

        # engine_id is unique per DP replica (suffixed with _dp{rank} in both
        # the Ray and multiprocessing paths), so it names a per-replica offload
        # region.
        self._engine_id = config.engine_id

    @override
    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            # Create scheduler-side SharedOffloadRegion (rank=None) so the
            # primary tier can eagerly create a memoryview over _base.
            scheduler_mmap = SharedOffloadRegion(
                engine_id=self._engine_id,
                num_blocks=self.num_blocks,
                rank=None,
                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            self._scheduler_mmap = scheduler_mmap

            # Create primary tier (CPU-based)
            primary_tier = CPUPrimaryTierOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                mmap_region=scheduler_mmap,
            )

            # Create secondary tiers
            primary_kv_view = primary_tier.get_kv_memoryview()
            secondary_tiers = []
            for i, tier_config in enumerate(self.secondary_tier_configs):
                try:
                    tier = SecondaryTierFactory.create_secondary_tier(
                        tier_config, primary_kv_view, self
                    )
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier #%d (%s)",
                        i,
                        tier.tier_type,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create secondary tier from config index %i: %s",
                        i,
                        e,
                    )
                    raise

            # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
            # get_worker(). Secondary tier transfers are handled by the
            # secondary tier managers and need no additional workers here.
            prefetch_config = self.prefetch_config
            if prefetch_config is not None:
                prefetch_config = replace(
                    prefetch_config, chunk_bytes=self.kv_bytes_per_chunk
                )

            tiering_manager = TieringOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
                admission_prefetch_chunks=self.admission_prefetch_chunks,
                prefetch_track_capacity=self.prefetch_track_capacity,
                prefetch_config=prefetch_config,
            )
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError(
                    "store_threshold is not supported for TieringOffloadingSpec"
                )
            self._manager = tiering_manager

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    @override
    def _uses_shared_region(self) -> bool:
        # Tiering always allocates on the shared region (every platform), so the
        # replicated-layout gate must not be narrowed by the CPU spec's
        # CUDA-alike check.
        return True

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        world_size = self.config.parallel.world_size
        if self.replicated_layout:
            rank = 0
        else:
            # Fold the global physical device index into the replica-local
            # [0, world_size) slot range.
            rank = torch.accelerator.current_device_index() % world_size
        worker_mmap = SharedOffloadRegion(
            engine_id=self._engine_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            mmap_region=worker_mmap,
        )
