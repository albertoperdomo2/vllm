# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"
    # Physical occupancy and its composition. CPU_CACHE_USAGE_PERC above
    # measures only *pinned* blocks, so it reads near zero on a completely
    # full but idle cache -- which is why a warm pool with no free blocks
    # looked like it had 84% headroom in the first live benchmark.
    CPU_CACHE_FILL_PERC = "vllm:kv_offload_cpu_cache_fill_perc"
    CPU_CACHE_FREE_BLOCKS = "vllm:kv_offload_cpu_cache_free_blocks"
    CPU_CACHE_EVICTABLE_BLOCKS = "vllm:kv_offload_cpu_cache_evictable_blocks"
    CPU_CACHE_WRITE_PENDING_BLOCKS = "vllm:kv_offload_cpu_cache_write_pending_blocks"
    CPU_CACHE_SPECULATIVE_BLOCKS = "vllm:kv_offload_cpu_cache_speculative_blocks"
    CPU_CACHE_SPECULATIVE_RESERVE_BLOCKS = (
        "vllm:kv_offload_cpu_cache_speculative_reserve_blocks"
    )
    CPU_CACHE_SPECULATIVE_RESERVE_FREE_BLOCKS = (
        "vllm:kv_offload_cpu_cache_speculative_reserve_free_blocks"
    )
    # Reserved blocks consumed by critical demand. Counted in BLOCKS, not
    # events: an event count says a borrow happened, blocks say whether it was
    # harmless (one block) or ate the whole prefetch budget. The gauges above
    # show a shortfall but not its cause; the first bounded-reserve run had to
    # infer the borrow from free=0 while reserve_free=64.
    CPU_CACHE_RESERVE_BORROWED_BLOCKS = (
        "vllm:kv_offload_cpu_cache_reserve_borrowed_blocks"
    )
    # Retention lease: ready speculative blocks currently held against
    # ordinary cache persistence, and blocks whose lease demand-critical
    # pressure had to break.
    CPU_CACHE_SPECULATIVE_LEASED_BLOCKS = (
        "vllm:kv_offload_cpu_cache_speculative_leased_blocks"
    )
    CPU_CACHE_SPECULATIVE_LEASE_RECLAIMED_BLOCKS = (
        "vllm:kv_offload_cpu_cache_speculative_lease_reclaimed_blocks"
    )


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """
