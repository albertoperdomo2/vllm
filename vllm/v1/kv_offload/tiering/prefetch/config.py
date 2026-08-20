# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class PrefetchConfig:
    """Configuration for the admission prefetch policy.

    Parsed from kv_connector_extra_config["prefetch"]. The estimator and
    gate constants are UNCALIBRATED placeholders until the V2.0
    characterization phase supplies measured values; shadow_mode therefore
    defaults to True so the policy logs decisions without moving data.
    """

    enabled: bool = False
    policy: str = "admission"
    policy_module_path: str | None = None
    shadow_mode: bool = True
    tier_idx: int = 0
    max_pending_bundles: int = 256
    # Global promotion I/O budget per scheduler step. Kept separate from the
    # per-bundle limit below: conflating them capped every bundle at 64 chunks
    # and discarded 94% of the verified-resident prefix in the first run.
    max_promotions_per_step: int = 256
    # Per-bundle ceiling, applied at admission: it bounds both the bundle and
    # the residency probe, and the candidate-window tail beyond it is counted
    # as BUNDLE_TRIM. A run that merely exceeds the remaining step budget is
    # not trimmed -- it is carried to later steps.
    max_bundle_chunks: int = 256
    # Frontier-scan depth. Residency probing is bounded by max_bundle_chunks,
    # since probing keys that could never be submitted only lengthens the
    # tier's lookup batch and pushes results past bundle deadlines.
    max_candidate_chunks: int = 1024
    # CPU blocks held back from demand to give speculative promotion bounded
    # headroom -- best-effort, since demand borrows the unused remainder
    # rather than fail a store. One block holds one chunk. None auto-derives
    # from the pool size; 0 disables prefetch allocation entirely.
    speculative_reserve_blocks: int | None = None
    # Derived by TieringOffloadingSpec; not accepted from user config.
    chunk_bytes: int = 0
    # Seed for the lead-time EWMA, replaced by observation within a few
    # scheduler steps. Default derived from Llama-3.1-Nemotron-Ultra-253B-FP8
    # at TP8/C64: 0.676 req/s is ~1.48 s per first-schedule event.
    initial_admission_interval_ms: float = 1450.0
    admission_interval_ewma_alpha: float = 0.2
    # Seeds for the measured transfer-cost model, used only until enough
    # real promotions have been observed. Defaults derived from the same
    # deployment: a 16-token chunk is 2 MiB at 128 KiB/token aggregate, and
    # local NVMe read averaged 921 MiB/s while active, so ~2.2 ms per chunk.
    # These are starting points, not a calibration -- the model refits from
    # whatever the deployment actually does.
    transfer_base_ms: float = 0.5
    transfer_per_chunk_ms: float = 2.2

    _BOOL_FIELDS = frozenset({"enabled", "shadow_mode"})
    _POSITIVE_INT_FIELDS = frozenset(
        {
            "max_pending_bundles",
            "max_promotions_per_step",
            "max_bundle_chunks",
            "max_candidate_chunks",
        }
    )
    _NON_NEGATIVE_INT_FIELDS = frozenset({"tier_idx"})
    _FLOAT_FIELDS = frozenset(
        {
            "initial_admission_interval_ms",
            "admission_interval_ewma_alpha",
            "transfer_base_ms",
            "transfer_per_chunk_ms",
        }
    )

    @classmethod
    def from_extra_config(
        cls, extra_config: Mapping[str, Any]
    ) -> "PrefetchConfig | None":
        """Parse and validate extra_config["prefetch"]; None if absent."""
        raw = extra_config.get("prefetch")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("prefetch must be a mapping")

        known = {f.name for f in fields(cls)} - {"chunk_bytes"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Unknown prefetch config keys: {sorted(unknown)}. "
                f"Supported: {sorted(known)}"
            )

        kwargs: dict[str, Any] = {}
        for name, value in raw.items():
            if name in cls._BOOL_FIELDS:
                if not isinstance(value, bool):
                    raise ValueError(f"prefetch.{name} must be a boolean")
            elif name in cls._POSITIVE_INT_FIELDS:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"prefetch.{name} must be an integer")
                if value < 1:
                    raise ValueError(f"prefetch.{name} must be >= 1")
            elif name in cls._NON_NEGATIVE_INT_FIELDS:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"prefetch.{name} must be an integer")
                if value < 0:
                    raise ValueError(f"prefetch.{name} must be >= 0")
            elif name in cls._FLOAT_FIELDS:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"prefetch.{name} must be a number")
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"prefetch.{name} must be finite and >= 0")
            elif name == "policy":
                if not isinstance(value, str) or not value:
                    raise ValueError("prefetch.policy must be a non-empty string")
            elif name == "speculative_reserve_blocks":
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError(f"prefetch.{name} must be an integer or null")
                    if value < 0:
                        raise ValueError(f"prefetch.{name} must be >= 0")
            elif (
                name == "policy_module_path"
                and value is not None
                and not isinstance(value, str)
            ):
                raise ValueError("prefetch.policy_module_path must be a string or null")
            kwargs[name] = value

        alpha = kwargs.get("admission_interval_ewma_alpha", 0.2)
        if not 0 < alpha <= 1:
            raise ValueError("prefetch.admission_interval_ewma_alpha must be in (0, 1]")

        config = cls(**kwargs)
        reserve = config.speculative_reserve_blocks
        if reserve is not None and 0 < reserve < config.max_bundle_chunks:
            raise ValueError(
                "prefetch.speculative_reserve_blocks must be >= max_bundle_chunks "
                f"({reserve} < {config.max_bundle_chunks}): a bundle larger than "
                "the reserve can never be fully resident"
            )
        return config
