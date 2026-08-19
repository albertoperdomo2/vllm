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
    max_promotions_per_step: int = 64
    max_candidate_chunks: int = 1024
    speculative_max_bytes: int = 0
    # Derived by TieringOffloadingSpec; not accepted from user config.
    chunk_bytes: int = 0
    # Lead-time estimator constants (UNCALIBRATED).
    initial_admission_interval_ms: float = 50.0
    admission_interval_ewma_alpha: float = 0.2
    # L_prefetch(B) = transfer_base_ms + transfer_per_chunk_ms * |B|
    # (UNCALIBRATED).
    transfer_base_ms: float = 5.0
    transfer_per_chunk_ms: float = 2.0
    # U(B) = p_use * demand_load_per_chunk_ms * |B|
    #        - delta_q_active_ms - c_failure_ms
    # E[C_eviction] is zero by construction: allocation never evicts.
    p_use: float = 0.9
    demand_load_per_chunk_ms: float = 2.0
    delta_q_active_ms: float = 0.0
    c_failure_ms: float = 0.0

    _BOOL_FIELDS = frozenset({"enabled", "shadow_mode"})
    _POSITIVE_INT_FIELDS = frozenset(
        {"max_pending_bundles", "max_promotions_per_step", "max_candidate_chunks"}
    )
    _NON_NEGATIVE_INT_FIELDS = frozenset({"tier_idx", "speculative_max_bytes"})
    _FLOAT_FIELDS = frozenset(
        {
            "initial_admission_interval_ms",
            "admission_interval_ewma_alpha",
            "transfer_base_ms",
            "transfer_per_chunk_ms",
            "p_use",
            "demand_load_per_chunk_ms",
            "delta_q_active_ms",
            "c_failure_ms",
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
        p_use = kwargs.get("p_use", 0.9)
        if p_use > 1:
            raise ValueError("prefetch.p_use must be in [0, 1]")

        return cls(**kwargs)
