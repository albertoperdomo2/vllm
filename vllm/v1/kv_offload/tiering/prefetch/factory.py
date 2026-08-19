# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.prefetch.base import PrefetchPolicy

logger = init_logger(__name__)


class PrefetchPolicyFactory:
    """Registry for PrefetchPolicy implementations, resolved by name.

    Mirrors CachePolicyFactory (vllm/v1/kv_offload/cpu/policies/factory.py):
    built-in policies are pre-registered below. External policies can either
    register_prefetch_policy() a friendly short name up front, or skip
    registration entirely and pass a module path at lookup time (out-of-tree,
    no vLLM fork/patch required) -- see get_prefetch_policy_cls.
    """

    _registry: dict[str, Callable[[], type[PrefetchPolicy]]] = {}

    @classmethod
    def register_prefetch_policy(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a prefetch policy with a lazy-loading module/class name."""
        if name in cls._registry:
            raise ValueError(f"Prefetch policy '{name}' is already registered.")

        def loader() -> type[PrefetchPolicy]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_prefetch_policy_cls(
        cls, name: str, module_path: str | None = None
    ) -> type[PrefetchPolicy]:
        """Get a prefetch policy class by name.

        Args:
            name: Name of the prefetch policy. Checked against the registry
                first; if it's not registered and `module_path` is given,
                `name` is imported from there instead.
            module_path: Python import path to load `name` from when it is
                not a registered policy.

        Returns:
            The prefetch policy class.

        Raises ValueError if the policy is neither registered nor resolvable
        via `module_path`.
        """
        if name in cls._registry:
            return cls._registry[name]()
        if module_path is None:
            raise ValueError(
                f"Unknown prefetch policy: {name!r}. "
                f"Supported: {list(cls._registry)}. "
                "For an out-of-tree policy, also set prefetch.policy_module_path."
            )
        logger.warning(
            "Loading out-of-tree prefetch policy '%s' from '%s'. This API is "
            "experimental and subject to change in the future as we "
            "iterate the design.",
            name,
            module_path,
        )
        module = importlib.import_module(module_path)
        policy_cls = getattr(module, name)
        assert issubclass(policy_cls, PrefetchPolicy)
        return policy_cls


# Register built-in policies here.
PrefetchPolicyFactory.register_prefetch_policy(
    "admission",
    "vllm.v1.kv_offload.tiering.prefetch.admission",
    "AdmissionPrefetchPolicy",
)
