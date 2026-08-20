# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLookupManager."""

import threading
import time
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager


def _key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def _ctx(req_id: str = "r1") -> ReqContext:
    return ReqContext(req_id=req_id)


class InMemoryLookupManager(AsyncLookupManager):
    """Test subclass backed by an in-memory set."""

    def __init__(self, existing_keys: set[OffloadKey] | None = None):
        super().__init__(tier_type="test")
        self._existing = existing_keys or set()
        self._results_ready = threading.Event()

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        results = [k in self._existing for k in keys]
        self._results_ready.set()
        return results


class ControlledLookupManager(AsyncLookupManager):
    """Lookup worker whose first speculative request can be held in flight."""

    def __init__(self):
        super().__init__(tier_type="controlled")
        self.release_blocker = threading.Event()
        self.called: dict[str, threading.Event] = {}
        self.call_order: list[str] = []
        self.call_lock = threading.Lock()

    def event(self, req_id: str) -> threading.Event:
        return self.called.setdefault(req_id, threading.Event())

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        with self.call_lock:
            self.call_order.append(req_context.req_id)
        self.event(req_context.req_id).set()
        if req_context.req_id == "blocker":
            self.release_blocker.wait(timeout=5.0)
        return [True] * len(keys)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TestAsyncLookupManager:
    def test_demand_bypasses_queued_prefetch_batch(self):
        mgr = ControlledLookupManager()
        try:
            mgr.lookup_prefetch(_key(1), _ctx("blocker"))
            mgr.flush()
            assert mgr.event("blocker").wait(timeout=5.0)

            mgr.lookup_prefetch(_key(2), _ctx("prefetch"))
            mgr.flush()
            mgr.lookup(_key(3), _ctx("demand"))
            mgr.flush()

            mgr.release_blocker.set()
            assert mgr.event("demand").wait(timeout=5.0)
            assert mgr.event("prefetch").wait(timeout=5.0)
            assert mgr.call_order[:3] == ["blocker", "demand", "prefetch"]
        finally:
            mgr.release_blocker.set()
            mgr.shutdown()

    def test_cleanup_skips_queued_prefetch_lookup(self):
        mgr = ControlledLookupManager()
        try:
            mgr.lookup_prefetch(_key(1), _ctx("blocker"))
            mgr.flush()
            assert mgr.event("blocker").wait(timeout=5.0)

            mgr.lookup_prefetch(_key(2), _ctx("cancelled"))
            mgr.flush()
            mgr.cleanup("cancelled")
            mgr.lookup(_key(3), _ctx("demand"))
            mgr.flush()

            mgr.release_blocker.set()
            assert mgr.event("demand").wait(timeout=5.0)

            def worker_idle():
                with mgr._work_lock:
                    return (
                        sum(mgr._queued_batches.values()) == 0
                        and mgr._inflight_priority is None
                    )

            assert _wait_until(worker_idle)
            assert "cancelled" not in mgr.call_order
        finally:
            mgr.release_blocker.set()
            mgr.shutdown()

    def test_new_key_returns_none(self):
        mgr = InMemoryLookupManager()
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.shutdown()

    def test_found_key_returns_true(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is True
        mgr.shutdown()

    def test_not_found_key_returns_false(self):
        mgr = InMemoryLookupManager(existing_keys=set())
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is False
        mgr.shutdown()

    def test_multiple_keys_single_step(self):
        existing = {_key(1), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()
        for i in range(1, 5):
            assert mgr.lookup(_key(i), ctx) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is False
        assert mgr.lookup(_key(3), ctx) is True
        assert mgr.lookup(_key(4), ctx) is False
        mgr.shutdown()

    def test_cleanup_removes_entries(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("req_a")
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True
        mgr.cleanup("req_a")
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_cleanup_preserves_shared_entries(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx_a = _ctx("req_a")
        ctx_b = _ctx("req_b")
        mgr.lookup(_key(1), ctx_a)
        mgr.lookup(_key(1), ctx_b)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        # Drain so result is applied
        mgr.lookup(_key(1), ctx_a)
        mgr.cleanup("req_a")
        # Key still present because req_b still references it
        assert _key(1) in mgr._lookup_state
        mgr.cleanup("req_b")
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_flush_no_queue_post_when_empty(self):
        mgr = InMemoryLookupManager()
        mgr.flush()
        assert mgr._lookup_queue.empty()
        mgr.shutdown()

    def test_repeated_lookup_same_key_no_duplicate_batch(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx()
        mgr.lookup(_key(1), ctx)
        mgr.lookup(_key(1), ctx)
        assert len(mgr._lookup_batch) == 1
        mgr.shutdown()

    def test_cleanup_unknown_req_id_is_noop(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("req_a")
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        mgr.lookup(_key(1), ctx)
        mgr.cleanup("nonexistent")
        assert _key(1) in mgr._lookup_state
        mgr.shutdown()

    def test_multiple_flushes_across_steps(self):
        existing = {_key(1), _key(2), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()

        # Step 1: lookup key 1, flush
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        # Step 2: lookup keys 2 and 3, flush
        mgr.lookup(_key(2), ctx)
        mgr.lookup(_key(3), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        # All results should be available
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is True
        assert mgr.lookup(_key(3), ctx) is True
        mgr.shutdown()

    def test_shutdown_unblocks_worker(self):
        mgr = InMemoryLookupManager()
        mgr.shutdown()
        assert not mgr._thread.is_alive()
