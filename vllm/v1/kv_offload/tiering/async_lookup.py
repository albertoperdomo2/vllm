# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupManager: per-tier async lookup manager for secondary tier
existence checks.

Each secondary tier that wants non-blocking lookups composes its own
AsyncLookupManager instance internally.  The manager maintains lookup
state and uses a background thread to execute batch_lookup() calls.

Locking design
--------------
Scheduler-owned lookup state remains lock-free. Small locks protect only the
cross-thread activity counters and cancellation set:

* _lookup_state, _lookup_batch, and _prefetch_lookup_batch are owned
  exclusively by the scheduler thread. lookup(), flush(), and cleanup() read
  and write them directly.

* _lookup_queue is written by the scheduler (flush → put, at most one item per
  priority per step) and read by the background thread (get).

* _pending_results is written by the background thread (put) and read by
  the scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design.

* _work_lock protects queued/in-flight priority state. _cancel_lock protects
  keys cancelled after a batch has entered the priority queue. Neither lock is
  held across filesystem lookup I/O.

lookup() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from the tier's on_schedule_end(), posting
the entire batch as a single queue item so the background thread sees one
batch per step.
drain_results() is called before any lookup() calls in the same step, so
lookup() is a pure OrderedDict operation.
"""

import enum
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


@dataclass(slots=True)
class LookupState:
    result: bool | None = None  # True (found), False (not found), None
    request_ids: set[str] = field(default_factory=set)  # requests asking for the lookup
    priority: "LookupPriority" = field(default_factory=lambda: LookupPriority.DEMAND)
    # A speculative lookup that was already flushed may be re-enqueued at
    # demand priority when the running request reaches it. This flag prevents
    # repeated upgrades while the result remains pending.
    demand_upgrade_queued: bool = False


class LookupPriority(enum.IntEnum):
    """Lookup service class.

    Demand is always ordered before speculative prefetch. An in-progress
    filesystem call cannot be preempted, but demand can bypass every queued
    speculative batch.
    """

    DEMAND = 0
    PREFETCH = 1


class AsyncLookupManager(ABC):
    """
    Per-tier async lookup manager for secondary tier existence checks.

    Each secondary tier that wants non-blocking lookups composes its own
    AsyncLookupManager instance internally. The manager maintains lookup
    state (cache, queue) and uses a background thread to execute the actual
    batch_lookup() calls.

    Subclasses implement only batch_lookup() — all queue management,
    state tracking, and result delivery is provided by this base class.

    The owning tier delegates its lookup(), on_schedule_end(), and
    on_request_finished() to this manager:
      - lookup() → drain_results() + lookup state check
      - on_schedule_end() → flush()
      - on_request_finished() → cleanup()
    """

    def __init__(
        self,
        tier_type: str,
    ) -> None:
        self._tier_type = tier_type

        # key → LookupState; scheduler-owned, no lock needed.
        self._lookup_state: dict[OffloadKey, LookupState] = {}
        # req_id → keys looked up by that request (reverse index for cleanup).
        self._req_keys: dict[str, set[OffloadKey]] = {}

        # Accumulates (key, req_context) pairs during lookup() calls.
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext]] = []
        self._prefetch_lookup_batch: list[tuple[OffloadKey, ReqContext]] = []

        # Scheduler → worker: one full step's batch per item. PriorityQueue
        # makes a later demand batch bypass speculative batches that have not
        # started yet. The monotonic sequence makes equal-priority ordering
        # stable and keeps list payloads out of tuple comparison.
        self._lookup_queue: queue.PriorityQueue[
            tuple[int, int, list[tuple[OffloadKey, ReqContext]] | None]
        ] = queue.PriorityQueue()
        self._queue_seq = 0

        # Worker activity is observed by the scheduler-side prefetch gate.
        # The lock covers only small integer/state updates; lookup I/O never
        # runs while holding it.
        self._work_lock = threading.Lock()
        self._queued_batches = {
            LookupPriority.DEMAND: 0,
            LookupPriority.PREFETCH: 0,
        }
        self._inflight_priority: LookupPriority | None = None

        # cleanup() cannot remove an item already inside PriorityQueue. It marks
        # keys that no active request still owns, and the worker filters them
        # before issuing filesystem calls.
        self._cancel_lock = threading.Lock()
        self._cancelled_keys: set[OffloadKey] = set()

        # Worker → scheduler: completed result batches.
        # Each item is a list of (key, found) pairs.
        # SimpleQueue is explicitly thread-safe for one writer / one reader.
        self._pending_results: queue.SimpleQueue[list[tuple[OffloadKey, bool]]] = (
            queue.SimpleQueue()
        )
        self._need_to_drain: bool = False

        self._thread = threading.Thread(
            target=self._worker,
            name=f"vllm_offloading_lookup_{tier_type}",
            daemon=True,
        )
        self._thread.start()

    @abstractmethod
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        """
        Check whether a batch of blocks exist in this tier.

        Called from the worker thread — must be synchronous and must not
        touch the primary tier or scheduler state.

        Returns a list parallel to keys: True if present, False if not.
        """
        ...

    # ------------------------------------------------------------------
    # Scheduler-thread API
    # ------------------------------------------------------------------

    def lookup(
        self,
        key: OffloadKey,
        req_context: ReqContext,
        *,
        priority: LookupPriority = LookupPriority.DEMAND,
    ) -> bool | None:
        """
        Non-blocking lookup called from the scheduler thread.

        Returns:
            True  — block is present in this tier.
            False — block is not present in this tier.
            None  — result not yet available; retry next step.
        """
        if self._need_to_drain:
            self.drain_results()
            self._need_to_drain = False
        req_id = req_context.req_id
        state = self._lookup_state.get(key)
        if state is None:
            state = LookupState(priority=priority)
            self._lookup_state[key] = state
            with self._cancel_lock:
                self._cancelled_keys.discard(key)
            batch = (
                self._lookup_batch
                if priority is LookupPriority.DEMAND
                else self._prefetch_lookup_batch
            )
            batch.append((key, req_context))
        elif (
            priority is LookupPriority.DEMAND
            and state.result is None
            and state.priority is LookupPriority.PREFETCH
        ):
            # If the speculative item has not been flushed, move it rather than
            # duplicating it. If it is already queued/in flight, enqueue one
            # demand-priority copy so the running request does not sit behind
            # other speculative batches. Duplicate existence checks are safe.
            state.priority = LookupPriority.DEMAND
            moved = False
            for idx, (pending_key, _pending_ctx) in enumerate(
                self._prefetch_lookup_batch
            ):
                if pending_key == key:
                    self._prefetch_lookup_batch.pop(idx)
                    self._lookup_batch.append((key, req_context))
                    moved = True
                    break
            if not moved and not state.demand_upgrade_queued:
                self._lookup_batch.append((key, req_context))
                state.demand_upgrade_queued = True
        state.request_ids.add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        return state.result

    def lookup_prefetch(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """Low-priority lookup used only by speculative admission work."""
        return self.lookup(key, req_context, priority=LookupPriority.PREFETCH)

    def _enqueue_batch(
        self,
        priority: LookupPriority,
        batch: list[tuple[OffloadKey, ReqContext]],
    ) -> None:
        self._queue_seq += 1
        with self._work_lock:
            self._queued_batches[priority] += 1
        self._lookup_queue.put((int(priority), self._queue_seq, batch))

    def flush(self) -> None:
        """Post this step's accumulated keys to the worker thread.

        Called once per step from on_schedule_end() after all lookup() calls
        are done. The worker receives the full batch and processes it during
        the model-execution window, maximising time available before the next
        step's drain_results().  Safe to call with an empty batch (no-op).
        """
        self._need_to_drain = True
        if self._lookup_batch:
            self._enqueue_batch(LookupPriority.DEMAND, self._lookup_batch)
            self._lookup_batch = []
        if self._prefetch_lookup_batch:
            self._enqueue_batch(LookupPriority.PREFETCH, self._prefetch_lookup_batch)
            self._prefetch_lookup_batch = []

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called from lookup() before checking state.
        """
        while True:
            try:
                batch = self._pending_results.get_nowait()
            except queue.Empty:
                break
            for key, result in batch:
                state = self._lookup_state.get(key)
                if state is not None:
                    state.result = result

    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        cancelled: set[OffloadKey] = set()
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state[key]
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]
                cancelled.add(key)
        if cancelled:
            self._lookup_batch = [
                item for item in self._lookup_batch if item[0] not in cancelled
            ]
            self._prefetch_lookup_batch = [
                item for item in self._prefetch_lookup_batch if item[0] not in cancelled
            ]
            with self._cancel_lock:
                self._cancelled_keys.update(cancelled)

    def has_demand_work(self) -> bool:
        """Whether reactive lookups are queued or currently executing."""
        if self._lookup_batch:
            return True
        with self._work_lock:
            return (
                self._queued_batches[LookupPriority.DEMAND] > 0
                or self._inflight_priority is LookupPriority.DEMAND
            )

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._queue_seq += 1
        # Priority -1 makes shutdown bypass queued work.
        self._lookup_queue.put((-1, self._queue_seq, None))
        self._thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            priority_value, _seq, pending = self._lookup_queue.get()
            if pending is None:
                break

            priority = LookupPriority(priority_value)
            with self._work_lock:
                self._queued_batches[priority] -= 1
                self._inflight_priority = priority
            try:
                with self._cancel_lock:
                    cancelled = set(self._cancelled_keys)
                pending = [item for item in pending if item[0] not in cancelled]

                # Group by req_id.
                batches: dict[str, tuple[ReqContext, list[OffloadKey]]] = {}
                for key, req_context in pending:
                    req_id = req_context.req_id
                    if req_id not in batches:
                        batches[req_id] = (req_context, [])
                    batches[req_id][1].append(key)

                results: list[tuple[OffloadKey, bool]] = []
                for req_context, keys in batches.values():
                    try:
                        hits = self.batch_lookup(keys, req_context)
                        for key, hit in zip(keys, hits):
                            results.append((key, hit))
                    except Exception as exc:
                        logger.warning(
                            "batch_lookup failed on tier %s for %d keys: %s",
                            self._tier_type,
                            len(keys),
                            exc,
                        )
                        results.extend((key, False) for key in keys)

                if results:
                    self._pending_results.put(results)
            finally:
                with self._work_lock:
                    self._inflight_priority = None
