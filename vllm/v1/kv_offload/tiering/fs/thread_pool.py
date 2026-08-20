# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool with demand-load, prefetch-load, and store queues.

Read threads prefer demand load, then prefetch load, then store. Write threads
prefer store, then demand load, then prefetch load. Speculative reads therefore
use otherwise idle read capacity without sitting ahead of reactive reads.
"""

import threading
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = ("_job_id", "_n_tasks", "_completed", "_success", "_lock")

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(self, success: bool) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += 1
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success


class DualQueueThreadPool:
    """
    Thread pool with demand-load, prefetch-load, and store queues.

    Both thread groups can drain every queue, so work can use all available
    threads. Demand reads always precede speculative reads.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._demand_load_q: deque = deque()
        self._prefetch_load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool]] = deque()
        self._inflight_jobs = 0  # guarded by _condition
        self._active_demand_load_tasks = 0

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
        *,
        is_prefetch: bool = False,
    ) -> None:
        """Enqueue demand or speculative load tasks for a job."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            self._inflight_jobs += 1
            queue = self._prefetch_load_q if is_prefetch else self._demand_load_q
            for fn in tasks:
                queue.append((fn, state))
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            self._inflight_jobs += 1
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[tuple[JobId, bool]]:
        # No lock needed: deque is thread-safe for concurrent append/popleft,
        # and the manager is the sole popper.
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        After this returns, every submitted job has had its last task
        finish, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def has_demand_load_work(self) -> bool:
        """Whether demand loads are queued or currently executing."""
        with self._condition:
            return bool(self._demand_load_q or self._active_demand_load_tasks > 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._demand_load_q.clear()
            self._prefetch_load_q.clear()
            self._store_q.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._stop
                        or self._demand_load_q
                        or self._prefetch_load_q
                        or self._store_q
                    )
                )
                if self._stop:
                    return
                queues = (
                    (self._demand_load_q, self._prefetch_load_q, self._store_q)
                    if load_priority
                    else (self._store_q, self._demand_load_q, self._prefetch_load_q)
                )
                selected = next(queue for queue in queues if queue)
                task, state = selected.popleft()
                is_demand_load = selected is self._demand_load_q
                if is_demand_load:
                    self._active_demand_load_tasks += 1
            try:
                task()
                job_finished, success = state.task_done(True)
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(False)

            with self._condition:
                if is_demand_load:
                    self._active_demand_load_tasks -= 1
                if job_finished:
                    self._finished_q.append((state.job_id, success))
                    self._inflight_jobs -= 1
                self._condition.notify_all()
