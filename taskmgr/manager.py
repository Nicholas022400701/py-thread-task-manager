"""Multi-threaded task manager.

Design:
- A bounded/unbounded PriorityQueue holds pending work items.
- A fixed pool of worker threads pulls items and executes them.
- Each task gets a unique monotonic id; results are stored in a dict
  protected by a lock, and an Event per task is used to signal completion.
- Retries re-enqueue the same task with a decremented attempt budget,
  optionally after a delay.
- Shutdown sends one sentinel per worker. `cancel_pending=True` drains the
  queue of unfinished tasks and marks them as CANCELLED before stopping.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class _QueueItem:
    priority: int
    seq: int
    task: "Task" = field(compare=False)


@dataclass
class Task:
    id: int
    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    priority: int = 5
    max_retries: int = 0
    retry_delay: float = 0.0
    attempts: int = 0


@dataclass
class TaskResult:
    id: int
    status: TaskStatus
    value: Any = None
    error: Optional[BaseException] = None
    traceback: Optional[str] = None
    attempts: int = 0


_SENTINEL = object()


class TaskManager:
    def __init__(self, num_workers: int = 4, name: str = "taskmgr") -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self._num_workers = num_workers
        self._queue: "queue.PriorityQueue[Any]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._id_seq = itertools.count(1)
        self._results: Dict[int, TaskResult] = {}
        self._events: Dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._pending = 0  # tasks submitted but not completed
        self._pending_cv = threading.Condition(self._lock)
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"{name}-worker-{i}",
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for t in self._workers:
            t.start()

    # -- public API --------------------------------------------------------
    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        priority: int = 5,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        **kwargs: Any,
    ) -> int:
        if self._shutdown.is_set():
            raise RuntimeError("TaskManager is shut down")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        task_id = next(self._id_seq)
        task = Task(
            id=task_id,
            fn=fn,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        with self._lock:
            self._events[task_id] = threading.Event()
            self._results[task_id] = TaskResult(id=task_id, status=TaskStatus.PENDING)
            self._pending += 1
        self._enqueue(task)
        return task_id

    def get_result(self, task_id: int, timeout: Optional[float] = None) -> TaskResult:
        with self._lock:
            ev = self._events.get(task_id)
        if ev is None:
            raise KeyError(f"unknown task id {task_id}")
        if not ev.wait(timeout):
            raise TimeoutError(f"task {task_id} did not finish within {timeout}s")
        with self._lock:
            return self._results[task_id]

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Block until all submitted tasks have completed. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._pending_cv:
            while self._pending > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._pending_cv.wait(timeout=remaining)
            return True

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        if self._shutdown.is_set():
            if wait:
                for t in self._workers:
                    t.join()
            return
        self._shutdown.set()
        if cancel_pending:
            self._drain_and_cancel()
        # Send one sentinel per worker with the highest priority so it exits
        # only after currently-queued work (for graceful) or immediately after
        # drain (for cancel_pending).
        sentinel_priority = -(2 ** 31) if cancel_pending else (2 ** 31)
        for _ in self._workers:
            self._queue.put(_QueueItem(sentinel_priority, next(self._seq), _SENTINEL))  # type: ignore[arg-type]
        if wait:
            for t in self._workers:
                t.join()

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "TaskManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True, cancel_pending=exc_type is not None)

    # -- internals --------------------------------------------------------
    def _enqueue(self, task: Task) -> None:
        self._queue.put(_QueueItem(task.priority, next(self._seq), task))

    def _drain_and_cancel(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item.task is _SENTINEL:
                # put sentinel back; we will add more later
                continue
            task: Task = item.task
            with self._lock:
                self._results[task.id] = TaskResult(
                    id=task.id,
                    status=TaskStatus.CANCELLED,
                    attempts=task.attempts,
                )
                ev = self._events.get(task.id)
                self._pending -= 1
                if self._pending == 0:
                    self._pending_cv.notify_all()
            if ev is not None:
                ev.set()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item.task is _SENTINEL:
                return
            task: Task = item.task
            task.attempts += 1
            with self._lock:
                self._results[task.id] = TaskResult(
                    id=task.id,
                    status=TaskStatus.RUNNING,
                    attempts=task.attempts,
                )
            try:
                value = task.fn(*task.args, **task.kwargs)
            except BaseException as exc:  # noqa: BLE001 - we record it
                if task.attempts <= task.max_retries:
                    if task.retry_delay > 0:
                        # Sleep is fine; retries are opt-in.
                        time.sleep(task.retry_delay)
                    self._enqueue(task)
                    continue
                tb_str = traceback.format_exc()
                with self._lock:
                    self._results[task.id] = TaskResult(
                        id=task.id,
                        status=TaskStatus.FAILED,
                        error=exc,
                        traceback=tb_str,
                        attempts=task.attempts,
                    )
                    ev = self._events.get(task.id)
                    self._pending -= 1
                    if self._pending == 0:
                        self._pending_cv.notify_all()
                if ev is not None:
                    ev.set()
            else:
                with self._lock:
                    self._results[task.id] = TaskResult(
                        id=task.id,
                        status=TaskStatus.SUCCESS,
                        value=value,
                        attempts=task.attempts,
                    )
                    ev = self._events.get(task.id)
                    self._pending -= 1
                    if self._pending == 0:
                        self._pending_cv.notify_all()
                if ev is not None:
                    ev.set()
