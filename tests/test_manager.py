import threading
import time

import pytest

from taskmgr import TaskManager, TaskStatus


def test_basic_submit_and_result():
    with TaskManager(num_workers=2) as tm:
        tid = tm.submit(lambda x, y: x + y, 2, 3)
        r = tm.get_result(tid, timeout=5)
        assert r.status == TaskStatus.SUCCESS
        assert r.value == 5
        assert r.attempts == 1


def test_failure_is_captured():
    def boom():
        raise ValueError("nope")

    with TaskManager(num_workers=2) as tm:
        tid = tm.submit(boom)
        r = tm.get_result(tid, timeout=5)
        assert r.status == TaskStatus.FAILED
        assert isinstance(r.error, ValueError)
        assert "nope" in (r.traceback or "")


def test_retry_eventually_succeeds():
    counter = {"n": 0}
    lock = threading.Lock()

    def flaky():
        with lock:
            counter["n"] += 1
            n = counter["n"]
        if n < 3:
            raise RuntimeError("try again")
        return "ok"

    with TaskManager(num_workers=1) as tm:
        tid = tm.submit(flaky, max_retries=5, retry_delay=0.0)
        r = tm.get_result(tid, timeout=5)
        assert r.status == TaskStatus.SUCCESS
        assert r.value == "ok"
        assert r.attempts == 3


def test_retry_exhausted():
    def always_fail():
        raise RuntimeError("x")

    with TaskManager(num_workers=1) as tm:
        tid = tm.submit(always_fail, max_retries=2)
        r = tm.get_result(tid, timeout=5)
        assert r.status == TaskStatus.FAILED
        assert r.attempts == 3


def test_priority_order():
    """With a single worker, lower-priority-value items execute first."""
    order = []
    olock = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    def gate():
        started.set()
        release.wait(timeout=5)

    def rec(name):
        with olock:
            order.append(name)

    with TaskManager(num_workers=1) as tm:
        # Occupy the single worker so the queue fills up while paused.
        tm.submit(gate, priority=0)
        assert started.wait(2)
        tm.submit(rec, "low", priority=10)
        tm.submit(rec, "mid", priority=5)
        tm.submit(rec, "high", priority=1)
        release.set()
        assert tm.wait_all(timeout=5)
    assert order == ["high", "mid", "low"]


def test_wait_all_timeout():
    with TaskManager(num_workers=1) as tm:
        tm.submit(time.sleep, 2)
        assert tm.wait_all(timeout=0.1) is False
        assert tm.wait_all(timeout=5) is True


def test_submit_after_shutdown_raises():
    tm = TaskManager(num_workers=1)
    tm.shutdown()
    with pytest.raises(RuntimeError):
        tm.submit(lambda: 1)


def test_cancel_pending_on_shutdown():
    started = threading.Event()
    release = threading.Event()

    def gate():
        started.set()
        release.wait(timeout=5)

    tm = TaskManager(num_workers=1)
    running_id = tm.submit(gate, priority=0)
    assert started.wait(2)
    pending_ids = [tm.submit(lambda: 1, priority=5) for _ in range(5)]
    release.set()
    tm.shutdown(wait=True, cancel_pending=True)
    statuses = [tm.get_result(i, timeout=2).status for i in pending_ids]
    # The running one must succeed; pending may be CANCELLED or SUCCESS depending
    # on exact ordering, but at least one should be CANCELLED.
    assert tm.get_result(running_id, timeout=2).status == TaskStatus.SUCCESS
    assert TaskStatus.CANCELLED in statuses


def test_invalid_num_workers():
    with pytest.raises(ValueError):
        TaskManager(num_workers=0)


def test_unknown_task_id():
    with TaskManager(num_workers=1) as tm:
        with pytest.raises(KeyError):
            tm.get_result(99999, timeout=0.1)


def test_get_result_timeout():
    with TaskManager(num_workers=1) as tm:
        tid = tm.submit(time.sleep, 2)
        with pytest.raises(TimeoutError):
            tm.get_result(tid, timeout=0.1)
