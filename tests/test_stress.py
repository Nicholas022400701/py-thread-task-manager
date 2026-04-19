"""Boundary + stress tests."""
import random
import threading
import time

import pytest

from taskmgr import TaskManager, TaskStatus


def test_many_tasks_cpu_light():
    N = 2000
    with TaskManager(num_workers=8) as tm:
        ids = [tm.submit(lambda x: x * x, i) for i in range(N)]
        assert tm.wait_all(timeout=30)
        results = [tm.get_result(i, timeout=5) for i in ids]
    assert all(r.status == TaskStatus.SUCCESS for r in results)
    assert sum(r.value for r in results) == sum(i * i for i in range(N))


def test_concurrent_submitters():
    """Many producer threads hammering submit() simultaneously."""
    N_PRODUCERS = 16
    PER_PRODUCER = 200
    total_expected = N_PRODUCERS * PER_PRODUCER

    with TaskManager(num_workers=8) as tm:
        submitted_ids = []
        id_lock = threading.Lock()

        def producer():
            local = []
            for _ in range(PER_PRODUCER):
                local.append(tm.submit(lambda x: x + 1, random.randint(0, 100)))
            with id_lock:
                submitted_ids.extend(local)

        threads = [threading.Thread(target=producer) for _ in range(N_PRODUCERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()

        assert len(submitted_ids) == total_expected
        assert len(set(submitted_ids)) == total_expected  # uniqueness under concurrency
        assert tm.wait_all(timeout=60)
        successes = sum(
            1 for i in submitted_ids if tm.get_result(i, timeout=5).status == TaskStatus.SUCCESS
        )
    assert successes == total_expected


def test_mixed_success_failure_retry():
    N = 500
    with TaskManager(num_workers=6) as tm:
        ids = []
        for i in range(N):
            if i % 7 == 0:
                ids.append(tm.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")), max_retries=1))
            else:
                ids.append(tm.submit(lambda x: x, i))
        assert tm.wait_all(timeout=30)
        results = [tm.get_result(i, timeout=5) for i in ids]
    failed = [r for r in results if r.status == TaskStatus.FAILED]
    succ = [r for r in results if r.status == TaskStatus.SUCCESS]
    assert len(failed) == sum(1 for i in range(N) if i % 7 == 0)
    assert len(succ) == N - len(failed)
    # Retries attempted twice for failures.
    assert all(r.attempts == 2 for r in failed)


def test_shutdown_idempotent_and_fast():
    tm = TaskManager(num_workers=4)
    for _ in range(50):
        tm.submit(lambda: 1)
    t0 = time.monotonic()
    tm.shutdown(wait=True)
    tm.shutdown(wait=True)  # second call should be a no-op
    elapsed = time.monotonic() - t0
    assert elapsed < 10


@pytest.mark.parametrize("workers", [1, 2, 4, 16])
def test_scales_across_worker_counts(workers):
    N = 300
    with TaskManager(num_workers=workers) as tm:
        ids = [tm.submit(lambda x: x * 2, i) for i in range(N)]
        assert tm.wait_all(timeout=30)
        total = sum(tm.get_result(i, timeout=5).value for i in ids)
    assert total == sum(i * 2 for i in range(N))


def test_invalid_retry_params():
    with TaskManager(num_workers=1) as tm:
        with pytest.raises(ValueError):
            tm.submit(lambda: 1, max_retries=-1)
        with pytest.raises(ValueError):
            tm.submit(lambda: 1, retry_delay=-0.5)
