# py-thread-task-manager

A lightweight multi-threaded task manager in Python.

## Features
- Thread pool with configurable worker count
- Priority queue (lower value = higher priority)
- Per-task retries with configurable delay
- Graceful shutdown (drain or cancel-pending)
- Result collection with blocking `get_result` and `wait_all`
- Thread-safe; usable as a context manager

## Quick start
```python
from taskmgr import TaskManager

with TaskManager(num_workers=4) as tm:
    tid = tm.submit(lambda x: x * 2, 21, priority=1)
    result = tm.get_result(tid, timeout=5)
    print(result.value)  # 42
```

## Dev
```bash
pip install -r requirements-dev.txt
pytest -q
```
