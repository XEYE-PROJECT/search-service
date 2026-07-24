import asyncio

from app.application.ports import LogEntry
from app.infrastructure.log_queue import SearchLogQueue
from tests.conftest import FakeBackend


def entry(term: str = "q") -> LogEntry:
    return LogEntry(
        user_id=1, api_key_id=2, list_id=3, list_name="L", endpoint="/search",
        search_term=term, total_results=1, duration_ms=5, session=None,
        results={"item": 0.9}, searched_at="2026-07-11T10:00:00+00:00",
    )


async def test_batches_are_delivered():
    backend = FakeBackend()
    queue = SearchLogQueue(backend, batch_max=10, flush_seconds=0.05)
    consumer = asyncio.create_task(queue.run())
    for i in range(3):
        assert queue.enqueue(entry(f"q{i}"))
    await asyncio.sleep(0.3)
    consumer.cancel()
    assert [e.search_term for e in backend.pushed_logs] == ["q0", "q1", "q2"]


async def test_retries_then_succeeds():
    backend = FakeBackend()
    backend.fail_push_logs = 2  # fallan los dos primeros intentos, el tercero funciona
    queue = SearchLogQueue(backend, batch_max=10, flush_seconds=0.05, retries=3)
    consumer = asyncio.create_task(queue.run())
    queue.enqueue(entry())
    await asyncio.sleep(2.5)  # backoff 0.5 + 1.0 más margen
    consumer.cancel()
    assert len(backend.pushed_logs) == 1


async def test_drops_after_max_retries():
    backend = FakeBackend()
    backend.fail_push_logs = 99
    queue = SearchLogQueue(backend, batch_max=10, flush_seconds=0.05, retries=2)
    consumer = asyncio.create_task(queue.run())
    queue.enqueue(entry())
    await asyncio.sleep(1.5)
    consumer.cancel()
    assert backend.pushed_logs == []
    assert queue.pending == 0


async def test_flush_drains_queue():
    backend = FakeBackend()
    queue = SearchLogQueue(backend, batch_max=2, flush_seconds=10)
    for i in range(5):
        queue.enqueue(entry(f"q{i}"))
    await queue.flush()
    assert len(backend.pushed_logs) == 5


def test_enqueue_drops_when_full():
    backend = FakeBackend()
    queue = SearchLogQueue(backend, queue_max=2)
    assert queue.enqueue(entry())
    assert queue.enqueue(entry())
    assert not queue.enqueue(entry())
