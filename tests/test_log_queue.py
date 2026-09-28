import asyncio

from app.application.ports import LogEntry
from app.infrastructure.log_queue import SearchLogQueue
from app.infrastructure.log_spool import LogSpool
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


async def test_drops_after_max_retries_without_spool():
    backend = FakeBackend()
    backend.fail_push_logs = 99
    queue = SearchLogQueue(backend, batch_max=10, flush_seconds=0.05, retries=2)
    consumer = asyncio.create_task(queue.run())
    queue.enqueue(entry())
    await asyncio.sleep(1.5)
    consumer.cancel()
    assert backend.pushed_logs == []
    assert queue.pending == 0


async def test_failed_batches_are_spooled_and_replayed(tmp_path):
    backend = FakeBackend()
    backend.fail_push_logs = 99
    spool = LogSpool(str(tmp_path / "spool"), max_bytes=10_000_000)
    queue = SearchLogQueue(backend, batch_max=10, flush_seconds=0.05, retries=1, spool=spool)
    consumer = asyncio.create_task(queue.run())
    queue.enqueue(entry("q0"))
    queue.enqueue(entry("q1"))
    await asyncio.sleep(0.4)
    consumer.cancel()
    assert backend.pushed_logs == []
    assert queue.spooled == 2  # nada se perdió: espera en disco
    assert len(list((tmp_path / "spool").glob("*.jsonl"))) == 1

    backend.fail_push_logs = 0  # el backend vuelve
    assert await queue.replay_once() == 2
    assert [e.search_term for e in backend.pushed_logs] == ["q0", "q1"]
    assert backend.pushed_logs[0].results == {"item": 0.9}  # ida y vuelta por JSON intacta
    assert queue.spooled == 0
    assert list((tmp_path / "spool").glob("*.jsonl")) == []


async def test_replay_stops_at_first_failure_and_keeps_files(tmp_path):
    backend = FakeBackend()
    spool = LogSpool(str(tmp_path), max_bytes=10_000_000)
    await spool.append([entry("a")])
    await spool.append([entry("b")])
    backend.fail_push_logs = 99
    queue = SearchLogQueue(backend, spool=spool)
    assert await queue.replay_once() == 0
    assert await spool.pending() == 2


async def test_spool_enforces_size_limit_by_dropping_oldest(tmp_path):
    spool = LogSpool(str(tmp_path), max_bytes=400)
    await spool.append([entry("old")])
    await spool.append([entry("mid")])
    dropped = await spool.append([entry("new")])  # ~200 bytes cada fichero: se descarta el más viejo
    assert dropped >= 1
    remaining = [e.search_term for _, batch in [await spool.oldest()] for e in batch]
    assert "old" not in remaining


async def test_flush_drains_queue():
    backend = FakeBackend()
    queue = SearchLogQueue(backend, batch_max=2, flush_seconds=10)
    for i in range(5):
        queue.enqueue(entry(f"q{i}"))
    await queue.flush()
    assert len(backend.pushed_logs) == 5


async def test_flush_spools_what_cannot_be_delivered(tmp_path):
    backend = FakeBackend()
    backend.fail_push_logs = 99
    spool = LogSpool(str(tmp_path), max_bytes=10_000_000)
    queue = SearchLogQueue(backend, batch_max=2, flush_seconds=10, spool=spool)
    for i in range(3):
        queue.enqueue(entry(f"q{i}"))
    await queue.flush()  # apagado con el backend caído: a disco, no a la basura
    assert queue.pending == 0
    assert await spool.pending() == 3


def test_enqueue_drops_when_full_without_spool():
    backend = FakeBackend()
    queue = SearchLogQueue(backend, queue_max=2)
    assert queue.enqueue(entry())
    assert queue.enqueue(entry())
    assert not queue.enqueue(entry())


async def test_enqueue_overflow_goes_to_spool(tmp_path):
    backend = FakeBackend()
    spool = LogSpool(str(tmp_path), max_bytes=10_000_000)
    queue = SearchLogQueue(backend, queue_max=1, batch_max=10, flush_seconds=0.05, spool=spool)
    assert queue.enqueue(entry("q0"))
    assert queue.enqueue(entry("q1"))  # no cabe en RAM: se acepta y va al spool
    assert queue.pending == 2
    consumer = asyncio.create_task(queue.run())
    await asyncio.sleep(0.3)
    consumer.cancel()
    assert [e.search_term for e in backend.pushed_logs] == ["q0"]
    assert await spool.pending() == 1
