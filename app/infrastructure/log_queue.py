"""Envío asíncrono, por lotes y best-effort de los logs de búsqueda al backend.

Una búsqueda nunca espera a la persistencia del log: ``enqueue`` no bloquea (descarta con
aviso si la cola está llena) y un único consumidor agrupa entradas y las postea con
reintentos. Los lotes fallidos se descartan al agotarlos — los logs son telemetría, no
datos críticos de negocio.
"""

from __future__ import annotations

import asyncio
import logging

from app.application.ports import BackendGateway, LogEntry

logger = logging.getLogger(__name__)


class SearchLogQueue:
    def __init__(
        self,
        backend: BackendGateway,
        *,
        queue_max: int = 10_000,
        batch_max: int = 50,
        flush_seconds: float = 2.0,
        retries: int = 3,
    ) -> None:
        self._backend = backend
        self._queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=queue_max)
        self._batch_max = batch_max
        self._flush_seconds = flush_seconds
        self._retries = retries
        self._stopping = False

    def enqueue(self, entry: LogEntry) -> bool:
        try:
            self._queue.put_nowait(entry)
            return True
        except asyncio.QueueFull:
            return False

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def run(self) -> None:
        """Bucle consumidor; se detiene cancelando la tarea (o llamando a flush)."""
        while True:
            batch = await self._collect_batch()
            if batch:
                await self._deliver(batch)

    async def flush(self) -> None:
        """Vacía lo que haya encolado ahora mismo (se usa en el apagado)."""
        while not self._queue.empty():
            batch: list[LogEntry] = []
            while not self._queue.empty() and len(batch) < self._batch_max:
                batch.append(self._queue.get_nowait())
            await self._deliver(batch)

    async def _collect_batch(self) -> list[LogEntry]:
        batch: list[LogEntry] = [await self._queue.get()]
        deadline = asyncio.get_running_loop().time() + self._flush_seconds
        while len(batch) < self._batch_max:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return batch

    async def _deliver(self, batch: list[LogEntry]) -> None:
        for attempt in range(1, self._retries + 1):
            try:
                await self._backend.push_logs(batch)
                logger.debug("Pushed %d search logs to the backend", len(batch))
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt == self._retries:
                    logger.error(
                        "Dropping %d search logs after %d failed pushes: %s",
                        len(batch), attempt, exc,
                    )
                    return
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))
