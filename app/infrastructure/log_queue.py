"""Envío asíncrono, por lotes y PERSISTENTE de los logs de búsqueda al backend.

Una búsqueda nunca espera a la persistencia del log: ``enqueue`` no bloquea y un único
consumidor agrupa entradas y las postea con reintentos. Lo que no se puede entregar no se
tira: un lote que agota sus reintentos, una cola llena o lo que quede al apagar va al spool
en disco (``LogSpool``) y un bucle de reenvío lo entrega cuando el backend vuelve. Sin spool
configurado (``LOG_SPOOL_DIR`` vacío) se conserva el comportamiento antiguo (descartar).
"""

from __future__ import annotations

import asyncio
import logging

from app.application.ports import BackendGateway, LogEntry
from app.infrastructure import metrics
from app.infrastructure.log_spool import LogSpool

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
        spool: LogSpool | None = None,
        replay_seconds: float = 30.0,
    ) -> None:
        self._backend = backend
        self._queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=queue_max)
        self._batch_max = batch_max
        self._flush_seconds = flush_seconds
        self._retries = retries
        self._spool = spool
        self._replay_seconds = replay_seconds
        self._overflow: list[LogEntry] = []  # entradas que no cupieron en la cola, camino del spool
        self.spooled = 0  # entradas en disco según el último recuento (para /health y métricas)

    def enqueue(self, entry: LogEntry) -> bool:
        """Nunca bloquea. Con la cola llena, la entrada va al spool en la siguiente pasada del
        consumidor (devuelve True); sin spool se descarta (False)."""
        try:
            self._queue.put_nowait(entry)
            return True
        except asyncio.QueueFull:
            if self._spool is None:
                metrics.LOG_DROPPED.inc()
                return False
            self._overflow.append(entry)
            return True

    @property
    def pending(self) -> int:
        return self._queue.qsize() + len(self._overflow)

    async def run(self) -> None:
        """Bucle consumidor; se detiene cancelando la tarea (o llamando a flush)."""
        while True:
            batch = await self._collect_batch()
            if self._overflow:
                await self._to_spool(self._overflow)
                self._overflow = []
            if batch:
                await self._deliver(batch)

    async def replay(self) -> None:
        """Bucle de reenvío del spool: cada ``replay_seconds`` intenta entregar lo guardado."""
        if self._spool is None:
            return
        while True:
            try:
                await self.replay_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Search-log spool replay failed: %s", exc)
            await asyncio.sleep(self._replay_seconds)

    async def replay_once(self) -> int:
        """Entrega ficheros del spool (el más antiguo primero) hasta vaciarlo o fallar. Devuelve
        cuántas entradas reenvió."""
        if self._spool is None:
            return 0
        sent = 0
        while True:
            item = await self._spool.oldest()
            if item is None:
                break
            path, entries = item
            try:
                await self._backend.push_logs(entries)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.info("Backend still unavailable for spooled search logs (%d waiting): %s", len(entries), exc)
                break
            await self._spool.discard(path)
            sent += len(entries)
        self.spooled = await self._spool.pending()
        if sent:
            logger.info("Replayed %d spooled search-log entries", sent)
        return sent

    async def flush(self) -> None:
        """Vacía lo encolado ahora mismo (apagado). Lo que no se entregue va al spool."""
        if self._overflow:
            await self._to_spool(self._overflow)
            self._overflow = []
        while not self._queue.empty():
            batch: list[LogEntry] = []
            while not self._queue.empty() and len(batch) < self._batch_max:
                batch.append(self._queue.get_nowait())
            await self._deliver(batch, attempts=1)

    async def _collect_batch(self) -> list[LogEntry]:
        batch: list[LogEntry] = [await self._queue.get()]
        deadline = asyncio.get_running_loop().time() + self._flush_seconds
        while len(batch) < self._batch_max:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except TimeoutError:
                break
        return batch

    async def _deliver(self, batch: list[LogEntry], attempts: int | None = None) -> None:
        max_attempts = max(1, self._retries if attempts is None else attempts)
        for attempt in range(1, max_attempts + 1):
            try:
                await self._backend.push_logs(batch)
                logger.debug("Pushed %d search logs to the backend", len(batch))
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt == max_attempts:
                    await self._to_spool(batch, reason=str(exc))
                    return
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))

    async def _to_spool(self, batch: list[LogEntry], reason: str = "queue overflow") -> None:
        if self._spool is None:
            metrics.LOG_DROPPED.inc(len(batch))
            logger.error("Dropping %d search logs (no spool configured): %s", len(batch), reason)
            return
        try:
            dropped = await self._spool.append(batch)
        except Exception as exc:
            metrics.LOG_DROPPED.inc(len(batch))
            logger.error("Dropping %d search logs: spool write failed (%s) after: %s", len(batch), exc, reason)
            return
        metrics.LOG_SPOOLED.inc(len(batch))
        if dropped:
            metrics.LOG_DROPPED.inc(dropped)
        self.spooled = await self._spool.pending()
        logger.warning("Spooled %d search logs to disk (%d waiting): %s", len(batch), self.spooled, reason)
