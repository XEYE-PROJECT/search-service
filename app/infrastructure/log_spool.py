"""Spool en disco de los logs de búsqueda: lo que no se pudo entregar al backend no se pierde.

Cada lote fallido se escribe como un fichero JSONL (``<ts>-<uuid>.jsonl``, una entrada por
línea) en ``LOG_SPOOL_DIR``; el reenvío recorre los ficheros por antigüedad y borra cada uno
al entregarlo. Si el directorio supera ``LOG_SPOOL_MAX_BYTES`` se descartan los ficheros más
antiguos (con log ERROR): un backend caído durante días no debe llenar el disco. Todo el
E/S de fichero corre en un hilo para no bloquear el event loop.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import uuid
from pathlib import Path

import anyio

from app.application.ports import LogEntry

logger = logging.getLogger(__name__)


class LogSpool:
    def __init__(self, directory: str, max_bytes: int) -> None:
        self._dir = Path(directory)
        self._max_bytes = max_bytes

    @property
    def directory(self) -> Path:
        return self._dir

    # ---- escritura -------------------------------------------------------------

    async def append(self, entries: list[LogEntry]) -> int:
        """Guarda un lote; devuelve cuántas entradas se descartaron por el tope de tamaño."""
        return await anyio.to_thread.run_sync(self._append_sync, entries)

    def _append_sync(self, entries: list[LogEntry]) -> int:
        if not entries:
            return 0
        self._dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.jsonl"
        tmp = self._dir / (name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(dataclasses.asdict(entry), ensure_ascii=False) + "\n")
        os.replace(tmp, self._dir / name)  # atómico: nunca se reenvía un fichero a medias
        return self._enforce_limit_sync()

    def _enforce_limit_sync(self) -> int:
        files = self._files_sync()
        total = sum(size for _, size in files)
        dropped = 0
        while total > self._max_bytes and len(files) > 1:
            path, size = files.pop(0)
            dropped += _count_lines(path)
            path.unlink(missing_ok=True)
            total -= size
        if dropped:
            logger.error("Search-log spool over %d bytes; dropped %d oldest entries", self._max_bytes, dropped)
        return dropped

    # ---- lectura / reenvío -----------------------------------------------------

    async def oldest(self) -> tuple[Path, list[LogEntry]] | None:
        """El fichero más antiguo con sus entradas (ilegibles se saltan), o None si no hay nada."""
        return await anyio.to_thread.run_sync(self._oldest_sync)

    def _oldest_sync(self) -> tuple[Path, list[LogEntry]] | None:
        for path, _ in self._files_sync():
            entries: list[LogEntry] = []
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(_entry_from_dict(json.loads(line)))
                        except (ValueError, TypeError) as exc:
                            logger.warning("Skipping unreadable spooled log line in %s: %s", path.name, exc)
            except OSError as exc:
                logger.warning("Could not read spooled logs %s: %s", path.name, exc)
                path.unlink(missing_ok=True)
                continue
            if not entries:
                path.unlink(missing_ok=True)
                continue
            return path, entries
        return None

    async def discard(self, path: Path) -> None:
        await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))

    async def pending(self) -> int:
        """Entradas esperando en disco (cuenta líneas; solo para /health y métricas)."""
        return await anyio.to_thread.run_sync(self._pending_sync)

    def _pending_sync(self) -> int:
        return sum(_count_lines(path) for path, _ in self._files_sync())

    def _files_sync(self) -> list[tuple[Path, int]]:
        if not self._dir.is_dir():
            return []
        files: list[tuple[Path, int]] = []
        for path in sorted(self._dir.glob("*.jsonl")):
            try:
                files.append((path, path.stat().st_size))
            except OSError:
                continue
        return files


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _entry_from_dict(data: dict) -> LogEntry:
    fields = {f.name for f in dataclasses.fields(LogEntry)}
    return LogEntry(**{key: value for key, value in data.items() if key in fields})
