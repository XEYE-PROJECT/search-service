"""Blobs de embeddings y modelos para embeber queries.

- Los embeddings de documentos llegan del backend como base64(np.save(float32 (n, d))) —
  el formato del worker de entrenamiento — y se decodifican, validan y L2-normalizan al cargar.
- La query se embebe en vivo con sentence-transformers; el NOMBRE del modelo es por lista
  y dinámico: sale del string opaco ``model`` del entrenamiento in_use (JSON codificado).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
import time
from collections import OrderedDict

import anyio
import numpy as np

logger = logging.getLogger(__name__)


def decode_embeddings(embeddings_b64: str | None, expected_rows: int) -> np.ndarray | None:
    """Blob .npy en base64 -> matriz float32, o None si no es usable.

    Las filas deben cuadrar con el número de elementos: la matriz refleja el lanzamiento
    del entrenamiento y la lista de elementos es la actual — con ediciones concurrentes
    divergen, y unos vectores mal asignados son peores que buscar solo por texto.
    """
    if not embeddings_b64:
        return None
    try:
        raw = base64.b64decode(embeddings_b64)
        matrix = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        logger.warning("Undecodable embeddings blob (%d chars); using text-only search", len(embeddings_b64))
        return None
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        logger.warning(
            "Embeddings shape %s does not match %d elements; using text-only search",
            getattr(matrix, "shape", None), expected_rows,
        )
        return None
    return np.ascontiguousarray(matrix, dtype=np.float32)


def parse_model_name(model_field: str | None) -> str | None:
    """Extrae el nombre del modelo de embedding del string opaco del entrenamiento."""
    if not model_field:
        return None
    try:
        parsed = json.loads(model_field)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("embedding_model")
    if not name or not isinstance(name, str) or name == "mock":
        return None
    return name


class ModelLoadError(Exception):
    pass


class ModelRegistry:
    """Carga y cachea modelos sentence-transformers por nombre (LRU, cupo pequeño).

    Lock por nombre: una carga lenta (disco o descarga de Hugging Face) no bloquea las
    queries de modelos ya residentes. Las cargas fallidas se cachean en negativo un rato —
    si no, una lista con un modelo inexistente reintentaría la descarga en cada búsqueda.
    El encode corre en hilos con lock por modelo (SentenceTransformer no es thread-safe).
    """

    _FAILURE_TTL_SECONDS = 300.0

    def __init__(self, default_model: str, max_loaded: int = 2) -> None:
        self._default_model = default_model
        self._max_loaded = max(1, max_loaded)
        self._models: OrderedDict[str, tuple[object, threading.Lock]] = OrderedDict()
        self._name_locks: dict[str, asyncio.Lock] = {}
        self._failed_at: dict[str, float] = {}

    def resolve_name(self, model_name: str | None) -> str:
        return model_name or self._default_model

    async def preload_default(self) -> None:
        try:
            await self._get(self._default_model)
        except Exception:
            logger.exception("Could not preload default embedding model %s", self._default_model)

    async def preload_models(self, names: list[str]) -> None:
        """Precalienta los modelos dados para que ninguna primera búsqueda pague la carga.

        Los ya residentes se saltan sin tocar su posición LRU, así un re-precalentamiento
        periódico nunca expulsa un modelo que las queries están usando. Los fallos son por
        modelo y no fatales — la búsqueda degrada a solo texto igualmente.
        """
        to_load = [n for n in dict.fromkeys(names) if n and n != "mock" and n not in self._models]
        if len(self._models) + len(to_load) > self._max_loaded:
            logger.warning(
                "Preloading %d embedding models with only %d cache slots (MODELS_MAX_LOADED); "
                "some will be evicted and reloaded on first use",
                len(self._models) + len(to_load), self._max_loaded,
            )
        for name in to_load:
            try:
                await self._get(name)
            except ModelLoadError:
                pass  # ya logueado en _get
            except Exception:
                logger.exception("Could not preload embedding model %s", name)

    async def embed_query(self, model_name: str | None, text: str) -> np.ndarray | None:
        """Vector float32 normalizado de la query, o None si el modelo no puede cargarse."""
        name = self.resolve_name(model_name)
        try:
            model, lock = await self._get(name)
        except ModelLoadError:
            return None  # ya logueado; la búsqueda degrada a solo texto
        except Exception:
            logger.exception("Could not load embedding model %s; using text-only search", name)
            return None

        def _encode() -> np.ndarray:
            with lock:
                vector = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm > 1e-12 else vector

        return await anyio.to_thread.run_sync(_encode)

    async def _get(self, name: str) -> tuple[object, threading.Lock]:
        entry = self._models.get(name)
        if entry is not None:
            self._models.move_to_end(name)
            return entry
        name_lock = self._name_locks.setdefault(name, asyncio.Lock())
        async with name_lock:
            entry = self._models.get(name)  # cargado mientras esperábamos
            if entry is not None:
                self._models.move_to_end(name)
                return entry
            failed_at = self._failed_at.get(name)
            if failed_at is not None and time.monotonic() - failed_at < self._FAILURE_TTL_SECONDS:
                raise ModelLoadError(name)

            def _load() -> object:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", name)
                try:
                    # Primero la caché: la imagen trae el modelo por defecto y un modelo
                    # cacheado debe cargar sin red. Ir online primero haría que
                    # sentence-transformers sondee el Hub y falle en seco sin DNS.
                    return SentenceTransformer(name, local_files_only=True)
                except Exception:
                    logger.info("Model %s is not cached; downloading it from the Hub", name)
                    return SentenceTransformer(name)

            try:
                model = await anyio.to_thread.run_sync(_load)
            except Exception as exc:
                self._failed_at[name] = time.monotonic()
                logger.error("Loading embedding model %s failed (retry in %.0fs): %s",
                             name, self._FAILURE_TTL_SECONDS, exc)
                raise ModelLoadError(name) from exc
            self._failed_at.pop(name, None)
            self._models[name] = (model, threading.Lock())
            while len(self._models) > self._max_loaded:
                evicted, _ = self._models.popitem(last=False)
                logger.info("Evicted embedding model %s (LRU)", evicted)
            return self._models[name]
