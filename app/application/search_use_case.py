"""Caso de uso de búsqueda: resolver la lista, tener sus datos en RAM, puntuar, responder, loguear.

Puntuación híbrida: fuzzy textual (rapidfuzz sobre texto normalizado) + semántica (coseno
sobre los embeddings entrenados, con la query embebida en vivo con el modelo de la lista).
Las listas pequeñas puntúan en exacto todos los elementos; las grandes usan dos fases:
top-K de FAISS HNSW ∪ top-K textual, todo con coseno exacto, y solo ese conjunto acotado
se rankea. Los elementos sin vector entrenado puntúan solo por texto.

Toda respuesta dice si se sirvió DEGRADADA (``degraded`` + ``degradation_reasons``): sin
embeddings entrenados, sin modelo de embedding disponible, modelo que no cuadra con los
vectores, o datos caducos que no se pudieron revalidar. El cliente puede decidir si se fía.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import anyio
import numpy as np
from rapidfuzz import fuzz, process

from app.application.catalog import CatalogService
from app.application.errors import ListNotFoundError, ListNotPublicError
from app.application.list_data import ListDataService
from app.application.ports import LogEntry, QueryEmbedder
from app.domain.models import ApiKeyInfo, ListMeta, ListSearchData, SearchHit
from app.domain.normalization import normalize_text
from app.domain.scoring import ScoringConfig, combine
from app.infrastructure import metrics

logger = logging.getLogger(__name__)

# Por encima de este tamaño la puntuación (texto + combinación + orden) pasa a un hilo
# para que una lista grande nunca bloquee el event loop.
_SCORING_THREAD_THRESHOLD = 2000
# Dimensionado del pool de candidatos de la ruta en dos fases (HNSW).
_MIN_CANDIDATES = 100
_CANDIDATE_FACTOR = 4

# Motivos de degradación (estables: los clientes pueden hacer switch sobre ellos).
NO_EMBEDDINGS = "no_embeddings"  # la lista no tiene vectores entrenados: solo texto
MODEL_UNAVAILABLE = "model_unavailable"  # el modelo de embedding no pudo cargarse: solo texto
MODEL_MISMATCH = "model_mismatch"  # la query se embebió con otra dimensión: solo texto
STALE_DATA = "stale_data"  # datos más viejos que el TTL y sin poder revalidar


def compute_text_scores(processed: list[str], query: str) -> np.ndarray:
    """Puntuación rapidfuzz combinada (0-1) de la query contra cada texto procesado."""
    if not query:
        return np.zeros(len(processed), dtype=np.float32)
    ratios = process.cdist([query], processed, scorer=fuzz.ratio, workers=1)[0]
    partials = process.cdist([query], processed, scorer=fuzz.partial_ratio, workers=1)[0]
    return (0.4 * ratios + 0.6 * partials).astype(np.float32) / 100.0


@dataclass
class _Semantic:
    """Puntuación semántica por elemento en una de dos formas (None = sin semántica).

    - exacta: ``row_scores`` puntúa cada fila de la matriz; la del elemento es
      ``row_scores[vector_rows[elemento]]`` (None si no tiene vector).
    - candidatos: ``by_element`` mapea un conjunto acotado de candidatos a puntuaciones
      (None = candidato solo textual sin vector); el resto se omite.
    """

    row_scores: np.ndarray | None = None
    by_element: dict[int, float | None] | None = None


@dataclass
class _Degradation:
    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)


class SearchUseCase:
    def __init__(
        self,
        catalog_service: CatalogService,
        list_data: ListDataService,
        embedder: QueryEmbedder,
        log_queue,  # SearchLogQueue (enqueue(entry) -> bool)
        scoring: ScoringConfig,
    ) -> None:
        self._catalogs = catalog_service
        self._list_data = list_data
        self._embedder = embedder
        self._log_queue = log_queue
        self._scoring = scoring

    async def resolve_public_list(self, user_id: int, list_name: str) -> ListMeta:
        """La lista del dueño de la key por nombre exacto; debe existir y ser PÚBLICA. La
        visibilidad la decide el propietario desde la consola, nunca el cliente de la API: las
        listas privadas solo se buscan desde la consola (``resolve_list_by_id``, vía backend)."""
        meta = self._catalogs.lists.resolve(user_id, list_name)
        if meta is None and await self._catalogs.refresh_on_miss():
            meta = self._catalogs.lists.resolve(user_id, list_name)
        if meta is None:
            raise ListNotFoundError(list_name)
        if not meta.is_public:
            raise ListNotPublicError(list_name)
        return meta

    async def resolve_list_by_id(self, list_id: int) -> ListMeta:
        """Cualquier lista por id, pública o privada: la usa la API interna para la consola,
        donde el backend ya ha comprobado la propiedad con la sesión del usuario."""
        meta = self._catalogs.lists.get(list_id)
        if meta is None and await self._catalogs.refresh_on_miss():
            meta = self._catalogs.lists.get(list_id)
        if meta is None:
            raise ListNotFoundError(list_id)
        return meta

    async def search(
        self,
        meta: ListMeta,
        auth: ApiKeyInfo | None,  # None = búsqueda desde la consola (sin API key)
        search_term: str,
        limit: int,
        *,
        include_breakdown: bool = False,
        session: str | None = None,
        register_log: bool = True,
    ) -> dict:
        """``register_log`` lo decide el SERVIDOR (la API pública siempre registra; la consola
        nunca): el antiguo campo del cliente se ignora."""
        started = time.perf_counter()
        data = await self._list_data.get_for_search(meta.id)
        if data is None:
            raise ListNotFoundError(meta.name)

        degradation = _Degradation()
        if self._list_data.is_serving_stale(meta.id):
            degradation.add(STALE_DATA)
        hits = await self._score(data, search_term, limit, degradation)
        duration_ms = int((time.perf_counter() - started) * 1000)

        results = []
        for hit in hits:
            entry: dict = {
                "item": hit.item,
                "score": round(hit.score, 2),
                "params": _parse_params(hit.params_raw),
            }
            if include_breakdown:
                entry["text_score"] = round(hit.text_score, 2)
                entry["semantic_score"] = round(hit.semantic_score, 2)
            results.append(entry)

        if register_log:
            self._enqueue_log(
                meta,
                auth,
                "/search",
                search_term,
                total_results=len(results),
                duration_ms=duration_ms,
                session=session,
                results={hit.item: round(hit.score, 2) for hit in hits},
            )

        degraded = bool(degradation.reasons)
        surface = "public" if auth is not None else "console"
        metrics.SEARCHES.labels(surface, "degraded" if degraded else "ok").inc()
        for reason in degradation.reasons:
            metrics.DEGRADED.labels(reason).inc()
        if degraded:
            logger.info("Search on list %d served degraded: %s", meta.id, ",".join(degradation.reasons))

        return {
            "success": True,
            "results": results,
            "total_results": len(results),
            "search_term": search_term,
            "list_name": meta.name,
            "duration_ms": duration_ms,
            "degraded": degraded,
            "degradation_reasons": list(degradation.reasons),
            "error": None,
        }

    async def target(self, meta: ListMeta, auth: ApiKeyInfo, target_term: str, session: str) -> dict:
        """Sin búsqueda: solo un log de auditoría de qué resultado eligió el usuario."""
        self._enqueue_log(
            meta,
            auth,
            "/target",
            target_term,
            total_results=0,
            duration_ms=0,
            session=session,
            results=None,
        )
        return {"success": True}

    async def _score(
        self, data: ListSearchData, search_term: str, limit: int, degradation: _Degradation
    ) -> list[SearchHit]:
        if data.size == 0:
            return []
        query_norm = normalize_text(search_term)
        big = data.size > _SCORING_THREAD_THRESHOLD

        if big:
            text_scores = await anyio.to_thread.run_sync(compute_text_scores, data.processed, query_norm)
        else:
            text_scores = compute_text_scores(data.processed, query_norm)

        semantic = await self._semantic_scores(data, search_term, text_scores, limit, degradation)

        if big:
            return await anyio.to_thread.run_sync(self._combine_hits, data, query_norm, text_scores, semantic, limit)
        return self._combine_hits(data, query_norm, text_scores, semantic, limit)

    def _combine_hits(
        self,
        data: ListSearchData,
        query_norm: str,
        text_scores: np.ndarray,
        semantic: _Semantic | None,
        limit: int,
    ) -> list[SearchHit]:
        candidates: Iterable[int]
        if semantic is not None and semantic.by_element is not None:
            candidates = sorted(semantic.by_element)
        else:
            candidates = range(data.size)

        hits: list[SearchHit] = []
        for i in candidates:
            text_score = float(text_scores[i])
            semantic_score = self._semantic_of(data, semantic, i)

            if data.processed[i] and data.processed[i] == query_norm:
                # Match normalizado exacto: puntuación máxima, pero a diferencia del
                # servicio original los demás resultados se conservan (por debajo).
                hits.append(SearchHit(data.texts[i], 1.0, data.params[i], 1.0, 1.0))
                continue

            final = combine(text_score, semantic_score, self._scoring)
            if final is None:
                continue
            hits.append(
                SearchHit(
                    data.texts[i],
                    float(min(max(final, 0.0), 1.0)),
                    data.params[i],
                    text_score,
                    semantic_score if semantic_score is not None else 0.0,
                )
            )

        hits.sort(key=lambda hit: -hit.score)
        return hits[:limit]

    @staticmethod
    def _semantic_of(data: ListSearchData, semantic: _Semantic | None, i: int) -> float | None:
        if semantic is None:
            return None
        if semantic.by_element is not None:
            return semantic.by_element.get(i)
        vector_rows, row_scores = data.vector_rows, semantic.row_scores
        if vector_rows is None or row_scores is None:  # sin vectores no hay puntuación semántica
            return None
        row = int(vector_rows[i])
        return float(row_scores[row]) if row >= 0 else None

    async def _semantic_scores(
        self,
        data: ListSearchData,
        search_term: str,
        text_scores: np.ndarray,
        limit: int,
        degradation: _Degradation,
    ) -> _Semantic | None:
        # list_builder garantiza que índice y alineación por id van juntos (los tres None o
        # los tres presentes); se comprueba aquí para que el tipo lo refleje.
        if data.index is None or data.vector_rows is None or data.element_of_row is None:
            degradation.add(NO_EMBEDDINGS)
            return None
        vector_rows, element_of_row = data.vector_rows, data.element_of_row
        query_vec = await self._embedder.embed_query(data.model_name, search_term)
        if query_vec is None:
            degradation.add(MODEL_UNAVAILABLE)
            return None
        if query_vec.shape[0] != data.index.dim:
            logger.warning(
                "Query embedding dim %d != list %d embeddings dim %d (model mismatch); text-only",
                query_vec.shape[0],
                data.list_id,
                data.index.dim,
            )
            degradation.add(MODEL_MISMATCH)
            return None

        if data.index.is_exact:
            return _Semantic(row_scores=data.index.scores_for_all(query_vec))

        # Dos fases: top-K semántico HNSW ∪ top-K textual — un conjunto ACOTADO (el fuzzy
        # da >0 a casi todo, así que "todo texto no nulo" no lo sería).
        k = max(limit * _CANDIDATE_FACTOR, _MIN_CANDIDATES)
        row_candidates = data.index.top_candidates(query_vec, k)
        by_element: dict[int, float | None] = {int(element_of_row[row]): score for row, score in row_candidates.items()}
        m = min(k, data.size)
        top_text = np.argpartition(-text_scores, m - 1)[:m]
        missing_rows: list[tuple[int, int]] = []
        for element in top_text:
            element = int(element)
            if text_scores[element] <= 0 or element in by_element:
                continue
            row = int(vector_rows[element])
            if row >= 0:
                missing_rows.append((element, row))
            else:
                by_element[element] = None  # aún sin vector: compite solo por texto
        exact_rows = data.index.scores_for_rows(query_vec, [row for _, row in missing_rows])
        for element, row in missing_rows:
            by_element[element] = exact_rows[row]
        return _Semantic(by_element=by_element)

    def _enqueue_log(
        self,
        meta: ListMeta,
        auth: ApiKeyInfo | None,
        endpoint: str,
        term: str,
        *,
        total_results: int,
        duration_ms: int,
        session: str | None,
        results: dict[str, float] | None,
    ) -> None:
        entry = LogEntry(
            user_id=meta.user_id,
            api_key_id=auth.id if auth is not None else None,
            list_id=meta.id,
            list_name=meta.name,
            endpoint=endpoint,
            search_term=term,
            total_results=total_results,
            duration_ms=duration_ms,
            session=session,
            results=results,
            searched_at=datetime.now(UTC).isoformat(),
        )
        if not self._log_queue.enqueue(entry):
            logger.warning("Search-log queue full and no spool; dropped a %s entry", endpoint)


def _parse_params(params_raw: str | None):
    if not params_raw:
        return None
    try:
        return json.loads(params_raw)
    except (TypeError, ValueError):
        return None
