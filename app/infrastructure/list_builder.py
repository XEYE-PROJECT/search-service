"""Convierte el payload de una lista del backend en ``ListSearchData`` buscable en RAM.

El trabajo de CPU (decode base64+npy, normalización, FAISS) corre en un hilo, y cualquier
blob de embeddings inutilizable degrada a solo texto en vez de fallar la carga. Los
vectores se alinean POR ID: las filas siguen los ids del LANZAMIENTO del entrenamiento
(``trained_element_ids``) y ``elements`` es el conjunto ACTUAL — los vectores de elementos
borrados se descartan y los creados después quedan sin vector hasta el reentrenamiento.
Los payloads legacy sin ids caen a la vieja guarda por conteo de filas.
"""

from __future__ import annotations

import logging

import anyio
import numpy as np

from app.application.ports import ListDataPayload
from app.core.config import Settings
from app.domain.models import ListSearchData
from app.domain.normalization import normalize_text
from app.infrastructure.embeddings import decode_embeddings, parse_model_name
from app.infrastructure.vector_index import VectorIndex

logger = logging.getLogger(__name__)

_PER_ELEMENT_OVERHEAD = 200  # sobrecoste aproximado de dict/list/punteros por elemento


def _align_by_id(
    matrix: np.ndarray, trained_ids: list[int], element_ids: list[int | None]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(matriz_alineada, vector_rows, element_of_row) o None si nada casa."""
    row_by_id = {element_id: row for row, element_id in enumerate(trained_ids)}
    matched = [
        (i, element_id)
        for i, element_id in enumerate(element_ids)
        if element_id is not None and element_id in row_by_id
    ]
    if not matched:
        return None
    positions = [i for i, _ in matched]
    aligned = np.ascontiguousarray(matrix[[row_by_id[element_id] for _, element_id in matched]])
    vector_rows = np.full(len(element_ids), -1, dtype=np.int32)
    vector_rows[positions] = np.arange(len(positions), dtype=np.int32)
    return aligned, vector_rows, np.asarray(positions, dtype=np.int32)


def build_list_data_sync(payload: ListDataPayload, settings: Settings) -> ListSearchData:
    elements = payload.elements
    element_ids = [element.get("id") for element in elements]
    texts = [element.get("text") or "" for element in elements]
    params = [element.get("params") for element in elements]
    processed = [normalize_text(text) for text in texts]

    trained_ids = payload.trained_element_ids
    expected_rows = len(trained_ids) if trained_ids is not None else len(elements)
    matrix = decode_embeddings(payload.embeddings_data, expected_rows=expected_rows)

    index = None
    embeddings = None
    vector_rows = None
    element_of_row = None
    if matrix is not None and elements:
        if trained_ids is not None:
            alignment = _align_by_id(matrix, trained_ids, element_ids)
            if alignment is None:
                matrix = None
                logger.warning(
                    "No trained element survives in list %d; using text-only search",
                    payload.meta.id,
                )
            else:
                matrix, vector_rows, element_of_row = alignment
                if len(element_of_row) < len(elements):
                    logger.info(
                        "List %d: %d/%d elements have vectors (rest text-only until retrain)",
                        payload.meta.id,
                        len(element_of_row),
                        len(elements),
                    )
        else:  # payload legacy: la guarda por conteo ya aseguró filas == elementos
            vector_rows = np.arange(len(elements), dtype=np.int32)
            element_of_row = np.arange(len(elements), dtype=np.int32)
        if matrix is not None:
            index = VectorIndex(
                matrix,
                exact_threshold=settings.exact_search_max_elements,
                hnsw_m=settings.hnsw_m,
                hnsw_ef_construction=settings.hnsw_ef_construction,
                hnsw_ef_search=settings.hnsw_ef_search,
            )
            embeddings = index.matrix

    if index is None:
        vector_rows = None
        element_of_row = None

    text_bytes = sum(len(t) for t in texts) + sum(len(t) for t in processed)
    param_bytes = sum(len(p) for p in params if p)
    memory = (
        (index.memory_bytes if index is not None else 0)
        + 2 * text_bytes  # un str ocupa más que su número de caracteres
        + 2 * param_bytes
        + _PER_ELEMENT_OVERHEAD * len(elements)
    )

    data = ListSearchData(
        list_id=payload.meta.id,
        user_id=payload.meta.user_id,
        element_ids=element_ids,
        texts=texts,
        processed=processed,
        params=params,
        embeddings=embeddings,
        vector_rows=vector_rows,
        element_of_row=element_of_row,
        model_name=parse_model_name(payload.model),
        index=index,
        memory_bytes=memory,
    )
    logger.info(
        "Built search data for list %d: %d elements, embeddings=%s, model=%s, %.1f MiB",
        data.list_id,
        data.size,
        "yes" if index is not None else "no",
        data.model_name or "-",
        memory / 2**20,
    )
    return data


async def build_list_data(payload: ListDataPayload, settings: Settings) -> ListSearchData:
    return await anyio.to_thread.run_sync(build_list_data_sync, payload, settings)
