"""Índice de similitud coseno sobre los embeddings de una lista.

La matriz se L2-normaliza al construir (coseno == producto interno). Dos regímenes:
n <= exact_threshold usa fuerza bruta exacta (numpy/BLAS — a esa escala es más rápida que
HNSW y con recall perfecto); por encima, FAISS IndexHNSWFlat preselecciona candidatos con
scores igualmente exactos. La matriz normalizada queda en RAM: el scorer híbrido también
necesita el coseno exacto de los candidatos por texto.
"""

from __future__ import annotations

import numpy as np

_ZERO_EPS = 1e-12


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < _ZERO_EPS] = 1.0
    return matrix / norms


class VectorIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        *,
        exact_threshold: int = 4096,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 96,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D matrix")
        self.matrix = l2_normalize(embeddings)
        self.size, self.dim = self.matrix.shape
        self._hnsw = None
        self._hnsw_bytes = 0
        if self.size > exact_threshold:
            import faiss  # import pesado, solo hace falta para listas grandes

            index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = hnsw_ef_construction
            index.hnsw.efSearch = hnsw_ef_search
            index.add(self.matrix)
            self._hnsw = index
            # IndexHNSWFlat guarda su propia copia de los vectores más ~2*M enlaces int64 por nodo.
            self._hnsw_bytes = self.size * self.dim * 4 + self.size * hnsw_m * 2 * 8

    @property
    def is_exact(self) -> bool:
        return self._hnsw is None

    @property
    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes) + self._hnsw_bytes

    def scores_for_all(self, query: np.ndarray) -> np.ndarray:
        """Coseno exacto de la query contra cada fila (recortado a [0, 1])."""
        scores = self.matrix @ np.asarray(query, dtype=np.float32)
        return np.clip(scores, 0.0, 1.0)

    def top_candidates(self, query: np.ndarray, k: int) -> dict[int, float]:
        """Filas candidatas con su coseno exacto: top-k del escaneo completo en régimen
        exacto; top-k aproximado (con scores exactos igualmente) en régimen HNSW."""
        k = min(k, self.size)
        if k <= 0:
            return {}
        query = np.asarray(query, dtype=np.float32)
        if self._hnsw is None:
            scores = self.scores_for_all(query)
            if k >= self.size:
                order = np.argsort(-scores)
            else:
                shortlist = np.argpartition(-scores, k - 1)[:k]
                order = shortlist[np.argsort(-scores[shortlist])]
            return {int(i): float(scores[i]) for i in order}
        distances, indices = self._hnsw.search(query.reshape(1, -1), k)
        return {int(i): float(np.clip(d, 0.0, 1.0)) for i, d in zip(indices[0], distances[0], strict=True) if i >= 0}

    def scores_for_rows(self, query: np.ndarray, rows: list[int]) -> dict[int, float]:
        """Coseno exacto de filas concretas (para los candidatos por texto)."""
        if not rows:
            return {}
        picked = self.matrix[rows] @ np.asarray(query, dtype=np.float32)
        picked = np.clip(picked, 0.0, 1.0)
        return {row: float(score) for row, score in zip(rows, picked, strict=True)}
