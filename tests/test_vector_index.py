import numpy as np
import pytest

from app.infrastructure.vector_index import VectorIndex, l2_normalize


def random_matrix(n: int, d: int = 16) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(n, d)).astype(np.float32)


def manual_cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix_n = l2_normalize(matrix)
    query_n = query / np.linalg.norm(query)
    return np.clip(matrix_n @ query_n, 0.0, 1.0)


def test_exact_scores_match_manual_cosine():
    matrix = random_matrix(50)
    index = VectorIndex(matrix, exact_threshold=100)
    assert index.is_exact
    query = random_matrix(1)[0]
    query = query / np.linalg.norm(query)
    np.testing.assert_allclose(index.scores_for_all(query), manual_cosine(matrix, query), atol=1e-5)


def test_exact_top_candidates_ordering():
    matrix = random_matrix(50)
    index = VectorIndex(matrix, exact_threshold=100)
    query = l2_normalize(random_matrix(1))[0]
    candidates = index.top_candidates(query, 5)
    assert len(candidates) == 5
    scores = index.scores_for_all(query)
    expected_top = set(np.argsort(-scores)[:5].tolist())
    assert set(candidates) == expected_top


def test_hnsw_used_above_threshold_and_finds_neighbours():
    faiss = pytest.importorskip("faiss")  # noqa: F841
    matrix = random_matrix(300, d=24)
    index = VectorIndex(matrix, exact_threshold=100, hnsw_m=16)
    assert not index.is_exact
    # Query = una fila existente: HNSW debe encontrarla con score ~1.0.
    query = l2_normalize(matrix)[7]
    candidates = index.top_candidates(query, 10)
    assert 7 in candidates
    assert candidates[7] == pytest.approx(1.0, abs=1e-4)


def test_scores_for_rows_matches_full_scan():
    matrix = random_matrix(40)
    index = VectorIndex(matrix, exact_threshold=100)
    query = l2_normalize(random_matrix(1))[0]
    full = index.scores_for_all(query)
    partial = index.scores_for_rows(query, [3, 17, 25])
    for row, score in partial.items():
        assert score == pytest.approx(float(full[row]), abs=1e-6)


def test_memory_accounting_positive():
    matrix = random_matrix(50)
    index = VectorIndex(matrix, exact_threshold=100)
    assert index.memory_bytes >= matrix.nbytes
