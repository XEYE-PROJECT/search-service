import numpy as np

from app.application.ports import ListDataPayload
from app.domain.models import ListMeta
from app.infrastructure.list_builder import build_list_data_sync
from tests.conftest import embeddings_b64, make_settings

SETTINGS = make_settings()


def payload(elements, matrix, trained_ids=None) -> ListDataPayload:
    return ListDataPayload(
        meta=ListMeta(id=1, user_id=1, name="L", is_public=True),
        elements=elements,
        embeddings_data=embeddings_b64(matrix) if matrix is not None else None,
        model=None,
        trained_element_ids=trained_ids,
    )


def elems(*ids):
    return [{"id": i, "text": f"texto {i}", "params": None, "description": None} for i in ids]


def test_legacy_payload_without_ids_keeps_row_alignment():
    matrix = np.eye(3, 4, dtype=np.float32)
    data = build_list_data_sync(payload(elems(1, 2, 3), matrix), SETTINGS)
    assert data.index is not None
    assert data.vector_rows.tolist() == [0, 1, 2]


def test_legacy_payload_count_mismatch_discards_embeddings():
    matrix = np.eye(3, 4, dtype=np.float32)
    data = build_list_data_sync(payload(elems(1, 2), matrix), SETTINGS)
    assert data.index is None


def test_id_alignment_survives_delete_plus_create():
    # Entrenada con ids [1,2,3]; el elemento 2 se borró y se creó el 4 (¡mismo conteo!).
    # La vieja guarda por conteo asignaría mal los vectores; la alineación por id no debe.
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    data = build_list_data_sync(payload(elems(1, 3, 4), matrix, trained_ids=[1, 2, 3]), SETTINGS)
    assert data.index is not None
    # elemento 1 -> fila 0 de la matriz, elemento 3 -> fila 2, elemento 4 -> sin vector
    assert data.vector_rows.tolist() == [0, 1, -1]
    np.testing.assert_allclose(data.embeddings[0], [1, 0, 0, 0])
    np.testing.assert_allclose(data.embeddings[1], [0, 0, 1, 0])
    assert data.element_of_row.tolist() == [0, 1]


def test_id_alignment_all_elements_replaced_degrades_to_text():
    matrix = np.eye(2, 4, dtype=np.float32)
    data = build_list_data_sync(payload(elems(10, 11), matrix, trained_ids=[1, 2]), SETTINGS)
    assert data.index is None
    assert data.vector_rows is None


def test_trained_ids_define_expected_row_count():
    # 3 ids entrenados pero la lista actual creció a 5 elementos: la matriz debe decodificar
    # igualmente (filas == len(trained_ids)) y los 2 elementos nuevos quedan sin vector.
    matrix = np.eye(3, 4, dtype=np.float32)
    data = build_list_data_sync(payload(elems(1, 2, 3, 4, 5), matrix, trained_ids=[1, 2, 3]), SETTINGS)
    assert data.index is not None
    assert data.vector_rows.tolist() == [0, 1, 2, -1, -1]
