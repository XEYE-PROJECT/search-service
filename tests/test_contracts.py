"""Contratos backend → search-service, verificados con el código real.

Las fixtures de ``tests/contracts/`` son copias byte a byte de las canónicas de
``backend/src/test/resources/contracts/`` (su README explica cada una; ``contracts-check.sh``
en xeye-infra comprueba que las copias coinciden). El backend tiene un test que las PRODUCE
con sus DTOs; aquí se CONSUMEN con los mismos esquemas y el mismo cliente HTTP que usa el
servicio, así un cambio de campo rompe en CI el lado que no se actualizó, no en producción.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np

from app.application.ports import ListDataPayload
from app.domain.models import ListMeta
from app.infrastructure.backend_client import BackendClient
from app.infrastructure.embeddings import decode_embeddings
from app.infrastructure.list_builder import build_list_data_sync
from app.infrastructure.web.schemas import IndexPushRequest
from tests.conftest import make_settings

CONTRACTS = Path(__file__).parent / "contracts"
INTERNAL = {"X-Internal-Token": "test-token"}  # el de make_settings()

# Matriz 2×4 float32 que codifica `embeddingsData` en las fixtures (base64 de np.save).
EXPECTED_MATRIX = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
MODEL_FIELD = (
    '{"embedding_model": "paraphrase-multilingual-MiniLM-L12-v2", "llm_model": "llama-3.3-70b-versatile", '
    '"strategy": "default", "dimension": 4, "list_id": 7}'
)


def load_contract(name: str) -> dict:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def _mock_backend_client(handler) -> BackendClient:
    """BackendClient real con el transporte sustituido; conserva sus cabeceras de servicio."""
    client = BackendClient(base_url="http://backend.test", internal_token="test-token")
    client._http = httpx.AsyncClient(
        base_url="http://backend.test",
        transport=httpx.MockTransport(handler),
        headers=dict(client._http.headers),
    )
    return client


# ---------------------------------------------------------------- search-index-push.json --


def test_index_push_contract_maps_search_index_command():
    request = IndexPushRequest.model_validate(load_contract("search-index-push.json"))

    assert request.list_id == 7
    assert request.user_id == 3
    assert request.list_name == "Herramientas"
    assert request.is_public is True
    assert request.model == MODEL_FIELD
    assert request.trained_element_ids == [101, 102]
    assert [element.id for element in request.elements] == [101, 102]

    first, second = request.elements
    assert first.text == "martillo de carpintero"
    assert first.params == '{"precio":12.5,"stock":8}'
    assert first.description == "Mango de madera, 500 g"
    assert second.params is None and second.description is None
    # `generatedDescription` es el JSON opaco del worker (el backend lo cachea y lo reenvía).
    assert second.generated_description is not None
    assert json.loads(second.generated_description)["summary"] == ["Destornillador de punta Phillips"]


def test_index_push_embeddings_decode_to_float32_matrix():
    request = IndexPushRequest.model_validate(load_contract("search-index-push.json"))
    assert request.trained_element_ids is not None

    matrix = decode_embeddings(request.embeddings_data, expected_rows=len(request.trained_element_ids))

    assert matrix is not None
    assert matrix.shape == (2, 4) and matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, EXPECTED_MATRIX)


# ------------------------------------------ search-bootstrap.json + search-bootstrap-page.json --


async def test_bootstrap_contract_with_keyset_page():
    bootstrap = load_contract("search-bootstrap.json")
    page = load_contract("search-bootstrap-page.json")
    # La fixture canónica cierra en una página; aquí se anuncia una segunda de listas, que
    # es exactamente el contrato `KeysetPage` de search-bootstrap-page.json.
    bootstrap_with_more = {**bootstrap, "listsNextAfterId": 8}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/internal/search/bootstrap":
            return httpx.Response(200, json=bootstrap_with_more)
        if request.url.path == "/internal/search/lists" and request.url.params.get("afterId") == "8":
            return httpx.Response(200, json=page)
        return httpx.Response(404, json={})

    data = await _mock_backend_client(handler).fetch_bootstrap()

    assert data.api_keys == [(1, 3, bootstrap["apiKeys"][0]["keyHash"])]
    assert len(data.api_keys[0][2]) == 64  # hash SHA-256, nunca la key en claro
    assert data.lists == [
        ListMeta(id=7, user_id=3, name="Herramientas", is_public=True),
        ListMeta(id=8, user_id=3, name="Privada", is_public=False),
        ListMeta(id=9, user_id=4, name="Otra", is_public=True),
    ]
    assert data.embedding_models == [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
    ]
    assert data.user_limits == [(3, 120)]
    # bootstrap + la página de listas, ambas autenticadas como servicio.
    assert [r.url.path for r in requests] == ["/internal/search/bootstrap", "/internal/search/lists"]
    for request in requests:
        assert request.headers["X-Internal-Token"] == "test-token"
        assert request.headers["X-Internal-Service"] == "search-service"


# ----------------------------------------------------------------- search-list-data.json --


async def test_list_data_contract_maps_list_search_data_response():
    body = load_contract("search-list-data.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/search/lists/7"
        return httpx.Response(200, json=body)

    payload = await _mock_backend_client(handler).fetch_list_data(7)

    assert payload is not None
    assert payload.meta == ListMeta(id=7, user_id=3, name="Herramientas", is_public=True)
    assert payload.model == MODEL_FIELD
    assert payload.trained_element_ids == [101, 102]
    assert payload.embeddings_data == body["embeddingsData"]
    assert payload.elements == [
        {
            "id": 101,
            "text": "martillo de carpintero",
            "params": '{"precio":12.5,"stock":8}',
            "description": "Mango de madera, 500 g",
        },
        {"id": 102, "text": "destornillador de estrella", "params": None, "description": None},
    ]
    _assert_builds_searchable_data(payload)


def _assert_builds_searchable_data(payload: ListDataPayload) -> None:
    """El payload del contrato pasa entero por el constructor real de datos en RAM."""
    data = build_list_data_sync(payload, make_settings())
    assert data.list_id == 7 and data.user_id == 3
    assert data.element_ids == [101, 102]
    assert data.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert data.index is not None and data.embeddings is not None
    assert data.embeddings.shape == (2, 4)
    assert data.vector_rows is not None and data.vector_rows.tolist() == [0, 1]


# ------------------------------------------------------------- push por HTTP, de punta a punta --


async def test_index_push_over_http_then_console_search(client, container):
    await container.catalog_service.refresh()  # catálogos vacíos, pero listos
    push = load_contract("search-index-push.json")

    response = await client.post("/v1/lists/7/index", headers=INTERNAL, json=push)

    assert response.status_code == 200
    assert response.json()["success"] is True

    # Búsqueda del playground sobre la lista recién indexada. El embedder de los tests
    # devuelve None para cualquier query → solo texto, y la respuesta lo declara.
    search = await client.post("/v1/lists/7/search", headers=INTERNAL, json={"search_term": "martillo"})

    assert search.status_code == 200
    body = search.json()
    assert body["list_name"] == "Herramientas"
    assert body["results"][0]["item"] == "martillo de carpintero"
    assert body["results"][0]["params"] == {"precio": 12.5, "stock": 8}
    assert body["degraded"] is True and body["degradation_reasons"] == ["model_unavailable"]
    assert search.headers["X-Search-Degraded"] == "true"
