import numpy as np
import pytest

from app.application.ports import ListDataPayload
from app.domain.models import ListMeta
from tests.conftest import FakeEmbedder, embeddings_b64, make_container, make_settings

API_KEY = "xeye_test-key"
MODEL_FIELD = '{"embedding_model": "test-model", "llm_model": null, "list_id": 5}'

TEXTS = ["Cámara réflex digital", "Teléfono móvil", "Auriculares bluetooth"]
# Embeddings de documento ortogonales: la fila i "significa" el elemento i.
DOC_MATRIX = np.eye(3, 4, dtype=np.float32)


def camera_payload(list_id: int = 5, texts: list[str] | None = None, name: str = "Cámaras") -> ListDataPayload:
    texts = texts or TEXTS
    return ListDataPayload(
        meta=ListMeta(id=list_id, user_id=10, name=name, is_public=True),
        elements=[
            {"id": i + 1, "text": text, "params": '{"stock": 3}' if i == 0 else None, "description": None}
            for i, text in enumerate(texts)
        ],
        embeddings_data=embeddings_b64(DOC_MATRIX[: len(texts)]),
        model=MODEL_FIELD,
    )


@pytest.fixture
def seeded_backend(backend):
    backend.api_keys = [(1, 10, API_KEY)]
    backend.lists = [
        ListMeta(id=5, user_id=10, name="Cámaras", is_public=True),
        ListMeta(id=6, user_id=10, name="Privada", is_public=False),
    ]
    backend.list_payloads[5] = camera_payload()
    return backend


@pytest.fixture
def seeded_embedder(embedder: FakeEmbedder):
    embedder.vectors["cámara de fotos"] = np.array([1.0, 0.0, 0.0, 0.0])
    return embedder


@pytest.fixture
async def ready_client(client, container, seeded_backend, seeded_embedder):
    await container.catalog_service.refresh()
    return client


async def search(client, term, list_name="Cámaras", key=API_KEY, **extra):
    return await client.post(
        "/api/v1/search",
        headers={"X-API-Key": key} if key else {},
        json={"list_name": list_name, "search_term": term, **extra},
    )


INTERNAL = {"X-Internal-Token": "test-token"}


class TestPublicSearch:
    async def test_semantic_search_lazy_loads_and_ranks(self, ready_client, container, seeded_backend, seeded_embedder):
        response = await search(ready_client, "cámara de fotos", limit=10)
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["list_name"] == "Cámaras"
        assert body["results"][0]["item"] == "Cámara réflex digital"
        assert body["results"][0]["score"] >= 0.75  # la semántica 1.0 domina
        assert body["results"][0]["params"] == {"stock": 3}
        assert isinstance(body["duration_ms"], int)
        # La carga perezosa llamó al backend una sola vez y usó el modelo de la lista.
        assert seeded_backend.list_fetches == [5]
        assert seeded_embedder.calls[0][0] == "test-model"
        # Segunda búsqueda: servida desde la caché.
        await search(ready_client, "cámara de fotos")
        assert seeded_backend.list_fetches == [5]

    async def test_exact_match_scores_one_and_keeps_others(self, ready_client):
        response = await search(ready_client, "teléfono móvil!!", limit=10)
        body = response.json()
        assert body["results"][0]["item"] == "Teléfono móvil"
        assert body["results"][0]["score"] == 1.0
        assert body["total_results"] == len(body["results"])

    async def test_breakdown_only_when_requested(self, ready_client):
        with_breakdown = (await search(ready_client, "camara", include_score_breakdown=True)).json()
        assert "text_score" in with_breakdown["results"][0]
        without = (await search(ready_client, "camara")).json()
        assert "text_score" not in without["results"][0]

    async def test_text_only_when_embedder_unknown_query(self, ready_client):
        # FakeEmbedder devuelve None para queries desconocidas -> puntuación solo textual.
        response = await search(ready_client, "auriculares")
        body = response.json()
        assert body["results"][0]["item"] == "Auriculares bluetooth"
        assert body["results"][0]["score"] > 0.5  # no capado por el peso textual 0.25

    async def test_search_logs_are_queued_and_pushed(self, ready_client, container, seeded_backend):
        await search(ready_client, "cámara de fotos", session="s-1")
        assert container.log_queue.pending == 1
        await container.log_queue.flush()
        log = seeded_backend.pushed_logs[-1]
        assert log.endpoint == "/search"
        assert log.user_id == 10 and log.api_key_id == 1 and log.list_id == 5
        assert log.session == "s-1"
        assert log.results  # {item: puntuación}

    async def test_register_log_false_skips_logging(self, ready_client, container):
        response = await search(ready_client, "cámara de fotos", register_log=False)
        assert response.status_code == 200
        assert container.log_queue.pending == 0

    async def test_target_only_logs(self, ready_client, container, seeded_backend):
        response = await ready_client.post(
            "/api/v1/target",
            headers={"X-API-Key": API_KEY},
            json={"list_name": "Cámaras", "target_term": "Cámara réflex digital", "session": "s-1"},
        )
        assert response.status_code == 200
        assert response.json() == {"success": True}
        await container.log_queue.flush()
        assert seeded_backend.pushed_logs[-1].endpoint == "/target"
        assert seeded_backend.pushed_logs[-1].total_results == 0


class TestPublicErrors:
    async def test_missing_and_invalid_api_key(self, ready_client):
        assert (await search(ready_client, "x", key=None)).status_code == 401
        response = await search(ready_client, "x", key="nope")
        assert response.status_code == 401
        assert response.json()["error"] == "API key inválida"

    async def test_unknown_list_404(self, ready_client):
        response = await search(ready_client, "x", list_name="No existe")
        assert response.status_code == 404
        assert response.json()["error"] == "Lista no encontrada"

    async def test_private_list_403(self, ready_client):
        response = await search(ready_client, "x", list_name="Privada")
        assert response.status_code == 403
        assert response.json()["error"] == "Lista no pública"

    async def test_private_list_allowed_with_allow_private(self, ready_client, seeded_backend):
        # El buscador del frontend manda allow_private=true; la key ya limita al dueño.
        seeded_backend.list_payloads[6] = camera_payload()
        response = await search(ready_client, "camara", list_name="Privada", allow_private=True)
        assert response.status_code == 200
        assert response.json()["total_results"] > 0

    async def test_backend_down_503(self, ready_client, seeded_backend):
        seeded_backend.fail_list_fetch = True
        response = await search(ready_client, "x")
        assert response.status_code == 503

    async def test_deleted_list_lazy_404_cleans_catalog(self, ready_client, container, seeded_backend):
        del seeded_backend.list_payloads[5]  # borrada en el backend, aún en nuestro catálogo
        response = await search(ready_client, "x")
        assert response.status_code == 404
        assert container.lists.get(5) is None

    async def test_rate_limit_429(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        settings = make_settings(rate_limit_per_minute=2)
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await search(client, "a")).status_code == 200
            assert (await search(client, "b")).status_code == 200
            response = await search(client, "c")
            assert response.status_code == 429
            assert response.json()["error"] == "Rate limit alcanzado"

    async def test_cors_preflight_bypasses_auth(self, ready_client):
        response = await ready_client.options(
            "/api/v1/search",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key,content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


class TestInternalApi:
    async def test_internal_requires_token(self, ready_client):
        assert (await ready_client.post("/v1/lists/5/invalidate")).status_code == 403
        assert (
            await ready_client.post("/v1/lists/5/invalidate", headers=INTERNAL)
        ).status_code == 200

    async def test_index_push_serves_search_without_lazy_load(self, ready_client, container, seeded_backend):
        push = {
            "listId": 7,
            "userId": 10,
            "listName": "Nueva",
            "isPublic": True,
            "embeddingsData": None,
            "model": None,
            "elements": [
                {"id": 1, "text": "Bicicleta de montaña", "params": None,
                 "description": None, "generatedDescription": None},
            ],
        }
        response = await ready_client.post("/v1/lists/7/index", headers=INTERNAL, json=push)
        assert response.status_code == 200
        result = await search(ready_client, "bicicleta", list_name="Nueva")
        assert result.status_code == 200
        assert result.json()["results"][0]["item"] == "Bicicleta de montaña"
        assert 7 not in seeded_backend.list_fetches  # caché caliente, sin carga perezosa

    async def test_index_push_with_bad_embeddings_degrades(self, ready_client):
        push = {
            "userId": 10, "listName": "Rota", "isPublic": True,
            "embeddingsData": "bm90LW5weQ==", "model": None,
            "elements": [{"id": 1, "text": "Algo", "params": None,
                          "description": None, "generatedDescription": None}],
        }
        response = await ready_client.post("/v1/lists/8/index", headers=INTERNAL, json=push)
        assert response.status_code == 200  # nunca 5xx: degrada a solo texto
        result = await search(ready_client, "algo", list_name="Rota")
        assert result.status_code == 200

    async def test_invalidate_triggers_lazy_reload(self, ready_client, container, seeded_backend):
        await search(ready_client, "camara")  # calienta la caché
        seeded_backend.list_payloads[5] = camera_payload(texts=TEXTS + ["Trípode nuevo"])
        await ready_client.post("/v1/lists/5/invalidate", headers=INTERNAL)
        result = (await search(ready_client, "tripode nuevo")).json()
        assert result["results"][0]["item"] == "Trípode nuevo"
        assert seeded_backend.list_fetches.count(5) == 2

    async def test_meta_update_renames_list(self, ready_client, seeded_backend):
        # Refleja el renombrado también en el fake (los misses re-sincronizan desde él y
        # la carga perezosa siempre trae los metadatos actuales del backend).
        seeded_backend.lists[0] = ListMeta(id=5, user_id=10, name="Fotografía", is_public=True)
        seeded_backend.list_payloads[5] = camera_payload(name="Fotografía")
        await ready_client.put(
            "/v1/lists/5/meta", headers=INTERNAL,
            json={"userId": 10, "name": "Fotografía", "isPublic": True},
        )
        assert (await search(ready_client, "camara", list_name="Fotografía")).status_code == 200
        assert (await search(ready_client, "camara", list_name="Cámaras")).status_code == 404

    async def test_visibility_off_makes_list_private(self, ready_client):
        await ready_client.put(
            "/v1/lists/5/meta", headers=INTERNAL,
            json={"userId": 10, "name": "Cámaras", "isPublic": False},
        )
        assert (await search(ready_client, "camara")).status_code == 403

    async def test_list_delete_removes_everything(self, ready_client, container, seeded_backend):
        await search(ready_client, "camara")
        # Refleja el borrado en el fake (un miss dispara un re-sync desde él).
        seeded_backend.lists = [meta for meta in seeded_backend.lists if meta.id != 5]
        del seeded_backend.list_payloads[5]
        await ready_client.delete("/v1/lists/5", headers=INTERNAL)
        assert (await search(ready_client, "camara")).status_code == 404
        assert container.cache.get(5) is None

    async def test_api_key_upsert_and_delete(self, ready_client):
        await ready_client.put(
            "/v1/api-keys/9", headers=INTERNAL,
            json={"userId": 10, "apiKey": "xeye_new-key"},
        )
        assert (await search(ready_client, "camara", key="xeye_new-key")).status_code == 200
        await ready_client.delete("/v1/api-keys/9", headers=INTERNAL)
        assert (await search(ready_client, "camara", key="xeye_new-key")).status_code == 401

    async def test_user_delete_drops_keys_and_lists(self, ready_client, container):
        await ready_client.delete("/v1/users/10", headers=INTERNAL)
        assert container.api_keys.resolve(API_KEY) is None
        assert container.lists.get(5) is None

    async def test_health_reports_state(self, ready_client, container):
        await search(ready_client, "camara")
        body = (await ready_client.get("/v1/health", headers=INTERNAL)).json()
        assert body["ready"] is True
        assert body["lists_cached"] == 1
        assert body["cache_bytes"] > 0
