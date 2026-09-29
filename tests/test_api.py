import numpy as np
import pytest

from app.application.ports import ListDataPayload
from app.core.security import hash_api_key
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
    backend.api_keys = [(1, 10, hash_api_key(API_KEY))]  # el backend solo manda hashes
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


def assert_error(response, status: int, code: str) -> dict:
    """Todos los errores comparten el sobre del backend: status/error/code/message."""
    body = response.json()
    assert response.status_code == status, body
    assert body["status"] == status and body["code"] == code
    assert isinstance(body["error"], str) and isinstance(body["message"], str) and body["message"]
    return body


class TestPublicErrors:
    async def test_missing_and_invalid_api_key(self, ready_client):
        assert_error(await search(ready_client, "x", key=None), 401, "API_KEY_MISSING")
        body = assert_error(await search(ready_client, "x", key="nope"), 401, "API_KEY_INVALID")
        assert body["error"] == "Unauthorized"

    async def test_unknown_list_404(self, ready_client):
        assert_error(await search(ready_client, "x", list_name="No existe"), 404, "LIST_NOT_FOUND")

    async def test_private_list_403(self, ready_client):
        assert_error(await search(ready_client, "x", list_name="Privada"), 403, "LIST_NOT_PUBLIC")

    async def test_private_list_cannot_be_unlocked_by_the_client(self, ready_client, seeded_backend):
        # El antiguo allow_private ya no existe: es un campo desconocido -> 422, y la lista
        # privada sigue siendo 403 por mucho que el cliente lo pida.
        seeded_backend.list_payloads[6] = camera_payload()
        assert_error(
            await search(ready_client, "camara", list_name="Privada", allow_private=True), 422, "VALIDATION_FAILED"
        )
        assert_error(await search(ready_client, "camara", list_name="Privada"), 403, "LIST_NOT_PUBLIC")

    @pytest.mark.parametrize(
        "payload",
        [
            {"list_name": "Cámaras", "search_term": "   "},
            {"list_name": "  ", "search_term": "x"},
            {"list_name": "Cámaras", "search_term": ""},
            {"list_name": "Cámaras", "search_term": "x", "limit": 0},
            {"list_name": "Cámaras", "search_term": "x", "session": ""},
            {"list_name": "Cámaras", "search_term": "x", "unknown": 1},
        ],
    )
    async def test_strict_body_validation_422(self, ready_client, payload):
        response = await ready_client.post("/api/v1/search", headers={"X-API-Key": API_KEY}, json=payload)
        body = assert_error(response, 422, "VALIDATION_FAILED")
        assert body["details"]  # campo -> motivo

    async def test_search_term_is_trimmed(self, ready_client):
        response = await search(ready_client, "  teléfono móvil  ")
        assert response.status_code == 200
        assert response.json()["search_term"] == "teléfono móvil"

    async def test_unknown_route_and_method_share_the_envelope(self, ready_client):
        assert_error(await ready_client.get("/api/v1/nope"), 404, "NOT_FOUND")
        assert_error(await ready_client.get("/api/v1/search"), 405, "METHOD_NOT_ALLOWED")

    async def test_public_body_too_large_413(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        settings = make_settings(max_request_bytes=200)
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await search(client, "camara")).status_code == 200
            big = {"list_name": "Cámaras", "search_term": "x" * 400}
            response = await client.post("/api/v1/search", headers={"X-API-Key": API_KEY}, json=big)
            assert_error(response, 413, "REQUEST_TOO_LARGE")
            # La API interna no se limita (el push de índice lleva embeddings).
            push = {
                "userId": 10,
                "listName": "Grande",
                "isPublic": True,
                "embeddingsData": None,
                "model": None,
                "elements": [
                    {"id": i, "text": f"elemento {i}", "params": None, "description": None} for i in range(50)
                ],
            }
            assert (await client.post("/v1/lists/9/index", headers=INTERNAL, json=push)).status_code == 200

    async def test_untrusted_host_400(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        settings = make_settings(allowed_hosts="search.test,localhost")
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://search.test") as client:
            assert (await search(client, "camara")).status_code == 200
            assert (await client.get("/health", headers={"Host": "localhost:8002"})).status_code == 200
            assert_error(await client.get("/health", headers={"Host": "evil.example"}), 400, "INVALID_HOST")

    async def test_backend_down_503(self, ready_client, seeded_backend):
        seeded_backend.fail_list_fetch = True
        assert_error(await search(ready_client, "x"), 503, "BACKEND_UNAVAILABLE")

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
            first = await search(client, "a")
            assert first.status_code == 200
            assert first.headers["x-ratelimit-limit"] == "2"
            assert first.headers["x-ratelimit-remaining"] == "1"
            assert int(first.headers["x-ratelimit-reset"]) >= 1
            assert (await search(client, "b")).status_code == 200
            response = await search(client, "c")
            body = assert_error(response, 429, "RATE_LIMITED")
            assert "per account" in body["message"]
            assert int(response.headers["retry-after"]) >= 1
            assert response.headers["x-ratelimit-remaining"] == "0"
            assert response.headers["x-ratelimit-reset"] == response.headers["retry-after"]

    async def test_rate_limit_is_per_user_across_keys(self, seeded_backend, seeded_embedder):
        # Dos keys del mismo usuario comparten el cupo; otro usuario tiene el suyo.
        import httpx

        from app.main import create_app

        seeded_backend.api_keys += [(2, 10, hash_api_key("xeye_second")), (3, 11, hash_api_key("xeye_other"))]
        seeded_backend.lists.append(ListMeta(id=7, user_id=11, name="Otra", is_public=True))
        seeded_backend.list_payloads[7] = camera_payload(list_id=7, name="Otra")
        settings = make_settings(rate_limit_per_minute=2)
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await search(client, "a", key=API_KEY)).status_code == 200
            assert (await search(client, "b", key="xeye_second")).status_code == 200
            assert (await search(client, "c", key="xeye_second")).status_code == 429
            assert (await search(client, "d", key="xeye_other", list_name="Otra")).status_code == 200

    async def test_rate_limit_per_user_override_from_backend(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        seeded_backend.user_limits = [(10, 1)]  # el admin bajó el cupo del usuario 10 a 1/min
        settings = make_settings(rate_limit_per_minute=1000)
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = await search(client, "a")
            assert first.status_code == 200 and first.headers["x-ratelimit-limit"] == "1"
            assert (await search(client, "b")).status_code == 429
            # Cambio en caliente desde el backend: PUT con null vuelve al límite por defecto.
            response = await client.put("/v1/users/10/limits", headers=INTERNAL, json={"rateLimitPerMinute": None})
            assert response.status_code == 200
            assert (await search(client, "c")).headers["x-ratelimit-limit"] == "1000"
            await client.put("/v1/users/10/limits", headers=INTERNAL, json={"rateLimitPerMinute": 5})
            assert (await search(client, "d")).headers["x-ratelimit-limit"] == "5"
            assert_error(
                await client.put("/v1/users/10/limits", headers=INTERNAL, json={"rateLimitPerMinute": 0}),
                422,
                "VALIDATION_FAILED",
            )

    async def test_ip_rate_limit_applies_before_the_key(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        settings = make_settings(rate_limit_per_ip_per_minute=2)
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await search(client, "a", key="bad-1")).status_code == 401
            assert (await search(client, "b", key="bad-2")).status_code == 401
            body = assert_error(await search(client, "c"), 429, "RATE_LIMITED")  # incluso con la key buena
            assert "per IP address" in body["message"]

    async def test_security_rejections_are_audited(self, ready_client, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="xeye.audit"):
            await search(ready_client, "x", key="xeye_secret-key-value-1234")
            await ready_client.post("/v1/lists/5/invalidate", headers={"X-Internal-Token": "wrong-token"})
        messages = [r.getMessage() for r in caplog.records if r.name == "xeye.audit"]
        assert any("API_KEY_INVALID" in m and "ip=" in m and "path=/api/v1/search" in m for m in messages)
        assert any("INTERNAL_TOKEN_INVALID" in m for m in messages)
        joined = "\n".join(messages)
        assert "xeye_secret-key-value-1234" not in joined  # solo el prefijo
        assert "xeye_secret-" in joined
        assert "wrong-token" not in joined

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
        assert (await ready_client.post("/v1/lists/5/invalidate", headers=INTERNAL)).status_code == 200

    async def test_internal_rejects_wrong_or_partial_token(self, ready_client):
        for token in ("nope", "test-toke", "test-token ", "TEST-TOKEN", ""):
            response = await ready_client.post("/v1/lists/5/invalidate", headers={"X-Internal-Token": token})
            assert_error(response, 403, "INTERNAL_TOKEN_INVALID")

    async def test_console_search_serves_private_lists_without_api_key(self, ready_client, container, seeded_backend):
        seeded_backend.list_payloads[6] = camera_payload(list_id=6, name="Privada")
        response = await ready_client.post(
            "/v1/lists/6/search",
            headers=INTERNAL,
            json={"search_term": "camara", "limit": 5, "include_score_breakdown": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["list_name"] == "Privada" and body["total_results"] > 0
        assert "text_score" in body["results"][0]
        assert response.headers["x-ratelimit-limit"]  # cuenta contra el cupo del dueño
        assert container.log_queue.pending == 0  # la consola no genera logs de búsqueda
        assert_error(
            await ready_client.post("/v1/lists/999/search", headers=INTERNAL, json={"search_term": "x"}),
            404,
            "LIST_NOT_FOUND",
        )
        assert_error(
            await ready_client.post(
                "/v1/lists/6/search", headers=INTERNAL, json={"search_term": "x", "allow_private": True}
            ),
            422,
            "VALIDATION_FAILED",
        )
        assert (await ready_client.post("/v1/lists/6/search", json={"search_term": "x"})).status_code == 403

    async def test_console_search_shares_the_user_quota(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        settings = make_settings(rate_limit_per_minute=2)
        container = make_container(settings, seeded_backend, seeded_embedder)
        await container.catalog_service.refresh()
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await search(client, "a")).status_code == 200
            console = await client.post("/v1/lists/5/search", headers=INTERNAL, json={"search_term": "b"})
            assert console.status_code == 200
            response = await client.post("/v1/lists/5/search", headers=INTERNAL, json={"search_term": "c"})
            assert_error(response, 429, "RATE_LIMITED")
            assert response.headers["retry-after"]

    async def test_internal_fails_closed_with_blank_configured_token(self, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        settings = make_settings(internal_token="   ")
        container = make_container(settings, seeded_backend, seeded_embedder)
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            for token in ("   ", ""):
                response = await client.post("/v1/lists/5/invalidate", headers={"X-Internal-Token": token})
                assert response.status_code == 403

    async def test_index_push_serves_search_without_lazy_load(self, ready_client, container, seeded_backend):
        push = {
            "listId": 7,
            "userId": 10,
            "listName": "Nueva",
            "isPublic": True,
            "embeddingsData": None,
            "model": None,
            "elements": [
                {
                    "id": 1,
                    "text": "Bicicleta de montaña",
                    "params": None,
                    "description": None,
                    "generatedDescription": None,
                },
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
            "userId": 10,
            "listName": "Rota",
            "isPublic": True,
            "embeddingsData": "bm90LW5weQ==",
            "model": None,
            "elements": [{"id": 1, "text": "Algo", "params": None, "description": None, "generatedDescription": None}],
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
            "/v1/lists/5/meta",
            headers=INTERNAL,
            json={"userId": 10, "name": "Fotografía", "isPublic": True},
        )
        assert (await search(ready_client, "camara", list_name="Fotografía")).status_code == 200
        assert (await search(ready_client, "camara", list_name="Cámaras")).status_code == 404

    async def test_visibility_off_makes_list_private(self, ready_client):
        await ready_client.put(
            "/v1/lists/5/meta",
            headers=INTERNAL,
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
            "/v1/api-keys/9",
            headers=INTERNAL,
            json={"userId": 10, "keyHash": hash_api_key("xeye_new-key")},
        )
        assert (await search(ready_client, "camara", key="xeye_new-key")).status_code == 200
        await ready_client.delete("/v1/api-keys/9", headers=INTERNAL)
        assert (await search(ready_client, "camara", key="xeye_new-key")).status_code == 401

    async def test_api_key_upsert_accepts_transitional_raw_key(self, ready_client):
        # Backend anterior a las keys hasheadas: manda apiKey en claro y se hashea aquí.
        await ready_client.put(
            "/v1/api-keys/9",
            headers=INTERNAL,
            json={"userId": 10, "apiKey": "xeye_legacy-key"},
        )
        assert (await search(ready_client, "camara", key="xeye_legacy-key")).status_code == 200

    async def test_api_key_upsert_rejects_malformed_hash(self, ready_client):
        response = await ready_client.put(
            "/v1/api-keys/9", headers=INTERNAL, json={"userId": 10, "keyHash": "not-a-sha256"}
        )
        assert response.status_code == 422
        response = await ready_client.put("/v1/api-keys/9", headers=INTERNAL, json={"userId": 10})
        assert response.status_code == 422

    async def test_user_delete_drops_keys_lists_and_limits(self, ready_client, container):
        container.user_limits.set(10, 7)
        await ready_client.delete("/v1/users/10", headers=INTERNAL)
        assert container.api_keys.resolve_raw(API_KEY) is None
        assert container.lists.get(5) is None
        assert container.user_limits.get(10, default=60) == 60

    async def test_health_reports_state(self, ready_client, container):
        await search(ready_client, "camara")
        body = (await ready_client.get("/v1/health", headers=INTERNAL)).json()
        assert body["ready"] is True
        assert body["lists_cached"] == 1
        assert body["cache_bytes"] > 0


class TestProbesAndDocs:
    async def test_ready_reports_503_until_bootstrapped(self, client, container):
        assert (await client.get("/health")).status_code == 200
        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "starting"

    async def test_ready_reports_200_once_bootstrapped(self, ready_client):
        response = await ready_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    async def test_docs_only_in_development(self, ready_client, seeded_backend, seeded_embedder):
        import httpx

        from app.main import create_app

        assert (await ready_client.get("/openapi.json")).status_code == 200  # development

        settings = make_settings(environment="production", internal_token="p" * 40)
        container = make_container(settings, seeded_backend, seeded_embedder)
        app = create_app(settings=settings, container=container)
        transport = httpx.ASGITransport(app=app)
        # En producción solo se aceptan los Host de ALLOWED_HOSTS (make_settings pone search.xeye.es).
        async with httpx.AsyncClient(transport=transport, base_url="http://search.xeye.es") as prod_client:
            for path in ("/docs", "/redoc", "/openapi.json"):
                assert (await prod_client.get(path)).status_code == 404, path
            assert (await prod_client.get("/health")).status_code == 200
