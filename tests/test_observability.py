"""Request id, access log JSON, métricas, sondas con estado real, 503 mientras no está listo e
indicador de degradación en la respuesta."""

import json
import logging

import httpx
import numpy as np

from app.core.logging import JsonFormatter
from app.core.request_context import request_id_var, sanitize_request_id
from app.infrastructure import metrics
from app.main import create_app
from tests.conftest import make_container, make_settings
from tests.test_api import (  # noqa: F401 — las fixtures importadas quedan disponibles aquí
    API_KEY,
    INTERNAL,
    assert_error,
    ready_client,
    search,
    seeded_backend,
    seeded_embedder,
)


async def _client(settings, backend, embedder, *, refresh=True):
    container = make_container(settings, backend, embedder)
    if refresh:
        await container.catalog_service.refresh()
    app = create_app(settings=settings, container=container)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t"), container


class TestRequestId:
    async def test_response_carries_a_request_id(self, ready_client):
        response = await ready_client.get("/health")
        assert len(response.headers["x-request-id"]) == 32

    async def test_proxy_request_id_is_echoed_when_sane(self, ready_client):
        response = await ready_client.get("/health", headers={"X-Request-Id": "caddy-abc.123"})
        assert response.headers["x-request-id"] == "caddy-abc.123"
        response = await ready_client.get("/health", headers={"X-Request-Id": "bad id\nwith newline"})
        assert response.headers["x-request-id"] != "bad id\nwith newline"

    def test_sanitize(self):
        assert sanitize_request_id(" abc-1 ") == "abc-1"
        assert sanitize_request_id("x" * 200) is None
        assert sanitize_request_id("") is None
        assert sanitize_request_id("has space") is None

    async def test_access_log_line_has_request_id_and_key_prefix(self, ready_client, caplog):
        with caplog.at_level(logging.INFO, logger="xeye.access"):
            await search(ready_client, "camara", key="xeye_secret-key-value-1234")
        records = [r for r in caplog.records if r.name == "xeye.access"]
        assert records, "one access line per request"
        line = records[-1]
        assert line.path == "/api/v1/search" and line.status == 401
        assert line.key_prefix == "xeye_secret-…" and "value-1234" not in line.getMessage()


class TestJsonLogs:
    def test_json_formatter_includes_request_id_and_extras(self):
        record = logging.LogRecord("xeye.access", logging.INFO, __file__, 1, "GET %s", ("/x",), None)
        record.status = 200
        token = request_id_var.set("rid-1")
        try:
            payload = json.loads(JsonFormatter().format(record))
        finally:
            request_id_var.reset(token)
        assert payload["msg"] == "GET /x" and payload["level"] == "INFO"
        assert payload["request_id"] == "rid-1" and payload["status"] == 200
        assert payload["ts"].endswith("Z")

    def test_log_format_auto_is_json_only_in_production(self):
        assert make_settings().log_format_resolved == "text"
        assert make_settings(environment="production", internal_token="p" * 40).log_format_resolved == "json"
        assert make_settings(log_format="json").log_format_resolved == "json"


class TestReadiness:
    async def test_public_search_is_503_not_401_while_catalog_is_unavailable(self, seeded_backend, seeded_embedder):
        seeded_backend.fail_bootstrap = True
        client, container = await _client(make_settings(), seeded_backend, seeded_embedder, refresh=False)
        async with client:
            body = assert_error(await search(client, "camara"), 503, "SERVICE_NOT_READY")
            assert body["error"] == "Service Unavailable"
            # Sin key sigue siendo 401: no hace falta catálogo para saber que falta la cabecera.
            assert_error(await search(client, "camara", key=None), 401, "API_KEY_MISSING")
            # El playground también espera al catálogo.
            assert_error(await client.post("/v1/lists/5/search", headers=INTERNAL, json={"search_term": "x"}),
                         503, "SERVICE_NOT_READY")
            # En cuanto el backend vuelve, la misma petición se sirve (recarga inmediata).
            seeded_backend.fail_bootstrap = False
            assert (await search(client, "camara")).status_code == 200
            assert container.catalog_service.ready

    async def test_health_and_ready_report_real_state(self, client, container, seeded_backend):
        health = (await client.get("/health")).json()
        assert health["status"] == "ok" and health["ready"] is False
        assert health["checks"]["catalog"] == "loading"
        ready = await client.get("/ready")
        assert ready.status_code == 503 and ready.json()["checks"]["catalog"] == "loading"

        await container.catalog_service.refresh()
        ready = (await client.get("/ready")).json()
        assert ready["status"] == "ready" and ready["degraded"] is False
        assert ready["checks"]["search_logs"] == "ok"
        assert (await client.get("/v1/health", headers=INTERNAL)).json()["checks"]["catalog"] == "ok"


class TestDegradation:
    async def test_text_only_list_is_flagged_no_embeddings(self, ready_client):
        push = {"userId": 10, "listName": "Texto", "isPublic": True, "embeddingsData": None, "model": None,
                "elements": [{"id": 1, "text": "Algo", "params": None, "description": None}]}
        await ready_client.post("/v1/lists/8/index", headers=INTERNAL, json=push)
        response = await search(ready_client, "algo", list_name="Texto")
        body = response.json()
        assert body["degraded"] is True and body["degradation_reasons"] == ["no_embeddings"]
        assert response.headers["x-search-degraded"] == "true"

    async def test_model_unavailable_is_flagged(self, ready_client):
        # FakeEmbedder devuelve None para queries desconocidas → sin semántica.
        body = (await search(ready_client, "auriculares")).json()
        assert body["degraded"] is True and body["degradation_reasons"] == ["model_unavailable"]

    async def test_healthy_search_is_not_degraded(self, ready_client):
        response = await search(ready_client, "cámara de fotos")
        body = response.json()
        assert body["degraded"] is False and body["degradation_reasons"] == []
        assert "x-search-degraded" not in response.headers

    async def test_model_mismatch_is_flagged(self, ready_client, seeded_embedder):
        seeded_embedder.vectors["otra dim"] = np.array([1.0, 0.0])  # la lista tiene dim 4
        body = (await search(ready_client, "otra dim")).json()
        assert body["degradation_reasons"] == ["model_mismatch"]

    async def test_stale_data_is_flagged_when_revalidation_fails(self, ready_client, container, seeded_backend):
        await search(ready_client, "cámara de fotos")
        container.list_data._stale.add(5)  # revalidación fallida (ver test_cache_ttl)
        body = (await search(ready_client, "cámara de fotos")).json()
        assert "stale_data" in body["degradation_reasons"]


class TestMetrics:
    async def test_metrics_endpoint_exposes_counters_and_gauges(self, ready_client):
        await search(ready_client, "cámara de fotos")
        await search(ready_client, "x", key="nope")
        text = (await ready_client.get("/metrics")).text
        assert 'xeye_search_http_requests_total{method="POST",path="/api/v1/search",status="200"}' in text
        assert 'xeye_search_searches_total{outcome="ok",surface="public"}' in text
        assert "xeye_search_ready 1.0" in text
        assert "xeye_search_cache_lists 1.0" in text
        assert "xeye_search_http_request_duration_seconds_bucket" in text

    async def test_metrics_can_be_disabled(self, seeded_backend, seeded_embedder):
        client, _ = await _client(make_settings(metrics_enabled=False), seeded_backend, seeded_embedder)
        async with client:
            assert (await client.get("/metrics")).status_code == 404

    def test_path_labels_are_bounded(self):
        assert metrics.normalize_path("/v1/lists/42/search") == "/v1/lists/{id}/search"
        assert metrics.normalize_path("/api/v1/search") == "/api/v1/search"

    async def test_rate_limited_counter(self, seeded_backend, seeded_embedder):
        before = metrics.RATE_LIMITED.labels("account")._value.get()
        client, _ = await _client(make_settings(rate_limit_per_minute=1), seeded_backend, seeded_embedder)
        async with client:
            await search(client, "a")
            assert (await search(client, "b")).status_code == 429
        assert metrics.RATE_LIMITED.labels("account")._value.get() == before + 1


class TestUsageLogsAreMandatory:
    async def test_register_log_false_is_ignored(self, ready_client, container):
        response = await search(ready_client, "cámara de fotos", register_log=False)
        assert response.status_code == 200
        assert container.log_queue.pending == 1  # se registra igualmente

    async def test_revoked_key_is_rejected_on_the_very_next_request(self, ready_client, container, seeded_backend):
        assert (await search(ready_client, "camara")).status_code == 200
        seeded_backend.api_keys = []  # el backend ya la borró: el push llega a continuación
        await ready_client.delete("/v1/api-keys/1", headers=INTERNAL)
        assert_error(await search(ready_client, "camara"), 401, "API_KEY_INVALID")
        assert container.api_keys.resolve_raw(API_KEY) is None
