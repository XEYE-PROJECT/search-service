"""BackendClient.fetch_bootstrap recorre las páginas por clave que anuncia el backend."""

from __future__ import annotations

import json

import httpx
import pytest

from app.infrastructure.backend_client import BackendClient


def _make_client(handler) -> BackendClient:
    client = BackendClient(base_url="http://backend", internal_token="t")
    client._http = httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(handler),
        headers={"X-Internal-Token": "t"},
    )
    return client


@pytest.mark.asyncio
async def test_bootstrap_follows_keyset_pages_for_keys_and_lists():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        path = request.url.path
        after = request.url.params.get("afterId")
        if path == "/internal/search/bootstrap":
            return httpx.Response(
                200,
                json={
                    "apiKeys": [{"id": 1, "userId": 10, "keyHash": "a" * 64}],
                    "apiKeysNextAfterId": 1,
                    "lists": [{"id": 5, "userId": 10, "name": "one", "isPublic": True}],
                    "listsNextAfterId": 5,
                    "embeddingModels": ["m"],
                    "userLimits": [{"userId": 10, "rateLimitPerMinute": 7}],
                },
            )
        if path == "/internal/search/api-keys":
            if after == "1":
                return httpx.Response(
                    200, json={"items": [{"id": 2, "userId": 11, "keyHash": "b" * 64}], "nextAfterId": 2}
                )
            return httpx.Response(200, json={"items": [], "nextAfterId": None})
        if path == "/internal/search/lists" and after == "5":
            return httpx.Response(
                200, json={"items": [{"id": 6, "userId": 11, "name": "two", "isPublic": False}], "nextAfterId": None}
            )
        return httpx.Response(404, json={})

    data = await _make_client(handler).fetch_bootstrap()

    assert [k[0] for k in data.api_keys] == [1, 2]
    assert [entry.id for entry in data.lists] == [5, 6]
    assert data.embedding_models == ["m"]
    assert data.user_limits == [(10, 7)]
    # bootstrap + 2 páginas de keys (la última vacía) + 1 de listas
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_bootstrap_without_more_pages_makes_a_single_call():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "apiKeys": [],
                "apiKeysNextAfterId": None,
                "lists": [],
                "listsNextAfterId": None,
                "embeddingModels": [],
                "userLimits": [],
            },
        )

    data = await _make_client(handler).fetch_bootstrap()

    assert data.api_keys == [] and data.lists == []
    assert calls == ["/internal/search/bootstrap"]


@pytest.mark.asyncio
async def test_legacy_bootstrap_without_next_fields_still_works():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=json.loads(
                json.dumps(
                    {
                        "apiKeys": [{"id": 1, "userId": 10, "keyHash": "a" * 64}],
                        "lists": [{"id": 5, "userId": 10, "name": "one", "isPublic": True}],
                        "embeddingModels": [],
                    }
                )
            ),
        )

    data = await _make_client(handler).fetch_bootstrap()
    assert len(data.api_keys) == 1 and len(data.lists) == 1
