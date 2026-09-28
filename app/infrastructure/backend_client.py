"""Cliente HTTP de la API interna del backend (/internal/search/*).

Auth: el secreto compartido X-Internal-Token; X-Internal-Service nos identifica en sus
logs. Todos los payloads van en camelCase (los defaults de Jackson en el backend).
Las API keys llegan como hash SHA-256 (``keyHash``), nunca en claro.

Las lecturas (bootstrap, páginas por clave, datos de lista) son idempotentes y se reintentan
con backoff ante fallos de red o 5xx; los pushes de logs no (su cola ya reintenta y spoolea).
Cada llamada cuenta en la métrica ``xeye_search_backend_requests_total``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.application.ports import BootstrapData, ListDataPayload, LogEntry
from app.core.security import hash_api_key
from app.domain.models import ListMeta
from app.infrastructure import metrics

logger = logging.getLogger(__name__)

# Tamaño de página del snapshot de arranque (el backend admite hasta 5000).
BOOTSTRAP_PAGE = 1000
_RETRY_BACKOFF_SECONDS = 0.5


class BackendClient:
    def __init__(
        self,
        base_url: str,
        internal_token: str,
        timeout: float = 30.0,
        *,
        connect_timeout: float = 5.0,
        retries: int = 2,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            headers={
                "X-Internal-Service": "search-service",
                "X-Internal-Token": internal_token,
            },
        )
        self._retries = max(0, retries)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, operation: str, path: str, params: dict | None = None) -> httpx.Response:
        """GET idempotente con reintentos (fallos de transporte y 5xx); 4xx no se reintenta."""
        attempt = 0
        while True:
            try:
                response = await self._http.get(path, params=params)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"backend answered {response.status_code}", request=response.request, response=response
                    )
                metrics.BACKEND_REQUESTS.labels(operation, "ok").inc()
                return response
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    metrics.BACKEND_REQUESTS.labels(operation, "client_error").inc()
                    raise
                if attempt >= self._retries:
                    metrics.BACKEND_REQUESTS.labels(operation, "error").inc()
                    raise
                attempt += 1
                metrics.BACKEND_REQUESTS.labels(operation, "retry").inc()
                logger.debug("Backend %s failed (%s); retry %d/%d", operation, exc, attempt, self._retries)
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))

    async def fetch_bootstrap(self) -> BootstrapData:
        """El bootstrap trae la primera página (id ascendente) de api keys y de listas; si el
        backend indica ``apiKeysNextAfterId`` / ``listsNextAfterId`` se siguen pidiendo páginas
        por clave a ``/internal/search/api-keys`` y ``/internal/search/lists``."""
        response = await self._get("bootstrap", "/internal/search/bootstrap", params={"limit": BOOTSTRAP_PAGE})
        response.raise_for_status()
        body = response.json()
        api_key_entries = list(body.get("apiKeys", []))
        api_key_entries += await self._fetch_keyset_pages("/internal/search/api-keys", body.get("apiKeysNextAfterId"))
        list_entries = list(body.get("lists", []))
        list_entries += await self._fetch_keyset_pages("/internal/search/lists", body.get("listsNextAfterId"))
        api_keys = [(entry["id"], entry["userId"], _key_hash(entry)) for entry in api_key_entries]
        lists = [
            ListMeta(
                id=entry["id"],
                user_id=entry["userId"],
                name=entry["name"],
                is_public=bool(entry.get("isPublic", False)),
            )
            for entry in list_entries
        ]
        embedding_models = [
            name for name in body.get("embeddingModels") or [] if isinstance(name, str) and name.strip()
        ]
        user_limits = [
            (entry["userId"], int(entry["rateLimitPerMinute"]))
            for entry in body.get("userLimits") or []
            if entry.get("rateLimitPerMinute")
        ]
        return BootstrapData(
            api_keys=api_keys, lists=lists, embedding_models=embedding_models, user_limits=user_limits
        )

    async def _fetch_keyset_pages(self, path: str, next_after_id: int | None) -> list[dict]:
        """Recorre ``{items, nextAfterId}`` hasta que ``nextAfterId`` sea nulo."""
        entries: list[dict] = []
        after_id = next_after_id
        while after_id is not None:
            response = await self._get("bootstrap_page", path, params={"afterId": after_id, "limit": BOOTSTRAP_PAGE})
            response.raise_for_status()
            page = response.json()
            entries.extend(page.get("items") or [])
            after_id = page.get("nextAfterId")
        return entries

    async def fetch_list_data(self, list_id: int) -> ListDataPayload | None:
        response = await self._get("list_data", f"/internal/search/lists/{list_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        meta = ListMeta(
            id=body["id"],
            user_id=body["userId"],
            name=body["name"],
            is_public=bool(body.get("isPublic", False)),
        )
        return ListDataPayload(
            meta=meta,
            elements=body.get("elements") or [],
            embeddings_data=body.get("embeddingsData"),
            model=body.get("model"),
            trained_element_ids=body.get("trainedElementIds"),
        )

    async def push_logs(self, entries: list[LogEntry]) -> None:
        payload = {
            "logs": [
                {
                    "userId": entry.user_id,
                    "apiKeyId": entry.api_key_id,
                    "listId": entry.list_id,
                    "listName": entry.list_name,
                    "endpoint": entry.endpoint,
                    "searchTerm": entry.search_term,
                    "totalResults": entry.total_results,
                    "durationMs": entry.duration_ms,
                    "session": entry.session,
                    "results": entry.results,
                    "searchedAt": entry.searched_at,
                }
                for entry in entries
            ]
        }
        try:
            response = await self._http.post("/internal/search/logs", json=payload)
            response.raise_for_status()
        except Exception:
            metrics.BACKEND_REQUESTS.labels("push_logs", "error").inc()
            raise
        metrics.BACKEND_REQUESTS.labels("push_logs", "ok").inc()


def _key_hash(entry: dict) -> str:
    """``keyHash`` del backend actual. Transitorio: un backend anterior a las keys hasheadas
    aún manda ``apiKey`` en claro; se hashea aquí para que el orden de despliegue no importe.
    Quitar el fallback cuando el backend con V6 lleve un tiempo en producción."""
    key_hash = entry.get("keyHash")
    if key_hash:
        return str(key_hash)
    return hash_api_key(str(entry["apiKey"]))
