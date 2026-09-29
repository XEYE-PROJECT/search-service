"""Métricas Prometheus del servicio (``GET /metrics``, solo por la red docker).

Contadores e histogramas se actualizan en caliente (middleware, casos de uso); los gauges de
estado (listo, caché, colas) se leen del contenedor en cada scrape con ``refresh_gauges``.
Las etiquetas de ruta se normalizan (ids numéricos → ``{id}``) para acotar la cardinalidad.
"""

from __future__ import annotations

import re

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_NUMERIC_SEGMENT = re.compile(r"/\d+")

HTTP_REQUESTS = Counter("xeye_search_http_requests_total", "HTTP requests served", ["method", "path", "status"])
HTTP_DURATION = Histogram(
    "xeye_search_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
SEARCHES = Counter("xeye_search_searches_total", "Searches executed", ["surface", "outcome"])
DEGRADED = Counter("xeye_search_degraded_total", "Searches served with a degradation reason", ["reason"])
RATE_LIMITED = Counter("xeye_search_rate_limited_total", "Requests rejected with 429", ["scope"])
BACKEND_REQUESTS = Counter(
    "xeye_search_backend_requests_total", "Calls to the backend internal API", ["operation", "outcome"]
)
CATALOG_REFRESHES = Counter("xeye_search_catalog_refreshes_total", "Catalog refreshes against the backend", ["outcome"])
CACHE_EVENTS = Counter("xeye_search_cache_events_total", "List cache events", ["event"])
LOG_SPOOLED = Counter("xeye_search_log_spooled_total", "Search-log entries written to the disk spool")
LOG_DROPPED = Counter("xeye_search_log_dropped_total", "Search-log entries dropped for good")

READY = Gauge("xeye_search_ready", "1 when the catalogs are loaded from the backend")
DEGRADED_STATE = Gauge("xeye_search_degraded_state", "1 when the service runs degraded (e.g. no embedding model)")
CACHE_LISTS = Gauge("xeye_search_cache_lists", "Lists resident in the search cache")
CACHE_BYTES = Gauge("xeye_search_cache_bytes", "Bytes used by the search cache")
CATALOG_LISTS = Gauge("xeye_search_catalog_lists", "Lists known in the catalog")
CATALOG_API_KEYS = Gauge("xeye_search_catalog_api_keys", "API keys known in the catalog")
LOG_QUEUE_PENDING = Gauge("xeye_search_log_queue_pending", "Search-log entries waiting in RAM")
LOG_SPOOL_ENTRIES = Gauge("xeye_search_log_spool_entries", "Search-log entries waiting on disk")
MODELS_LOADED = Gauge("xeye_search_embedding_models_loaded", "Embedding models resident in RAM")


def normalize_path(path: str) -> str:
    """``/v1/lists/42/search`` → ``/v1/lists/{id}/search`` (cardinalidad acotada)."""
    return _NUMERIC_SEGMENT.sub("/{id}", path) or "/"


def observe_request(method: str, path: str, status: int, seconds: float) -> None:
    label = normalize_path(path)
    HTTP_REQUESTS.labels(method, label, str(status)).inc()
    HTTP_DURATION.labels(method, label).observe(seconds)


def refresh_gauges(container) -> None:
    """Vuelca el estado del contenedor en los gauges (se llama en cada scrape y en /health)."""
    status = container.health()
    READY.set(1 if status["ready"] else 0)
    DEGRADED_STATE.set(1 if status["degraded"] else 0)
    CACHE_LISTS.set(len(container.cache))
    CACHE_BYTES.set(container.cache.total_bytes)
    CATALOG_LISTS.set(len(container.lists))
    CATALOG_API_KEYS.set(len(container.api_keys))
    LOG_QUEUE_PENDING.set(container.log_queue.pending)
    LOG_SPOOL_ENTRIES.set(container.log_queue.spooled)
    MODELS_LOADED.set(len(getattr(container.embedder, "loaded_models", ()) or ()))


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
