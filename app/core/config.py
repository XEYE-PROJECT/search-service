"""Configuración del servicio (variables de entorno / .env, vía pydantic-settings).

``ENVIRONMENT`` es ``production`` por defecto a propósito: sin un ``INTERNAL_TOKEN`` fuerte el
servicio NO arranca (fallo cerrado). Desarrollo debe declararse (``ENVIRONMENT=development``),
y solo entonces se admite el token de dev.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Valores que aparecen en .env.example / compose de desarrollo: nunca válidos en producción.
INSECURE_INTERNAL_TOKENS = frozenset({"dev-internal-token", "changeme", "change-me", "secret", "password"})
MIN_INTERNAL_TOKEN_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "xeye-search-service"
    service_version: str = "2.1.0"
    log_level: str = "INFO"
    #: json (una línea JSON por evento, con request_id) | text | auto (json en producción).
    log_format: Literal["auto", "json", "text"] = "auto"
    #: production (por defecto): token interno fuerte obligatorio, sin Swagger/OpenAPI.
    environment: Literal["development", "production"] = "production"
    #: ``GET /metrics`` (Prometheus). Solo alcanzable por la red docker: el proxy no lo publica.
    metrics_enabled: bool = True

    # Integración con el backend (servidor a servidor).
    backend_url: str = "http://localhost:8000"
    #: Secreto compartido con el backend (SEARCH_INTERNAL_TOKEN). SecretStr: no sale en logs/repr.
    internal_token: SecretStr = SecretStr("dev-internal-token")
    backend_timeout_seconds: float = 30.0
    backend_connect_timeout_seconds: float = 5.0
    #: Reintentos de las lecturas idempotentes al backend (bootstrap, datos de lista) ante
    #: fallos de red o 5xx, con backoff. Los pushes de logs los reintenta su propia cola.
    backend_retries: int = 2

    # Error tracking (Sentry). Vacío = desactivado.
    sentry_dsn: str = ""
    sentry_release: str = ""  # commit desplegado; lo fija el Dockerfile (SENTRY_RELEASE)

    # Embeddings. El modelo por lista llega del backend (el entrenamiento in_use);
    # este es solo el fallback para listas sin él.
    embedding_model_default: str = "paraphrase-multilingual-MiniLM-L12-v2"
    models_max_loaded: int = 2

    # Caché en RAM de los datos de búsqueda por lista (LRU por bytes reales).
    cache_max_bytes: int = 1_073_741_824  # 1 GiB
    #: Edad máxima de una lista cacheada antes de revalidarla contra el backend en segundo
    #: plano (stale-while-revalidate: mientras tanto se sirve lo cacheado). Red de seguridad
    #: para invalidaciones perdidas; 0 la desactiva (solo pushes/invalidate).
    cache_ttl_seconds: int = 3600
    # Hasta este tamaño se puntúa con coseno exacto (numpy); por encima, FAISS HNSW.
    exact_search_max_elements: int = 4096
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 96

    # Puntuación híbrida.
    search_text_weight: float = 0.25
    search_semantic_weight: float = 0.75
    score_override_threshold: float = 0.75

    # API pública.
    #: Búsquedas/minuto por USUARIO (todas sus API keys comparten el cupo). Es el "plan por
    #: defecto": el backend puede fijar otro valor por usuario (admin), que llega en el bootstrap
    #: y por PUT /v1/users/{id}/limits.
    rate_limit_per_minute: int = 60
    #: Peticiones/minuto por IP a la API pública, contadas ANTES de resolver la key: acota los
    #: intentos de adivinar keys (401) y a un cliente que reparta el tráfico entre varias keys.
    rate_limit_per_ip_per_minute: int = 300
    #: Tamaño máximo del body de la API pública (413 por encima). 16 KiB sobra: search_term
    #: tiene 500 caracteres de tope. La API interna no se limita (el push de índice lleva embeddings).
    max_request_bytes: int = 16_384
    #: Cabeceras Host aceptadas (TrustedHost): el dominio público, el nombre del contenedor
    #: (backend por la red docker) y localhost (healthcheck). "*" = cualquiera (solo desarrollo).
    allowed_hosts: str = "*"
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:5173"

    # Sincronización de catálogos.
    refresh_interval_seconds: int = 3600  # re-sync completo periódico; 0 lo desactiva
    refresh_min_interval_seconds: float = 30.0  # throttle de los refresh por miss

    # Cola de logs de búsqueda (por lotes, asíncrona). Persistente: lo que no se puede
    # entregar (backend caído, cola llena, apagado) se guarda en disco y se reenvía después.
    log_queue_max: int = 10_000
    log_batch_max: int = 50
    log_flush_seconds: float = 2.0
    log_push_retries: int = 3
    #: Directorio del spool en disco (JSONL). Vacío = sin spool (solo RAM, se descarta al fallar).
    log_spool_dir: str = "data/log-spool"
    #: Tope del spool; por encima se descartan los ficheros más antiguos (con log ERROR).
    log_spool_max_bytes: int = 52_428_800  # 50 MiB
    #: Cada cuánto se intenta reenviar lo spooleado.
    log_spool_replay_seconds: float = 30.0

    @property
    def log_format_resolved(self) -> str:
        if self.log_format == "auto":
            return "json" if self.is_production else "text"
        return self.log_format

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_host_list(self) -> list[str]:
        return [host.strip().lower() for host in self.allowed_hosts.split(",") if host.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def docs_enabled(self) -> bool:
        """Swagger/OpenAPI solo en desarrollo: en producción ni la API interna aparece en un esquema."""
        return not self.is_production

    @model_validator(mode="after")
    def _fail_fast_in_production(self) -> "Settings":
        """Fallo cerrado: en producción ningún valor de desarrollo pasa del arranque.

        Cada problema nombra la variable de entorno para que `docker compose logs` diga qué
        arreglar. Se acumulan todos en un único error en vez de fallar uno a uno.
        """
        if not self.is_production:
            return self
        problems: list[str] = []

        token = self.internal_token.get_secret_value().strip()
        if not token:
            problems.append("INTERNAL_TOKEN must be set in production (the internal API fails closed)")
        elif token.lower() in INSECURE_INTERNAL_TOKENS:
            problems.append(
                "INTERNAL_TOKEN is a known development value; generate one with `openssl rand -hex 32` "
                "(or set ENVIRONMENT=development on a dev machine)"
            )
        elif len(token) < MIN_INTERNAL_TOKEN_LENGTH:
            problems.append(f"INTERNAL_TOKEN must be at least {MIN_INTERNAL_TOKEN_LENGTH} characters in production")

        # La API pública la llama el navegador desde la consola: solo orígenes https reales.
        if not self.cors_origin_list:
            problems.append("CORS_ORIGINS must list the console origins (https://xeye.es,...)")
        for origin in self.cors_origin_list:
            if not origin.startswith("https://"):
                problems.append(f"CORS_ORIGINS must contain only https:// origins (found '{origin}')")
            elif is_local_host(origin):
                problems.append(f"CORS_ORIGINS must not contain localhost origins in production (found '{origin}')")

        # Dentro del contenedor, localhost es el propio contenedor: el backend nunca está ahí.
        if not self.backend_url.startswith(("http://", "https://")):
            problems.append(f"BACKEND_URL must be an http(s) URL (found '{self.backend_url}')")
        elif is_local_host(self.backend_url):
            problems.append(
                "BACKEND_URL must point to the backend over the docker network "
                f"(http://xeye-backend:8000), not localhost (found '{self.backend_url}')"
            )

        if self.rate_limit_per_minute <= 0:
            problems.append("RATE_LIMIT_PER_MINUTE must be > 0 in production")
        if self.rate_limit_per_ip_per_minute <= 0:
            problems.append("RATE_LIMIT_PER_IP_PER_MINUTE must be > 0 in production")
        if self.max_request_bytes <= 0:
            problems.append("MAX_REQUEST_BYTES must be > 0 in production")

        # Sin TrustedHost, una petición con Host arbitrario (p. ej. vía un proxy mal configurado)
        # llegaría a la app; el proxy real reenvía el Host público tal cual.
        hosts = self.allowed_host_list
        if not hosts or "*" in hosts:
            problems.append(
                "ALLOWED_HOSTS must list the accepted Host headers in production "
                "(search.xeye.es,search-service,localhost), not '*'"
            )

        if problems:
            raise ValueError("Unsafe production configuration:\n - " + "\n - ".join(problems))
        return self


def is_local_host(url: str) -> bool:
    """``localhost``, ``127.0.0.1``, ``0.0.0.0`` o ``[::1]`` como host de una URL u origen."""
    rest = url.strip().lower()
    if "://" in rest:
        rest = rest.split("://", 1)[1]
    for stop in "/?#":
        rest = rest.split(stop, 1)[0]
    host = rest.rsplit("@", 1)[-1]
    host = host[1:].split("]", 1)[0] if host.startswith("[") else host.split(":", 1)[0]
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".localhost")


@lru_cache
def get_settings() -> Settings:
    return Settings()
