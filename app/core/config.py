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
    service_version: str = "2.0.0"
    log_level: str = "INFO"
    #: production (por defecto): token interno fuerte obligatorio, sin Swagger/OpenAPI.
    environment: Literal["development", "production"] = "production"

    # Integración con el backend (servidor a servidor).
    backend_url: str = "http://localhost:8000"
    #: Secreto compartido con el backend (SEARCH_INTERNAL_TOKEN). SecretStr: no sale en logs/repr.
    internal_token: SecretStr = SecretStr("dev-internal-token")
    backend_timeout_seconds: float = 30.0

    # Error tracking (Sentry). Vacío = desactivado.
    sentry_dsn: str = ""
    sentry_release: str = ""  # commit desplegado; lo fija el Dockerfile (SENTRY_RELEASE)

    # Embeddings. El modelo por lista llega del backend (el entrenamiento in_use);
    # este es solo el fallback para listas sin él.
    embedding_model_default: str = "paraphrase-multilingual-MiniLM-L12-v2"
    models_max_loaded: int = 2

    # Caché en RAM de los datos de búsqueda por lista (LRU por bytes reales).
    cache_max_bytes: int = 1_073_741_824  # 1 GiB
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
    rate_limit_per_minute: int = 60
    cors_origins: str = (
        "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:5173"
    )

    # Sincronización de catálogos.
    refresh_interval_seconds: int = 3600  # re-sync completo periódico; 0 lo desactiva
    refresh_min_interval_seconds: float = 30.0  # throttle de los refresh por miss

    # Cola de logs de búsqueda (por lotes, asíncrona, best-effort).
    log_queue_max: int = 10_000
    log_batch_max: int = 50
    log_flush_seconds: float = 2.0
    log_push_retries: int = 3

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def docs_enabled(self) -> bool:
        """Swagger/OpenAPI solo en desarrollo: en producción ni la API interna aparece en un esquema."""
        return not self.is_production

    @model_validator(mode="after")
    def _require_strong_internal_token_in_production(self) -> "Settings":
        if not self.is_production:
            return self
        token = self.internal_token.get_secret_value().strip()
        if not token:
            raise ValueError("INTERNAL_TOKEN must be set in production (the internal API fails closed)")
        if token.lower() in INSECURE_INTERNAL_TOKENS:
            raise ValueError(
                "INTERNAL_TOKEN is a known development value; generate one with `openssl rand -hex 32` "
                "(or set ENVIRONMENT=development on a dev machine)"
            )
        if len(token) < MIN_INTERNAL_TOKEN_LENGTH:
            raise ValueError(
                f"INTERNAL_TOKEN must be at least {MIN_INTERNAL_TOKEN_LENGTH} characters in production"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
