"""Configuración del servicio (variables de entorno / .env, vía pydantic-settings)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "xeye-search-service"
    service_version: str = "2.0.0"
    log_level: str = "INFO"

    # Integración con el backend (servidor a servidor).
    backend_url: str = "http://localhost:8000"
    internal_token: str = "dev-internal-token"
    backend_timeout_seconds: float = 30.0

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
