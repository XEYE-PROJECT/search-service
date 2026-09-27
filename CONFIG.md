# Referencia de configuración — search-service

Todas las variables (pydantic-settings, `app/core/config.py`), con lo que es **obligatorio en
producción**. `ENVIRONMENT=production` es el valor por defecto a propósito: con él el servicio
**no arranca** si alguna comprobación falla, y el error lista todas las variables afectadas.
En una máquina de desarrollo hay que declarar `ENVIRONMENT=development`.

Leyenda: **Prod** = obligatorio en producción · 🔑 = secreto (nunca en git ni en logs; `SecretStr`) ·
Valida = lo comprueba el validador de `Settings` al arrancar en `production`.

## Entorno

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `ENVIRONMENT` | `development` o `production` | `production` | **production** | — |
| `LOG_LEVEL` | Nivel de log | `INFO` | opcional | — |
| `SERVICE_NAME` / `SERVICE_VERSION` | Lo que anuncia `/` y OpenAPI | `xeye-search-service` / `2.0.0` | opcional | — |

## Integración con el backend (servidor a servidor)

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `BACKEND_URL` | URL del backend por la red docker (`http://xeye-backend:8000`) | `http://localhost:8000` | **sí** | http(s), sin localhost |
| `DOCKER_BACKEND_URL` | Solo `docker-compose.yml` de dev: la URL vista desde el contenedor | `http://xeye-java-backend:8000` | — | — |
| `INTERNAL_TOKEN` 🔑 | Secreto compartido `X-Internal-Token` (= `SEARCH_INTERNAL_TOKEN` del backend) | `dev-internal-token` | **sí** | ≥ 32 chars, no dev |
| `BACKEND_TIMEOUT_SECONDS` | Timeout de las llamadas al backend | `30` | opcional | — |

## API pública (la llama el navegador con `X-API-Key`)

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `CORS_ORIGINS` | Orígenes exactos de la consola (mismos que el backend) | localhost:3000… | **sí** | solo `https://`, sin localhost |
| `ALLOWED_HOSTS` | Cabeceras `Host` aceptadas: dominio público, nombre del contenedor (backend) y `localhost` (healthcheck). `*` = cualquiera | `*` | **sí** | lista explícita, sin `*` |
| `RATE_LIMIT_PER_MINUTE` | Búsquedas/min por **usuario** (todas sus keys + consola); plan por defecto, un admin puede fijar otro por usuario | `60` | **> 0** | > 0 |
| `RATE_LIMIT_PER_IP_PER_MINUTE` | Peticiones/min por IP a la API pública, antes de resolver la key | `300` | **> 0** | > 0 |
| `MAX_REQUEST_BYTES` | Tamaño máximo del body en `/api/v1/*` (413 por encima; la API interna no se limita) | `16384` | **> 0** | > 0 |
| `FORWARDED_ALLOW_IPS` | Proxies de los que uvicorn acepta `X-Forwarded-For` (IP real para el límite por IP y el log de auditoría) | `*` (Dockerfile) | opcional | — |
| `SEARCH_SERVICE_PORT` | Solo compose de dev: puerto publicado en el host | `8002` | — | — |

## Embeddings y caché

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `EMBEDDING_MODEL_DEFAULT` | Fallback para listas sin modelo entrenado (horneado en la imagen) | MiniLM | recomendado | — |
| `MODELS_MAX_LOADED` | Modelos sentence-transformers en RAM a la vez (LRU) | `2` | `1` en el VPS | — |
| `CACHE_MAX_BYTES` | Tope de la caché LRU de listas | 1 GiB | ajustar a la RAM | — |
| `EXACT_SEARCH_MAX_ELEMENTS` | Hasta aquí coseno exacto; por encima FAISS HNSW | `4096` | opcional | — |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` / `HNSW_EF_SEARCH` | Parámetros HNSW | `32` / `200` / `96` | opcional | — |
| `SEARCH_TEXT_WEIGHT` / `SEARCH_SEMANTIC_WEIGHT` / `SCORE_OVERRIDE_THRESHOLD` | Puntuación híbrida | `0.25` / `0.75` / `0.75` | opcional | — |

## Sincronización y logs de búsqueda

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `REFRESH_INTERVAL_SECONDS` | Re-sync completo periódico (0 desactiva) | `3600` | opcional | — |
| `REFRESH_MIN_INTERVAL_SECONDS` | Throttle del re-sync por miss | `30` | opcional | — |
| `LOG_QUEUE_MAX` / `LOG_BATCH_MAX` / `LOG_FLUSH_SECONDS` / `LOG_PUSH_RETRIES` | Cola de logs hacia el backend | `10000` / `50` / `2.0` / `3` | opcional | — |

## Observabilidad

| Variable | Descripción | Default | Prod | Valida |
|---|---|---|---|---|
| `SENTRY_DSN` | DSN del proyecto `xeye-search-service` (vacío = desactivado) | vacío | recomendado | — |
| `SENTRY_RELEASE` | Commit desplegado; lo fija el Dockerfile (`GIT_SHA`) | vacío | automático | — |

## Qué NO sale nunca en los logs

- `INTERNAL_TOKEN` es `SecretStr`: no aparece en `repr(settings)` ni en trazas.
- Las cabeceras `X-API-Key`, `X-Internal-Token`, `Authorization` y `Cookie` se eliminan de los
  eventos de Sentry (`_scrub_sensitive_headers`).
- Las API keys se manejan por hash (`hash_api_key`); el valor crudo solo vive en la petición.
- El logger `xeye.audit` (401/403/429) escribe IP, ruta y los 12 primeros caracteres de la key
  (el mismo prefijo que muestra la consola), nunca la key completa ni el token interno.

## Comprobación rápida

```bash
docker run --rm -e ENVIRONMENT=production -e INTERNAL_TOKEN=dev-internal-token ghcr.io/xeye-project/search-service:latest
# -> ValidationError "Unsafe production configuration" listando INTERNAL_TOKEN, CORS_ORIGINS, BACKEND_URL y ALLOWED_HOSTS
docker run --rm --env-file env/search.env ghcr.io/xeye-project/search-service:latest   # arranca
```
