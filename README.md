# XEYE search-service

Microservicio de búsqueda semántica de XEYE, reescrito desde cero (sustituye a
`../XEYE-search-service`). Python 3.11 + FastAPI, **sin Redis ni workers externos**:
todo el estado vive en RAM y se reconstruye desde el backend Java.

## Arquitectura

Hexagonal ligera, dependencias hacia dentro:

```
app/
  domain/          # puro: modelos, normalización de texto, combinación de scores
  application/     # casos de uso y puertos (Protocols): búsqueda, catálogos, datos de lista
  infrastructure/  # adaptadores: cliente HTTP del backend, FAISS/numpy, caché LRU,
                   # cola de logs, rate limiter, capa web (FastAPI)
  core/            # settings (pydantic-settings) y wiring (container.py)
```

### Flujo de datos

- **Arranque**: `GET {backend}/internal/search/bootstrap` carga *solo* los catálogos
  ligeros (los **hashes SHA-256** de las API keys —aquí nunca vive una key en claro—,
  metadatos de listas y los modelos de embeddings disponibles). Si el backend está caído se reintenta con backoff; el servicio arranca
  igualmente. Tras el bootstrap se **precalientan todos** los modelos disponibles (y el
  default desde el primer instante), de modo que la primera búsqueda con cualquier
  modelo no paga su carga/descarga; el re-sync periódico precalienta modelos nuevos.
- **Lazy loading**: los datos pesados de una lista (elementos + embeddings + índice) se
  cargan en el **primer** search de esa lista (`GET /internal/search/lists/{id}`) y se
  cachean. Locks por `list_id` garantizan una única carga concurrente (single-flight).
- **Caché RAM**: LRU acotada por **bytes reales** (`CACHE_MAX_BYTES`, 1 GiB por defecto):
  matriz de embeddings + estructuras FAISS + textos. Al superarse el límite se expulsan
  las listas menos usadas.
- **Push del backend**: al completarse un entrenamiento el backend hace
  `POST /v1/lists/{id}/index` (contrato `SearchIndexCommand`, ahora incluye `model`) y la
  caché se rellena en caliente. Cambios de nombre/visibilidad, borrados, API keys y
  usuarios llegan por los endpoints `/v1/*` de notificación; si alguna notificación se
  pierde, el lazy-reload y el re-sync periódico (`REFRESH_INTERVAL_SECONDS`) lo curan.
- **Modelo dinámico**: el modelo de embeddings ya no es un `.env` fijo — llega por lista
  desde el entrenamiento `in_use` (campo `model` del push / del lazy-load) y los modelos
  de sentence-transformers viven en su propia LRU (`MODELS_MAX_LOADED`; súbelo si el
  backend anuncia más modelos que slots). `EMBEDDING_MODEL_DEFAULT` es solo el fallback.
- **Logs de búsqueda**: cada llamada pública se encola en una `asyncio.Queue` y un
  consumidor la envía por lotes a `POST {backend}/internal/search/logs` (tabla
  `searches` de MariaDB, gestionada por el backend). Nunca bloquea la búsqueda.

### Scoring (híbrido texto + semántico)

1. Texto: rapidfuzz sobre texto normalizado — `0.4*ratio + 0.6*partial_ratio`.
2. Semántico: coseno entre el embedding de la query (modelo de la lista) y los
   embeddings entrenados (normalizados L2).
   - Los vectores se **alinean por id de elemento** (`trainedElementIds`, capturados por
     el backend al lanzar el entrenamiento): si la lista cambió durante el training, los
     vectores de elementos borrados se descartan y los elementos nuevos puntúan solo por
     texto hasta el siguiente retrain — nunca se sirven vectores desalineados.
   - Listas ≤ `EXACT_SEARCH_MAX_ELEMENTS` (4096): fuerza bruta numpy — **exacto**, 100%
     recall, más rápido que HNSW a esa escala.
   - Listas grandes: **FAISS `IndexHNSWFlat`** (inner product) genera candidatos top-K,
     unidos al top-K de texto (conjunto acotado); todos con coseno exacto, y la
     combinación/ordenación corre en un worker thread para no bloquear el event loop.
3. Combinación: match exacto normalizado → 1.0; score ≥ 0.75 → domina ese score; si no,
   `0.25*texto + 0.75*semántico`; sin embeddings → solo texto (sin capar por el peso).

## API

Sondas sin auth: `GET /health` (liveness, docker) y `GET /ready` (readiness: 503 hasta que el
bootstrap contra el backend ha terminado; es la que vigila el monitor de uptime).

Pública (cabecera `X-API-Key`, rate limit 60/min por key, CORS habilitado; la key se hashea y
se busca en el catálogo):

- `POST /api/v1/search` — `{list_name, search_term, limit?, session?, include_score_breakdown?, register_log?, allow_private?}`
  → `{success, results:[{item, score, params, text_score?, semantic_score?}], total_results, search_term, list_name, duration_ms}`.
  Solo listas **públicas** del dueño de la key. Errores: `{error, detail}` con 401/403/404/429/503.
- `POST /api/v1/target` — `{list_name, target_term, session}` → `{success}` (solo auditoría).

Interna (cabecera `X-Internal-Token`, comparada en tiempo constante; solo el backend):

- `POST /v1/lists/{id}/index` · `PUT /v1/lists/{id}/meta` · `DELETE /v1/lists/{id}` ·
  `POST /v1/lists/{id}/invalidate` · `PUT|DELETE /v1/api-keys/{id}` ·
  `DELETE /v1/users/{id}` · `POST /v1/refresh` · `GET /v1/health`

## Ejecutar

```bash
# desarrollo en el host (backend en localhost:8000)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8002 --reload

# tests (no necesitan torch/sentence-transformers)
pytest

# docker (red compartida xeye-network)
docker compose up --build
```

Variables en `.env.example`; referencia completa con lo **obligatorio en producción** en
[CONFIG.md](CONFIG.md). En el backend: `SEARCH_PROVIDER=http`,
`SEARCH_SERVICE_URL=http://localhost:8002` (o `http://xeye-search-service:8002` en
docker) y el mismo `SEARCH_INTERNAL_TOKEN`.

## Producción (fallo cerrado)

`ENVIRONMENT=production` es el valor por defecto: el servicio **no arranca** si
`INTERNAL_TOKEN` está vacío, es el de desarrollo o tiene menos de 32 caracteres, si algún
`CORS_ORIGINS` no es `https://` o es `localhost`, si `BACKEND_URL` apunta a `localhost` (dentro
del contenedor sería él mismo) o si `RATE_LIMIT_PER_MINUTE` es 0; el error de arranque lista
todos los problemas nombrando la variable. Tampoco expone `/docs`, `/redoc` ni `/openapi.json`. En una máquina de desarrollo hay que declarar
`ENVIRONMENT=development`. `SENTRY_DSN` activa el error tracking (las cabeceras `X-API-Key`
y `X-Internal-Token` se eliminan de los eventos). El proxy solo publica `/api/v1/*`, `/`,
`/health` y `/ready`.

Para probar la API a mano hay una colección de [Bruno](https://www.usebruno.com/) en
`bruno/` (ábrela con "Open Collection" y selecciona el entorno `local`). `apiKey` e
`internalToken` son *secret vars* del entorno: rellénalas en Bruno, **nunca en un `.bru`**
(quedaría en git).
