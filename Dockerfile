# XEYE search-service — imagen solo CPU, proceso sin privilegios.
# Base fijada por digest (Dependabot abre PR cuando cambia): builds reproducibles.
FROM python:3.14-slim@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

# Dependencias con versiones EXACTAS (requirements.lock, generado desde requirements.txt):
# dos builds del mismo commit dan la misma imagen. Primero torch de CPU (mucho más pequeño que
# el build CUDA por defecto; su versión también viene del lock), luego el resto.
COPY requirements.lock .
RUN pip install --no-cache-dir "$(grep -i '^torch==' requirements.lock)" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.lock

# Pre-descarga el modelo de embedding por defecto para que el arranque en frío no use red.
# El ARG hace también de default en runtime: ambos nombran siempre el mismo modelo.
ARG EMBEDDING_MODEL_DEFAULT=paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL_DEFAULT=${EMBEDDING_MODEL_DEFAULT}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL_DEFAULT}')"

# Modelos extra que los entrenamientos pueden usar (el ModelRegistry carga lo que nombre el
# entrenamiento, pero un modelo horneado no necesita red al consultar). Mantener en sincronía
# con el worker de entrenamiento y `xeye.training.embedding-models` del backend.
ARG EXTRA_EMBEDDING_MODELS="paraphrase-multilingual-mpnet-base-v2"
RUN for m in ${EXTRA_EMBEDDING_MODELS}; do \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$m')"; \
    done

COPY app ./app

# Usuario sin privilegios: dueño de la caché de modelos (solo lectura en runtime) y del
# directorio de datos, donde vive el spool en disco de los logs de búsqueda (montarlo como
# volumen para que sobreviva a los reinicios).
RUN useradd --system --uid 1001 --home-dir /app --shell /usr/sbin/nologin search \
    && mkdir -p /app/data/log-spool \
    && chown -R search:search /app
ENV LOG_SPOOL_DIR=/app/data/log-spool
USER search

# Commit desplegado, para etiquetar los eventos de Sentry (lo pasa el workflow con --build-arg).
ARG GIT_SHA=unknown
ENV SENTRY_RELEASE=${GIT_SHA}

# Detrás del proxy (Caddy) la IP real del cliente llega en X-Forwarded-For: uvicorn la usa como
# request.client (rate limit por IP, logs de auditoría) solo si el proxy está en
# FORWARDED_ALLOW_IPS. El contenedor solo es alcanzable desde la red docker (proxy y backend),
# así que se confía en cualquier origen; fuera de docker, restringir a la IP del proxy.
ENV FORWARDED_ALLOW_IPS=*

EXPOSE 8002
# Liveness (el proceso atiende). La readiness (/ready) la vigila el monitor externo: con el
# backend caído el servicio arranca igual y no debe reiniciarse en bucle.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=3)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--proxy-headers", "--no-access-log"]
