# XEYE search-service — imagen solo CPU.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

# Primero torch de CPU (mucho más pequeño que el build CUDA por defecto), luego el resto.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

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

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
