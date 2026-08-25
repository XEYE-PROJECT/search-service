# Evaluación de precisión del buscador

Paquete independiente de la app (no se despliega ni entra en pytest) para medir la calidad
de los entrenamientos lanzando consultas reales contra el endpoint público
`POST /api/v1/search` y calculando métricas de ranking. Pensado para la memoria del TFG.

## Requisitos

Python 3.11 con `httpx` (ya en `requirements.txt`) y `matplotlib` para las gráficas
(`requirements-dev.txt`). Ejecutar siempre desde la raíz de `search-service/` **con el venv
del proyecto** (`source .venv/bin/activate`, o prefijar los comandos con `.venv/bin/python`);
el Python del sistema no tiene las dependencias.

## Dataset

Un JSON por lista en `datasets/<slug-de-la-lista>.json`:

```json
[
  {"query": "cascos sin cables", "expected": "Auriculares inalámbricos Bluetooth con cancelación activa de ruido", "category": "sinonimo"}
]
```

- `expected` es el **texto exacto del elemento** (el matching normaliza tildes/mayúsculas).
- `category` es opcional (`sinonimo`, `intencion`, `marca`, `typo`, `atributo`) y desglosa
  las métricas por tipo de consulta.
- No uses consultas idénticas al texto del elemento: el servicio devuelve score 1.0 por
  match exacto y la consulta no mide nada (el script avisa).

Incluidos: `datasets/productos.json` (35 consultas sobre la lista de productos de ejemplo),
`datasets/actividades.json` (55 consultas sobre las 250 actividades de negocio de
`activities_cyber.json`, redactadas como un usuario describiría su profesión) y
`datasets/cpv.json` (61 consultas sobre el vocabulario CPV de contratación pública —
`training-service/storage/examples/codigos_cpv_reducido.json`, 4 727 códigos —, redactadas
como un técnico describiría lo que quiere licitar). El nombre del fichero es el *slug* del
nombre de la lista, así que la lista del CPV debe llamarse `CPV` o hay que pasar `--dataset`.

## Evaluar un run

```bash
# contra producción
python -m evaluation.evaluate --list Productos --api-key xeye_... \
    --search-url https://search.xeye.es --label nombre-del-modelo

# etiquetado automático del modelo/training en uso preguntando al backend
python -m evaluation.evaluate --list Productos --api-key xeye_... \
    --search-url https://search.xeye.es \
    --backend-url https://backend.xeye.es --email tu@correo --password ...
```

Flags útiles: `--allow-private` (listas privadas propias), `--limit` (resultados por
consulta, def. 50), `--dataset` (fichero alternativo), `--qpm` (def. 55; el servicio corta a
60 req/min por API key y el script espacia y reintenta los 429). La API key puede ir en la
variable de entorno `XEYE_API_KEY`.

Métricas del run: **top-1 accuracy** (principal), recall@1/3/5/10, MRR, rango medio del
esperado, no encontrados, latencia media/p95 del servicio, % de aciertos ganados por la
puntuación semántica y desglose por categoría. Cada run se guarda en
`results/<lista>/<timestamp>_<etiqueta>.json` (con el top-5 de cada consulta para revisar
fallos) y se añade una fila a `results/<lista>/runs.csv`.

## Comparar runs (p. ej. distintos modelos de embedding)

Entrena la lista con cada modelo, lanza un `evaluate` por modelo y después:

```bash
python -m evaluation.compare --list Productos
```

Genera en `results/<lista>/`: `comparison.csv`, `comparison.md` (tabla lista para pegar en
la memoria) y `charts/{metrics,latency,ranks}.png`.
