# Prueba de carga (k6)

`search.js` mide `POST /api/v1/search` con [k6](https://k6.io): 20 usuarios virtuales durante
30 s rotando consultas realistas, con umbrales **p95 < 500 ms** y **< 1 % de errores**. No corre
en CI: se lanza a mano con la imagen `grafana/k6`, sin instalar nada.

Hace falta una **API key de prueba** cuyo usuario tenga una **lista pública ya entrenada** (el
script falla en `setup` si falta la key o si `/ready` no responde 200). Variables: `BASE_URL`,
`API_KEY`, `LIST_NAME` (por defecto `Productos`) y `THINK_S` (pausa por VU entre peticiones,
0.1 s por defecto → unas 150-200 peticiones/s).

**Cupos.** El servicio limita por usuario (`RATE_LIMIT_PER_MINUTE`, 60/min por defecto; un
admin puede subirlo por usuario desde la consola) y por IP (`RATE_LIMIT_PER_IP_PER_MINUTE`,
300/min). Una carga de este tamaño los supera en segundos: los 429 cuentan como fallo (y se ven
aparte en la métrica `rate_limited`). Antes de lanzar, sube el cupo del usuario de prueba y, en
el `.env` del servicio, `RATE_LIMIT_PER_IP_PER_MINUTE` (p. ej. `100000`); en producción, solo
durante la prueba y con una key que después se borre.

## Contra local

Con el stack de desarrollo levantado (backend en `:8000`, search-service en `:8002`):

```bash
docker run --rm -i --network host \
  -e BASE_URL=http://localhost:8002 -e API_KEY="$API_KEY" -e LIST_NAME=Productos \
  grafana/k6 run - < loadtest/search.js
```

Sin `--network host` (Docker Desktop / WSL) usa `BASE_URL=http://host.docker.internal:8002`.
Ten en cuenta que en local la primera búsqueda paga la carga perezosa de la lista y el modelo
de embeddings compite por la CPU con el propio k6: el p95 no es comparable con producción.

## Contra producción

Con una key de prueba (créala en la consola y bórrala al terminar) y el cupo subido:

```bash
docker run --rm -i \
  -e BASE_URL=https://search.xeye.es -e API_KEY="$API_KEY" -e LIST_NAME=Productos \
  grafana/k6 run - < loadtest/search.js
```

Pasa por Cloudflare y el proxy, así que la latencia incluye la red; `service_duration_ms`
(el `duration_ms` que devuelve el servicio) aísla el tiempo de búsqueda. Mira también
`degraded` (búsquedas servidas sin toda la calidad) y el panel de Prometheus del servicio
durante la prueba.

Para conservar el resultado, monta un volumen y exporta el resumen:
`-v "$PWD/loadtest/results:/out" … grafana/k6 run --summary-export=/out/summary.json - < …`
(`loadtest/results/` está en `.gitignore`).
