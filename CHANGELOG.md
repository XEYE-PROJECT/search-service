# Changelog

Formato [Keep a Changelog](https://keepachangelog.com/es/1.1.0/); versiones [SemVer](https://semver.org/lang/es/).
Las entradas nuevas van en "Unreleased"; `bash release.sh X.Y.Z` las convierte en una versión y
crea el tag que publica la imagen `ghcr.io/xeye-project/search-service:vX.Y.Z` y la GitHub Release.

## [Unreleased]

### Añadido
- Microservicio de búsqueda reconstruido (FastAPI, hexagonal): catálogo en memoria sincronizado
  con el backend (bootstrap por páginas + notificaciones), búsqueda híbrida texto + embeddings con
  alineación de filas por id de elemento, degradación explícita (`degraded`, `X-Search-Degraded`).
- API pública con API keys hasheadas y cupos por usuario e IP; búsqueda de consola reenviada por
  el backend; logs de búsqueda en cola con spool a disco.
- Producción segura por defecto: token interno fuerte obligatorio, Swagger apagado, `/ready`,
  cabeceras de seguridad, Sentry, métricas Prometheus.
- CI: gitleaks, ruff + mypy, pytest (incluidos tests de contrato con el backend), imagen
  escaneada con Trivy, despliegue por SHA con rollback; prueba de carga k6 en `loadtest/`.

[Unreleased]: https://github.com/XEYE-PROJECT/search-service/compare/master...HEAD
