"""Errores de aplicación. La capa web los mapea al sobre JSON público
{"error": ..., "detail": ...} — la forma que espera el manejo de errores del frontend."""


class ApiException(Exception):
    def __init__(self, status_code: int, error: str, detail: str | None = None) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.detail = detail


class MissingApiKeyError(ApiException):
    def __init__(self) -> None:
        super().__init__(401, "API key requerida", "Incluye la cabecera X-API-Key")


class InvalidApiKeyError(ApiException):
    def __init__(self) -> None:
        super().__init__(401, "API key inválida", "La API key no existe o fue revocada")


class RateLimitedError(ApiException):
    def __init__(self, limit_per_minute: int) -> None:
        super().__init__(429, "Rate limit alcanzado", f"Máximo {limit_per_minute} peticiones por minuto")


class ListNotFoundError(ApiException):
    def __init__(self, list_name: str) -> None:
        super().__init__(404, "Lista no encontrada", f"No existe una lista '{list_name}' para este usuario")


class ListNotPublicError(ApiException):
    def __init__(self, list_name: str) -> None:
        super().__init__(403, "Lista no pública", f"La lista '{list_name}' no es pública")


class BackendUnavailableError(ApiException):
    def __init__(self) -> None:
        super().__init__(503, "Backend no disponible", "No se pudieron cargar los datos de la lista")


class InvalidInternalTokenError(ApiException):
    def __init__(self) -> None:
        super().__init__(403, "Token interno inválido", "Cabecera X-Internal-Token incorrecta")
