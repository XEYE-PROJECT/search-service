"""Errores de aplicación. La capa web los serializa TODOS (también los de validación, las rutas
inexistentes y los 500) con el mismo sobre JSON que usa el backend Java::

    {"status": 401, "error": "Unauthorized", "code": "API_KEY_INVALID",
     "message": "The API key does not exist or was revoked", "details": {...}?}

``code`` es el identificador estable para máquinas (los clientes hacen switch sobre él);
``message`` es texto en inglés para humanos y puede cambiar. Nunca se incluyen valores de
cabeceras ni secretos en el mensaje.
"""

from __future__ import annotations

from http import HTTPStatus


class ApiException(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers or {}

    def to_body(self) -> dict:
        return error_body(self.status_code, self.code, self.message, self.details)


def error_body(status_code: int, code: str, message: str, details: dict | None = None) -> dict:
    body = {
        "status": status_code,
        "error": HTTPStatus(status_code).phrase,
        "code": code,
        "message": message,
    }
    if details:
        body["details"] = details
    return body


class MissingApiKeyError(ApiException):
    def __init__(self) -> None:
        super().__init__(401, "API_KEY_MISSING", "Missing X-API-Key header")


class InvalidApiKeyError(ApiException):
    def __init__(self) -> None:
        super().__init__(401, "API_KEY_INVALID", "The API key does not exist or was revoked")


class RateLimitedError(ApiException):
    """429 con ``Retry-After`` y las cabeceras ``X-RateLimit-*`` de la ventana agotada."""

    def __init__(self, limit_per_minute: int, retry_after_seconds: int, scope: str = "account") -> None:
        retry_after = max(1, retry_after_seconds)
        super().__init__(
            429,
            "RATE_LIMITED",
            f"Rate limit of {limit_per_minute} requests per minute per {scope} exceeded; "
            f"retry in {retry_after} seconds",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            },
        )


class ListNotFoundError(ApiException):
    def __init__(self, list_name: str | int) -> None:
        super().__init__(404, "LIST_NOT_FOUND", f"No list '{list_name}' exists for this account")


class ListNotPublicError(ApiException):
    def __init__(self, list_name: str) -> None:
        super().__init__(
            403,
            "LIST_NOT_PUBLIC",
            f"List '{list_name}' is private; the public API only serves public lists",
        )


class BackendUnavailableError(ApiException):
    def __init__(self) -> None:
        super().__init__(503, "BACKEND_UNAVAILABLE", "The list data could not be loaded from the backend")


class InvalidInternalTokenError(ApiException):
    def __init__(self) -> None:
        super().__init__(403, "INTERNAL_TOKEN_INVALID", "Missing or invalid X-Internal-Token header")


class RequestTooLargeError(ApiException):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(413, "REQUEST_TOO_LARGE", f"Request body exceeds {max_bytes} bytes")


class InvalidHostError(ApiException):
    def __init__(self) -> None:
        super().__init__(400, "INVALID_HOST", "Host header not allowed")
