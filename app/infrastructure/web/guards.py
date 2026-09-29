"""Guardas ASGI previas al enrutado: cabecera Host permitida y tamaño del body público.

Es un middleware ASGI puro (no ``BaseHTTPMiddleware``) para no copiar el body ni tocar el
streaming. Responde con el mismo sobre JSON de error que el resto de la API.

- **TrustedHost**: solo los ``Host`` de ``ALLOWED_HOSTS`` (sin puerto; ``*`` = cualquiera). El
  proxy reenvía el Host público tal cual; el backend usa el nombre del contenedor y el
  healthcheck ``localhost``.
- **Tamaño**: los bodies de ``/api/v1/*`` se limitan a ``MAX_REQUEST_BYTES`` (413). Se comprueba
  ``Content-Length`` antes de leer nada y, si el cliente envía por trozos sin declararlo, se
  cuenta lo recibido y se corta al pasarse. La API interna no se limita: el push de índice
  lleva los embeddings de la lista.
"""

from __future__ import annotations

import json

from app.application.errors import InvalidHostError, RequestTooLargeError

_PUBLIC_PREFIX = "/api/v1/"


class _BodyTooLarge(Exception):
    pass


class RequestGuardMiddleware:
    def __init__(self, app, *, allowed_hosts: list[str], max_public_body_bytes: int) -> None:
        self._app = app
        self._hosts = {h.lower() for h in allowed_hosts}
        self._any_host = not self._hosts or "*" in self._hosts
        self._max_bytes = max_public_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not self._any_host and not self._host_allowed(scope):
            await _send_error(send, InvalidHostError())
            return

        path: str = scope.get("path", "")
        if not path.startswith(_PUBLIC_PREFIX) or self._max_bytes <= 0:
            await self._app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self._max_bytes:
            await _send_error(send, RequestTooLargeError(self._max_bytes))
            return

        received = 0
        response_started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await _send_error(send, RequestTooLargeError(self._max_bytes))

    def _host_allowed(self, scope) -> bool:
        host = ""
        for name, value in scope.get("headers", []):
            if name == b"host":
                host = value.decode("latin-1")
                break
        host = host.strip().lower()
        # Sin puerto; un literal IPv6 conserva sus corchetes.
        host = host.split("]", 1)[0] + "]" if host.startswith("[") else host.split(":", 1)[0]
        return host in self._hosts


def _content_length(scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_error(send, exc) -> None:
    body = json.dumps(exc.to_body()).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers += [(k.lower().encode(), v.encode()) for k, v in exc.headers.items()]
    await send({"type": "http.response.start", "status": exc.status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body})
