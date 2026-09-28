"""Configuración de logging a stdout compartida por la app y los tests.

Dos formatos: ``text`` (legible, desarrollo) y ``json`` (una línea JSON por evento, con
``request_id`` del contexto, para agregarlos y filtrarlos en producción). Los loggers de
uvicorn se reformatean igual; su access log se apaga porque ``RequestContextMiddleware``
escribe el nuestro (con request id, prefijo de key y duración).
"""

from __future__ import annotations

import json
import logging
import sys
import time

from app.core.request_context import current_request_id

_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """``{"ts","level","logger","msg","request_id"?, ...extra}``; las excepciones van en ``exc``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = current_request_id()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s - %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = current_request_id()
        return f"{line} [rid={request_id}]" if request_id else line


def make_formatter(fmt: str) -> logging.Formatter:
    return JsonFormatter() if fmt == "json" else TextFormatter()


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    root = logging.getLogger()
    formatter = make_formatter(fmt)
    if root.handlers:  # ya configurado (uvicorn o una llamada previa): solo reformatear
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error"):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(formatter)
    # Nuestro middleware escribe el access log (con request id); el de uvicorn duplicaría.
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.setLevel(logging.WARNING)
