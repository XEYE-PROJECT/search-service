"""Configuración de logging a stdout compartida por la app y los tests."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:  # ya configurado (uvicorn o una llamada previa)
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
