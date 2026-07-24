"""Limitador de ventana fija en RAM (por API key). Monoinstancia a propósito: el servicio
ya guarda todo el estado de búsqueda en RAM local, así que uno distribuido no aportaría nada."""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._counts: dict[str, tuple[int, int]] = {}  # clave -> (ventana, contador)

    def allow(self, key: str) -> bool:
        if self._limit <= 0:  # desactivado
            return True
        window = int(time.time() // 60)
        entry_window, count = self._counts.get(key, (window, 0))
        if entry_window != window:
            count = 0
        count += 1
        self._counts[key] = (window, count)
        if len(self._counts) > 10_000:
            self._counts = {k: v for k, v in self._counts.items() if v[0] == window}
        return count <= self._limit
