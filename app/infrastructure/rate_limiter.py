"""Limitador de ventana fija (60 s) en RAM, con límite por clave. Monoinstancia a propósito:
el servicio ya guarda todo el estado de búsqueda en RAM local, así que uno distribuido no
aportaría nada. Se usa dos veces: por usuario (todas sus API keys comparten el cupo, con
límite propio si el backend lo fijó) y por IP (antes de resolver la key)."""

from __future__ import annotations

import time
from dataclasses import dataclass

WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int  # segundos hasta que se abre la ventana siguiente

    def headers(self) -> dict[str, str]:
        """Cabeceras informativas ``X-RateLimit-*`` (en 200 y en 429)."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_seconds),
        }


class RateLimiter:
    def __init__(self, default_limit_per_minute: int) -> None:
        self._default = default_limit_per_minute
        self._counts: dict[str, tuple[int, int]] = {}  # clave -> (ventana, contador)

    def check(self, key: str, limit: int | None = None) -> RateLimitDecision:
        """Cuenta un acceso de ``key`` contra ``limit`` (o el por defecto). ``limit <= 0`` desactiva."""
        effective = self._default if limit is None else limit
        now = time.time()
        window = int(now // WINDOW_SECONDS)
        reset = int((window + 1) * WINDOW_SECONDS - now) or 1
        if effective <= 0:
            return RateLimitDecision(True, 0, 0, reset)
        entry_window, count = self._counts.get(key, (window, 0))
        if entry_window != window:
            count = 0
        count += 1
        self._counts[key] = (window, count)
        if len(self._counts) > 10_000:
            self._counts = {k: v for k, v in self._counts.items() if v[0] == window}
        return RateLimitDecision(count <= effective, effective, max(0, effective - count), reset)

    def allow(self, key: str, limit: int | None = None) -> bool:
        return self.check(key, limit).allowed
