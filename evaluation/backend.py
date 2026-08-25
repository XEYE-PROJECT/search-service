"""Cliente mínimo del backend Java para etiquetar runs con el modelo en uso (opcional)."""

import json

import httpx


def model_label(model: str | None) -> str | None:
    """El campo model del training es opaco; si es el JSON del worker, usa embedding_model."""
    if not model:
        return None
    try:
        parsed = json.loads(model)
    except (json.JSONDecodeError, TypeError):
        return model
    if isinstance(parsed, dict) and parsed.get("embedding_model"):
        return str(parsed["embedding_model"])
    return model


class BackendError(Exception):
    pass


class BackendClient:
    def __init__(self, base_url: str, email: str, password: str, timeout: float = 15.0):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._login(email, password)

    def _login(self, email: str, password: str) -> None:
        response = self._client.post("/auth/login", json={"email": email, "password": password})
        if response.status_code != 200:
            raise BackendError(f"login fallido ({response.status_code}): {response.text[:200]}")
        token = response.json().get("token")
        if not token:
            raise BackendError("login sin token en la respuesta")
        self._client.headers["Authorization"] = f"Bearer {token}"

    def find_list_id(self, name: str) -> int:
        response = self._client.get("/lists")
        if response.status_code != 200:
            raise BackendError(f"GET /lists fallo ({response.status_code})")
        for item in response.json():
            if item.get("name") == name:
                return item["id"]
        raise BackendError(f"lista {name!r} no encontrada en el backend")

    def in_use_training(self, list_id: int) -> tuple[int, str] | None:
        """(training_id, model) del training activo de la lista, o None si no hay."""
        response = self._client.get(f"/lists/{list_id}/trainings")
        if response.status_code != 200:
            raise BackendError(f"GET /lists/{list_id}/trainings fallo ({response.status_code})")
        for training in response.json():
            if training.get("inUse"):
                return training["id"], model_label(training.get("model")) or "?"
        return None

    def close(self) -> None:
        self._client.close()
