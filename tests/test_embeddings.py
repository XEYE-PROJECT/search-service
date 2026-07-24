import threading

import numpy as np

from app.infrastructure.embeddings import (
    ModelLoadError,
    ModelRegistry,
    decode_embeddings,
    parse_model_name,
)
from tests.conftest import embeddings_b64


def test_decode_valid_matrix():
    matrix = np.random.rand(3, 8).astype(np.float32)
    decoded = decode_embeddings(embeddings_b64(matrix), expected_rows=3)
    assert decoded is not None
    assert decoded.shape == (3, 8)
    np.testing.assert_allclose(decoded, matrix, rtol=1e-6)


def test_decode_rejects_row_mismatch():
    matrix = np.random.rand(3, 8).astype(np.float32)
    assert decode_embeddings(embeddings_b64(matrix), expected_rows=4) is None


def test_decode_rejects_garbage_and_none():
    assert decode_embeddings("bm90LW5weQ==", expected_rows=1) is None  # base64("not-npy")
    assert decode_embeddings(None, expected_rows=1) is None
    assert decode_embeddings("", expected_rows=1) is None


def test_parse_model_name():
    raw = '{"embedding_model": "paraphrase-multilingual-MiniLM-L12-v2", "llm_model": null, "list_id": 7}'
    assert parse_model_name(raw) == "paraphrase-multilingual-MiniLM-L12-v2"


def test_parse_model_name_rejects_mock_and_garbage():
    assert parse_model_name('{"embedding_model": "mock", "llm_model": null, "list_id": 1}') is None
    assert parse_model_name("not json") is None
    assert parse_model_name(None) is None
    assert parse_model_name('"just a string"') is None


def _registry_with_fake_loader(loaded: list[str]) -> ModelRegistry:
    registry = ModelRegistry("default-model", max_loaded=4)

    async def fake_get(name: str):
        loaded.append(name)
        registry._models[name] = (object(), threading.Lock())
        return registry._models[name]

    registry._get = fake_get
    return registry


async def test_preload_models_dedups_and_skips_resident_and_mock():
    loaded: list[str] = []
    registry = _registry_with_fake_loader(loaded)
    await registry.preload_models(["model-a", "model-a", "", "mock", "model-b"])
    assert loaded == ["model-a", "model-b"]
    # Un re-precalentamiento (refresh periódico) solo carga lo que no está residente.
    await registry.preload_models(["model-a", "model-b", "model-c"])
    assert loaded == ["model-a", "model-b", "model-c"]


async def test_preload_models_survives_per_model_failures():
    loaded: list[str] = []
    registry = _registry_with_fake_loader(loaded)
    real_get = registry._get

    async def flaky_get(name: str):
        if name == "broken":
            raise ModelLoadError(name)
        return await real_get(name)

    registry._get = flaky_get
    await registry.preload_models(["broken", "model-a"])  # no debe lanzar
    assert loaded == ["model-a"]
