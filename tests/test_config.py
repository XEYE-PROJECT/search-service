"""Fallo cerrado en producción: sin un INTERNAL_TOKEN fuerte el servicio no arranca."""

import pytest
from pydantic import ValidationError

from app.core.config import MIN_INTERNAL_TOKEN_LENGTH, Settings
from app.core.security import constant_time_equals, hash_api_key


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_production_is_the_default_environment():
    # (conftest fija ENVIRONMENT=development en el proceso; el default del modelo es production)
    assert Settings.model_fields["environment"].default == "production"
    with pytest.raises(ValidationError, match="INTERNAL_TOKEN"):
        settings(environment="production")  # token por defecto (dev) -> no arranca
    assert settings(environment="production", internal_token="p" * MIN_INTERNAL_TOKEN_LENGTH).is_production


@pytest.mark.parametrize("token", ["", "   ", "dev-internal-token", "changeme", "SECRET", "short-token"])
def test_production_rejects_blank_dev_or_short_tokens(token):
    with pytest.raises(ValidationError, match="INTERNAL_TOKEN"):
        settings(environment="production", internal_token=token)


def test_production_accepts_a_strong_token_and_hides_it():
    cfg = settings(environment="production", internal_token="a" * 48)
    assert cfg.internal_token.get_secret_value() == "a" * 48
    assert "a" * 48 not in repr(cfg)
    assert not cfg.docs_enabled


def test_development_accepts_the_dev_token_and_enables_docs():
    cfg = settings(environment="development")
    assert cfg.internal_token.get_secret_value() == "dev-internal-token"
    assert cfg.docs_enabled


def test_hash_api_key_matches_the_backend_format():
    # Vector conocido de SHA-256("abc"); mismo formato que ApiKeyHasher (Java) y SHA2() en V6.
    assert hash_api_key("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert len(hash_api_key("xeye_whatever")) == 64


def test_constant_time_equals_fails_closed():
    assert constant_time_equals("a-secret-value", "a-secret-value")
    assert not constant_time_equals("a-secret-value", "a-secret-valuf")
    assert not constant_time_equals("a-secret-value", "a-secret")
    assert not constant_time_equals("", "")
    assert not constant_time_equals("   ", "   ")
    assert not constant_time_equals(None, "x")
    assert not constant_time_equals("a-secret-value", None)
