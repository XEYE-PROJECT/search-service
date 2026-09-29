"""Fallo cerrado en producción: sin un INTERNAL_TOKEN fuerte el servicio no arranca."""

import pytest
from pydantic import ValidationError

from app.core.config import MIN_INTERNAL_TOKEN_LENGTH, Settings, is_local_host
from app.core.security import constant_time_equals, hash_api_key

#: Lo que un env de producción válido tiene además del token (ver xeye-infra/env/search.env.example).
PROD_OK = dict(
    cors_origins="https://xeye.es,https://www.xeye.es",
    backend_url="http://xeye-backend:8000",
    allowed_hosts="search.xeye.es,search-service,localhost",
)


def settings(**overrides) -> Settings:
    if overrides.get("environment") == "production":
        overrides = {**PROD_OK, **overrides}
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


@pytest.mark.parametrize(
    ("field", "value", "variable"),
    [
        ("cors_origins", "", "CORS_ORIGINS"),
        ("cors_origins", "https://xeye.es,http://xeye.es", "CORS_ORIGINS"),
        ("cors_origins", "https://xeye.es,https://localhost:3000", "CORS_ORIGINS"),
        ("backend_url", "http://localhost:8000", "BACKEND_URL"),
        ("backend_url", "http://127.0.0.1:8000", "BACKEND_URL"),
        ("backend_url", "xeye-backend:8000", "BACKEND_URL"),
        ("rate_limit_per_minute", 0, "RATE_LIMIT_PER_MINUTE"),
        ("rate_limit_per_ip_per_minute", 0, "RATE_LIMIT_PER_IP_PER_MINUTE"),
        ("max_request_bytes", 0, "MAX_REQUEST_BYTES"),
        ("allowed_hosts", "*", "ALLOWED_HOSTS"),
        ("allowed_hosts", "", "ALLOWED_HOSTS"),
        ("allowed_hosts", "search.xeye.es,*", "ALLOWED_HOSTS"),
    ],
)
def test_production_rejects_dev_urls_and_origins(field, value, variable):
    with pytest.raises(ValidationError, match=variable):
        settings(environment="production", internal_token="a" * 48, **{field: value})


def test_production_reports_every_problem_at_once():
    with pytest.raises(ValidationError) as info:
        settings(
            environment="production",
            internal_token="short",
            cors_origins="http://localhost:3000",
            backend_url="http://localhost:8000",
        )
    message = str(info.value)
    assert "INTERNAL_TOKEN" in message and "CORS_ORIGINS" in message and "BACKEND_URL" in message


def test_development_accepts_localhost_everywhere():
    cfg = settings(environment="development")  # defaults: CORS y backend en localhost
    assert cfg.backend_url.startswith("http://localhost")
    assert cfg.allowed_host_list == ["*"]


def test_allowed_hosts_are_normalized():
    cfg = settings(environment="development", allowed_hosts=" Search.XEYE.es, localhost ,")
    assert cfg.allowed_host_list == ["search.xeye.es", "localhost"]


def test_is_local_host():
    assert is_local_host("http://localhost:8000")
    assert is_local_host("https://user@127.0.0.1/x?y")
    assert is_local_host("http://[::1]:8000")
    assert is_local_host("https://app.localhost")
    assert not is_local_host("http://xeye-backend:8000")
    assert not is_local_host("https://xeye.es/localhost")
    assert not is_local_host("https://localhost.xeye.es")


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
