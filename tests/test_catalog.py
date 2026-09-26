from app.application.catalog import ApiKeyStore, CatalogService, ListCatalog
from app.core.security import hash_api_key
from app.domain.models import ListMeta
from tests.conftest import FakeBackend


def test_api_key_store_upsert_and_remove():
    store = ApiKeyStore()
    # El backend solo entrega hashes; las búsquedas llegan con la key en claro.
    store.replace_all([(1, 10, hash_api_key("key-a")), (2, 20, hash_api_key("key-b"))])
    assert store.resolve_raw("key-a").user_id == 10
    assert store.resolve_hash(hash_api_key("key-a")).id == 1
    store.upsert(3, 10, hash_api_key("key-c"))
    assert store.resolve_raw("key-c").id == 3
    store.remove(1)
    assert store.resolve_raw("key-a") is None
    store.remove_user(10)
    assert store.resolve_raw("key-c") is None
    assert store.resolve_raw("key-b") is not None
    assert store.resolve_raw(hash_api_key("key-b")) is None  # el hash no vale como key


def test_list_catalog_rename_updates_name_index():
    catalog = ListCatalog()
    catalog.upsert(ListMeta(id=1, user_id=10, name="Cámaras", is_public=True))
    assert catalog.resolve(10, "Cámaras").id == 1
    catalog.upsert(ListMeta(id=1, user_id=10, name="Fotografía", is_public=True))
    assert catalog.resolve(10, "Cámaras") is None
    assert catalog.resolve(10, "Fotografía").id == 1


def test_list_catalog_scoped_by_user():
    catalog = ListCatalog()
    catalog.upsert(ListMeta(id=1, user_id=10, name="Lista", is_public=True))
    catalog.upsert(ListMeta(id=2, user_id=20, name="Lista", is_public=False))
    assert catalog.resolve(10, "Lista").id == 1
    assert catalog.resolve(20, "Lista").id == 2
    assert catalog.remove_user(10) == [1]
    assert catalog.resolve(10, "Lista") is None


async def test_catalog_service_refresh_and_miss_throttle():
    backend = FakeBackend()
    backend.api_keys = [(1, 10, hash_api_key("raw-key"))]
    backend.lists = [ListMeta(id=1, user_id=10, name="L", is_public=True)]
    service = CatalogService(backend, ApiKeyStore(), ListCatalog(), min_refresh_interval=999)

    await service.refresh()
    assert service.ready
    assert service.api_keys.resolve_raw("raw-key") is not None
    assert backend.bootstrap_calls == 1

    # Dentro de la ventana de throttle el refresh por miss se salta.
    assert await service.refresh_on_miss() is False
    assert backend.bootstrap_calls == 1


async def test_catalog_service_miss_refresh_runs_when_stale():
    backend = FakeBackend()
    service = CatalogService(backend, ApiKeyStore(), ListCatalog(), min_refresh_interval=0)
    assert await service.refresh_on_miss() is True
    assert backend.bootstrap_calls == 1


async def test_catalog_service_survives_backend_failure():
    backend = FakeBackend()
    backend.fail_bootstrap = True
    service = CatalogService(backend, ApiKeyStore(), ListCatalog(), min_refresh_interval=0)
    assert await service.refresh_on_miss() is False
    assert not service.ready
