import json
from pathlib import Path

import httpx
import pytest

from ozonenv.OzonEnv import OzonEnv, OzonEnvRest
from ozonenv.core.OzonClient import OzonDataApiClient
from ozonenv.core.OzonOrm import OzonModelRest
from ozonenv.core.exceptions import SessionException
from tests.test_common import User, get_i18n_localedir_tr


pytestmark = pytest.mark.asyncio


def _cfg(models_folder: Path, **extra):
    cfg = {
        "app_code": "test-rest",
        "backend_interface": "rest",
        "models_folder": str(models_folder),
    }
    cfg.update(extra)
    return cfg


def _settings():
    return {
        "rec_name": "test-rest",
        "upload_folder": "/uploads",
        "tz": "Europe/Rome",
    }


def _sessions():
    return [
        {
            "rec_name": "session.admin",
            "uid": "admin",
            "token": "BA6BA930",
            "expire_datetime": "2030-01-01T00:00:00+00:00",
            "is_public": False,
            "user": {"uid": "admin"},
            "app_code": "test-rest",
            "active": True,
        },
        {
            "rec_name": "session.public",
            "uid": "public",
            "token": "PUBLIC",
            "expire_datetime": "2030-01-01T00:00:00+00:00",
            "is_public": True,
            "user": {"uid": "public"},
            "app_code": "test-rest",
            "active": True,
        },
    ]


def _years_component():
    path = Path(__file__).parents[1] / "data" / "test_resource_2_formio_schema_years.json"
    return json.loads(path.read_text())


def _set_i18n_env(monkeypatch):
    monkeypatch.setenv("OZON_LOCALEDIR", get_i18n_localedir_tr())
    monkeypatch.setenv("OZON_APPLANG", "it")


async def test_rest_init_env_generates_model_from_components(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)
    env = OzonEnvRest(_cfg(tmp_path / "models"))

    await env.init_env(
        components=[_years_component()],
        sessions=_sessions(),
        settings=_settings(),
    )

    assert env.orm.__class__.__name__ == "OzonOrmRest"
    assert env.get("years") is not None
    assert (tmp_path / "models" / "years.py").exists()

    await env.close_env()


async def test_rest_init_env_loads_existing_model_from_models_folder(
    monkeypatch, tmp_path
):
    _set_i18n_env(monkeypatch)
    cfg = _cfg(tmp_path / "models")
    env = OzonEnvRest(cfg)
    await env.init_env(
        components=[_years_component()],
        sessions=_sessions(),
        settings=_settings(),
    )
    await env.close_env()

    env2 = OzonEnvRest(cfg)
    await env2.init_env(
        sessions=_sessions(),
        settings=_settings(),
    )

    assert env2.get("years") is not None

    await env2.close_env()


async def test_rest_session_validation_is_local(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)
    env = OzonEnvRest(_cfg(tmp_path / "models"))
    await env.init_env(
        sessions=_sessions(),
        settings=_settings(),
    )

    env.params = {"current_session_token": "BA6BA930"}
    res = await env.session_app()
    assert res.fail is False
    assert env.user_session.uid == "admin"

    env.params = {"current_session_token": "NOT-VALID"}
    res = await env.session_app()
    assert res.fail is True
    assert res.msg == "Token NOT-VALID non abilitato"

    await env.close_env()


async def test_rest_session_is_optional_without_token(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)
    env = OzonEnvRest(_cfg(tmp_path / "models", rest_token="cfg-token"))
    await env.init_env(
        settings=_settings(),
    )

    env.params = {}
    res = await env.session_app()

    assert res.fail is False
    assert env.user_session is None
    assert env.orm.rest_client.get_headers()["Authorization"] == "Bearer cfg-token"

    await env.close_env()


async def test_rest_public_session_denies_private_models(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)
    env = OzonEnvRest(_cfg(tmp_path / "models"))
    await env.init_env(
        local_model={"user": User},
        local_model_private=["user"],
        sessions=_sessions(),
        settings=_settings(),
    )

    env.params = {"current_session_token": "PUBLIC"}
    res = await env.session_app()
    assert res.fail is False

    with pytest.raises(SessionException) as excinfo:
        await env.get("user").find({})

    assert excinfo.value.detail == "Permission Denied"
    await env.close_env()


async def test_rest_model_calls_expected_operations(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)
    env = OzonEnvRest(_cfg(tmp_path / "models"))
    await env.init_env(
        local_model={"user": User},
        sessions=_sessions(),
        settings=_settings(),
    )
    await env.orm.init_session("BA6BA930")
    model = env.get("user")

    calls = []

    async def fake_post(operation_name, payload=None):
        calls.append((operation_name, payload))
        if operation_name == "find":
            return {
                "data": [
                    {
                        "rec_name": "admin",
                        "uid": "admin",
                        "password": "secret",
                        "token": "BA6BA930",
                        "expire_datetime": "2030-01-01T00:00:00+00:00",
                    }
                ]
            }
        if operation_name == "load":
            return {
                "data": {
                    "rec_name": "admin",
                    "uid": "admin",
                    "password": "secret",
                    "token": "BA6BA930",
                    "expire_datetime": "2030-01-01T00:00:00+00:00",
                }
            }
        if operation_name == "insert":
            return {
                "data": {
                    "rec_name": "user.new",
                    "uid": "user.new",
                    "password": "secret",
                    "token": "",
                    "expire_datetime": "2030-01-01T00:00:00+00:00",
                }
            }
        return {"data": {}}

    monkeypatch.setattr(env.orm.rest_client, "post_operation", fake_post)

    found = await model.find({"uid": "admin"})
    loaded = await model.load({"uid": "admin"})
    record = await model.new(
        {
            "rec_name": "user.new",
            "uid": "user.new",
            "password": "secret",
            "expire_datetime": "2030-01-01T00:00:00+00:00",
        }
    )
    inserted = await model.insert(record)

    assert found[0].uid == "admin"
    assert loaded.uid == "admin"
    assert inserted.uid == "user.new"
    assert [name for name, _payload in calls] == ["find", "load", "insert"]

    await env.close_env()


async def test_rest_custom_cls_model_is_preserved(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)

    class CustomRestModel(OzonModelRest):
        pass

    env = OzonEnvRest(_cfg(tmp_path / "models"), cls_model=CustomRestModel)
    await env.init_env(
        local_model={"user": User},
        sessions=_sessions(),
        settings=_settings(),
    )

    assert env.orm.cls_model is CustomRestModel
    assert env.get("user").__class__ is CustomRestModel

    await env.close_env()


async def test_rest_init_db_models_uses_bootstrap_api(monkeypatch, tmp_path):
    _set_i18n_env(monkeypatch)
    calls = []

    async def fake_get_resource(self, resource_path, params=None):
        calls.append(resource_path)
        if resource_path == "collections_names":
            return {"data": ["user", "years"]}
        if resource_path == "init_settings/test-rest":
            return {"data": _settings()}
        return {}

    env = OzonEnvRest(
        _cfg(
            tmp_path / "models",
            rest_base_url="http://base_usr",
            rest_api_prefix="/base_usr/v2",
            rest_token="cfg-token",
        )
    )
    monkeypatch.setattr(
        OzonDataApiClient, "get_resource", fake_get_resource
    )

    await env.init_env(
        local_model={"user": User},
    )

    assert env.orm.db_models == ["user", "years"]
    assert env.orm.app_settings.rec_name == "test-rest"
    assert calls == ["collections_names", "init_settings/test-rest"]

    await env.close_env()


async def test_rest_backend_rejects_db_model_class(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        OzonEnv(_cfg(tmp_path / "models"))

    assert "interface_type 'db'" in str(excinfo.value)


async def test_rest_client_uses_bearer_token(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return DummyResponse()

    async def fake_get(self, url, params=None, headers=None):
        captured["get_url"] = url
        captured["get_params"] = params
        captured["get_headers"] = headers
        return DummyResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    client = OzonDataApiClient.create(
        base_url="http://example.test",
        api_prefix="/base_usr/v2",
        token="secret-token",
    )
    await client.post_operation("find", {"model": "user"})
    await client.get_resource("collections_names")

    assert captured["url"] == "http://example.test/base_usr/v2/find"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["json"]["model"] == "user"
    assert captured["get_url"] == "http://example.test/base_usr/v2/collections_names"
    assert captured["get_headers"]["Authorization"] == "Bearer secret-token"
