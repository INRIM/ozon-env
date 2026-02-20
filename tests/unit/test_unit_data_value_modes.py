from types import SimpleNamespace

import pytest

from ozonenv.core.BaseModels import Settings
from ozonenv.core.OzonModel import OzonMBase
from ozonenv.core.OzonOrm import OzonEnvBase


def _base_cfg(**extra):
    cfg = {
        "app_code": "test-app",
        "mongo_user": "test",
        "mongo_pass": "test",
        "mongo_url": "localhost:27017",
        "mongo_db": "testdb",
        "models_folder": "/tmp/models",
    }
    cfg.update(extra)
    return cfg


class DummyComputeService:
    def __init__(self):
        self.calls = 0

    async def compute_data_value(self, dati, pdata_value=None):
        self.calls += 1
        res = dati.copy()
        res["data_value"] = {"computed": True}
        return res


class DummyBgModel:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def update_data_value_bg(self, window="update_dt", hours=2):
        self.calls.append((window, hours))
        return {
            "model": self.name,
            "scanned": 2,
            "updated": 1,
            "skipped": False,
        }


def test_data_value_mode_runtime_default():
    env = OzonEnvBase(cfg=_base_cfg())

    assert env.data_value_mode == "runtime"
    assert env.is_data_value_runtime_enabled("invoice") is True
    assert env.is_data_value_runtime_enabled("session") is True


def test_data_value_mode_background_runtime_only_models():
    env = OzonEnvBase(cfg=_base_cfg(data_value_mode="background"))

    assert env.is_data_value_runtime_enabled("invoice") is False
    assert env.is_data_value_runtime_enabled("session") is True
    assert env.is_data_value_runtime_enabled("component") is True
    assert env.is_data_value_runtime_enabled("settings") is True
    assert env.is_data_value_runtime_enabled("user") is True


def test_data_value_mode_invalid_fallback_to_runtime():
    env = OzonEnvBase(cfg=_base_cfg(data_value_mode="invalid"))

    assert env.data_value_mode == "runtime"
    assert env.is_data_value_runtime_enabled("invoice") is True


@pytest.mark.asyncio
async def test_make_data_value_skips_runtime_compute_in_background_mode():
    model = OzonMBase("invoice", setting_app=Settings())
    service = DummyComputeService()
    model.service = service
    model.env = SimpleNamespace(
        is_data_value_runtime_enabled=lambda _model_name: False
    )

    res = await model.make_data_value(
        {"name": "record-1"},
        pdata_value={"old": "value"},
    )

    assert service.calls == 0
    assert res["data_value"] == {"old": "value"}


@pytest.mark.asyncio
async def test_make_data_value_computes_in_runtime_mode():
    model = OzonMBase("invoice", setting_app=Settings())
    service = DummyComputeService()
    model.service = service
    model.env = SimpleNamespace(
        is_data_value_runtime_enabled=lambda _model_name: True
    )

    res = await model.make_data_value({"name": "record-1"}, pdata_value={})

    assert service.calls == 1
    assert res["data_value"] == {"computed": True}


@pytest.mark.asyncio
async def test_env_update_data_value_bg_skips_runtime_only_models():
    env = OzonEnvBase(cfg=_base_cfg(data_value_mode="background"))
    session_model = DummyBgModel("session")
    component_model = DummyBgModel("component")
    invoice_model = DummyBgModel("invoice")
    env.models = {
        "session": session_model,
        "component": component_model,
        "invoice": invoice_model,
    }

    res = await env.update_data_value_bg(window="create_dt", hours=2)

    assert "session" not in res
    assert "component" not in res
    assert "invoice" in res
    assert invoice_model.calls == [("create_dt", 2)]


@pytest.mark.asyncio
async def test_env_update_data_value_bg_window_none_updates_all_records():
    env = OzonEnvBase(cfg=_base_cfg(data_value_mode="background"))
    invoice_model = DummyBgModel("invoice")
    env.models = {"invoice": invoice_model}

    res = await env.update_data_value_bg(window=None, hours=3)

    assert "invoice" in res
    assert invoice_model.calls == [(None, 3)]
