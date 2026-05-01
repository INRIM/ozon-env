import os
from pathlib import Path

import pytest

from ozonenv.core.BaseModels import DataReturn
from ozonenv.core.BaseModels import OzonEnvCoreSettings

TESTS_DIR = Path(__file__).parent.parent


class TestOzonEnvCoreSettingsFromEnv:
    def test_from_env_reads_yaml(self, monkeypatch):
        # uses tests/.ozonenv/config.yaml — acts as usage example too
        # env vars come from tests/.env-test loaded by pytest-dotenv
        monkeypatch.chdir(TESTS_DIR)

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.app_code == "test"  # APP_CODE=test in .env-test
        assert cfg.mongo_db == "servicetest"  # MONGO_DB=servicetest
        assert cfg.api_prefix == "/v2"  # hardcoded in yaml
        assert cfg.require_auth is False  # hardcoded in yaml

    def test_from_env_yaml_missing_fields_use_defaults(
        self, tmp_path, monkeypatch
    ):
        ozonenv_dir = tmp_path / ".ozonenv"
        ozonenv_dir.mkdir()
        (ozonenv_dir / "config.yaml").write_text("app_code: myapp\n")
        monkeypatch.chdir(tmp_path)

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.app_code == "myapp"
        assert cfg.api_prefix == "/v2"
        assert cfg.require_auth is True
        assert cfg.auth_mode == "session"
        assert cfg.keycloak_algorithms == "RS256"

    def test_from_env_no_yaml_uses_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APP_CODE", "envapp")
        monkeypatch.setenv("MONGO_DB", "envdb")
        monkeypatch.setenv("OZON_API_PREFIX", "/v99")

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.app_code == "envapp"
        assert cfg.mongo_db == "envdb"
        assert cfg.api_prefix == "/v99"

    def test_from_env_yaml_interpolates_env_vars(self, tmp_path, monkeypatch):
        ozonenv_dir = tmp_path / ".ozonenv"
        ozonenv_dir.mkdir()
        (ozonenv_dir / "config.yaml").write_text(
            "app_code: ${APP_CODE}\n" "mongo_db: ${MONGO_DB}\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APP_CODE", "from_env")
        monkeypatch.setenv("MONGO_DB", "mydb")

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.app_code == "from_env"
        assert cfg.mongo_db == "mydb"

    def test_from_env_yaml_unset_var_uses_field_default(
        self, tmp_path, monkeypatch
    ):
        ozonenv_dir = tmp_path / ".ozonenv"
        ozonenv_dir.mkdir()
        (ozonenv_dir / "config.yaml").write_text(
            "app_code: myapp\n"
            "api_prefix: ${UNSET_VAR_XYZ}\n"  # unset → key dropped
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("UNSET_VAR_XYZ", raising=False)

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.api_prefix == "/v2"  # Pydantic field default

    def test_from_env_yaml_takes_precedence_over_env(
        self, tmp_path, monkeypatch
    ):
        ozonenv_dir = tmp_path / ".ozonenv"
        ozonenv_dir.mkdir()
        (ozonenv_dir / "config.yaml").write_text("app_code: yaml_wins\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APP_CODE", "env_loses")

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.app_code == "yaml_wins"

    def test_from_env_empty_yaml_uses_all_defaults(
        self, tmp_path, monkeypatch
    ):
        ozonenv_dir = tmp_path / ".ozonenv"
        ozonenv_dir.mkdir()
        (ozonenv_dir / "config.yaml").write_text("")
        monkeypatch.chdir(tmp_path)

        cfg = OzonEnvCoreSettings.from_env()

        assert cfg.app_code is None
        assert cfg.api_prefix == "/v2"
        assert cfg.backend_interface == "db"


class TestDataReturn:
    def test_default_initialization(self):
        """Test default initialization of DataReturn"""
        result = DataReturn()

        assert result.data is None  # data should default to None
        assert result.fail is False  # fail should default to False
        assert result.msg == ""  # msg should default to an empty string

    def test_custom_initialization(self):
        """Test custom initialization of DataReturn with various types"""
        result = DataReturn(data=42, fail=True, msg="Custom message")

        assert result.data == 42
        assert result.fail is True
        assert result.msg == "Custom message"

    def test_list_data(self):
        """Test using a list as the data"""
        data = [1, 2, 3]
        result = DataReturn(data=data)

        assert result.data == data
        assert result.data[0] == 1

    def test_dict_data(self):
        """Test using a dictionary as the data"""
        data = {"key": "value"}
        result = DataReturn(data=data)

        assert result.data == data
        assert result.data["key"] == "value"

    def test_none_data(self):
        """Test explicitly passing None to data"""
        result = DataReturn(data=None, fail=False, msg="No data")

        assert result.data is None
        assert result.fail is False
        assert result.msg == "No data"
