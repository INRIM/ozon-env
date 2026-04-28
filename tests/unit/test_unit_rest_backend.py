import os

import pytest

from ozonenv.OzonEnv import OzonEnv, OzonEnvRest
from tests.helpers.keycloak import get_m2m_token
from tests.helpers.rest_sidecar import (
    RealOzonEnvApiServer,
    build_test_db_cfg,
)
from tests.test_common import auth_env, get_auth_token, init_main_collections

pytestmark = pytest.mark.asyncio


async def _authenticated_env(tmp_path, name: str = "db-auth"):
    cfg = build_test_db_cfg(
        tmp_path / name,
        app_code=os.getenv("APP_CODE", "test"),
    )
    env = OzonEnv(cfg)
    await env.init_env()
    await init_main_collections(env.db)
    result = await auth_env(
        env,
        username="adminuser",
        password="adminpass",
    )
    assert result.fail is False
    return env, cfg


async def test_db_user_auth_uses_real_keycloak_and_persists_token(tmp_path):
    env, _cfg = await _authenticated_env(tmp_path)
    try:
        assert "session" not in env.models
        assert env.user_session.uid == "adminuser"
        assert env.orm.user_session.uid == "adminuser"
        assert env.session_token

        stored_user = await env.db.engine.get_collection("user").find_one(
            {"uid": "adminuser"}
        )
        assert stored_user["token"]["access_token"] == env.session_token
        assert stored_user["last_login"] is not None
    finally:
        await env.close_env()


async def test_db_user_auth_provisions_missing_user_from_valid_jwt(tmp_path):
    cfg = build_test_db_cfg(
        tmp_path / "db-auth-provision",
        app_code=os.getenv("APP_CODE", "test"),
    )
    env = OzonEnv(cfg)
    await env.init_env()
    await init_main_collections(env.db)
    try:
        user_collection = env.db.engine.get_collection("user")
        await user_collection.delete_many({"uid": "testuser"})

        result = await auth_env(
            env,
            username="testuser",
            password="testpass",
        )
        stored_user = await user_collection.find_one({"uid": "testuser"})

        assert result.fail is False
        assert env.user_session.uid == "testuser"
        assert stored_user is not None
        assert stored_user["rec_name"] == "testuser"
        assert stored_user["full_name"] == "Test User"
        assert stored_user["nome"] == "Test"
        assert stored_user["cognome"] == "User"
        assert stored_user["user_role"] == ["user"]
        assert stored_user["is_admin"] is False
        assert stored_user["active"] is True
        assert stored_user["token"]["access_token"] == env.session_token
        assert stored_user["last_login"] is not None
    finally:
        await env.close_env()


async def test_jobcontext_crud_and_real_m2m_validation(tmp_path):
    env, cfg = await _authenticated_env(tmp_path, name="jobcontext-db")
    api_env = None
    try:
        client_id = os.environ["OZON_M2M_CLIENT_ID"]
        job_context = await env.create_job_context(
            client_id=client_id,
            expire_sec=120,
            job_key="job-real-m2m",
            process_instance_key="proc-real-m2m",
        )

        loaded = await env.get("jobcontext").load(
            {"job_token": job_context.job_token}
        )
        assert loaded.job_token == job_context.job_token
        assert loaded.resolved_user_id == "adminuser"

        m2m_token = await get_m2m_token()
        api_env = OzonEnv(cfg)
        await api_env.init_env()
        resolved = await api_env.init_api_job_context(
            m2m_token=f"Bearer {m2m_token}",
            job_token=job_context.job_token,
        )

        assert resolved.job_token == job_context.job_token
        assert api_env.current_job_context.job_token == job_context.job_token
        assert api_env.user_session.uid == "adminuser"

        deleted = await env.delete_job_context(job_context.job_token)
        assert deleted is True
    finally:
        if api_env:
            await api_env.close_env()
        await env.close_env()


async def test_rest_client_calls_endpoint_backed_by_real_ozonenv_db(tmp_path):
    env, cfg = await _authenticated_env(tmp_path, name="rest-sidecar-db")
    rest = None
    try:
        user_collection = env.db.engine.get_collection("user")
        await user_collection.delete_many(
            {"uid": {"$in": ["api-user", "api-user.new"]}}
        )
        await user_collection.insert_one(
            {
                "rec_name": "api-user",
                "uid": "api-user",
                "nome": "Api",
                "cognome": "User",
                "mail": "api-user@example.test",
                "function": "worker",
                "active": True,
            }
        )
        job_context = await env.create_job_context(
            client_id=os.environ["OZON_M2M_CLIENT_ID"],
            expire_sec=300,
            job_key="job-rest-sidecar",
            process_instance_key="proc-rest-sidecar",
        )
        current_token = await get_auth_token(
            username="adminuser",
            password="adminpass",
        )

        with RealOzonEnvApiServer(cfg) as api:
            rest = OzonEnvRest(
                {
                    "app_code": "test-rest-sidecar",
                    "models_folder": str(tmp_path / "rest-client-models"),
                    "rest_base_url": api.base_url,
                    "rest_oauth_url": os.environ["OZON_OAUTH_URL"],
                    "rest_client_id": os.environ["OZON_M2M_CLIENT_ID"],
                    "rest_client_secret": os.environ["OZON_M2M_CLIENT_SECRET"],
                    "token_audience": os.environ["OZON_TOKEN_AUDIENCE"],
                }
            )
            rest.params = {
                "current_token": current_token,
                "job_token": job_context.job_token,
                "current_user": {
                    "uid": "adminuser",
                    "full_name": "Admin User",
                    "mail": "admin@example.test",
                },
            }
            await rest.init_env(
                settings={
                    "rec_name": "test-rest-sidecar",
                    "upload_folder": "/uploads",
                    "tz": "Europe/Rome",
                }
            )
            result = await rest.session_app()
            assert result.fail is False

            user_model = rest.get("user")
            users = await user_model.find({"uid": "api-user"})
            record = await user_model.new(
                {
                    "rec_name": "api-user.new",
                    "uid": "api-user.new",
                    "active": True,
                }
            )
            saved = await user_model.insert(record)

        headers = rest.orm.rest_client.get_headers()
        stored_user = await user_collection.find_one({"uid": "api-user.new"})

        assert len(users) == 1
        assert saved.uid == "api-user.new"
        assert stored_user["uid"] == "api-user.new"
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["job_token"] == job_context.job_token
    finally:
        if rest:
            await rest.close_env()
        await env.close_env()
