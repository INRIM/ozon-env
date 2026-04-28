import asyncio

from ozonenv.OzonEnv import OzonEnv
from ozonenv.core.db.mongodb_utils import (
    connect_to_mongo,
    close_mongo_connection,
    DbSettings,
    AsyncIOMotorCollection,
)
from test_common import *
from tests.helpers.keycloak import get_keycloak_token

pytestmark = pytest.mark.asyncio


@pytestmark
async def test_keycloak_user_token() -> str:
    return await get_keycloak_token(
        username="testuser",
        password="testpass",
    )


@pytestmark
async def test_ozonenv_cfg():
    env = OzonEnv()
    assert env.config_system['app_code'] == 'test'
    assert env.config_system["mongo_url"] == "localhost:10002"
    assert (
        env.config_system["keycloak_issuer"]
        == "http://localhost:10765/realms/test"
    )


@pytestmark
async def test_ozonenv_from_os_env():
    env = OzonEnv()
    assert env.config_system['app_code'] == 'test'


@pytestmark
async def test_init_env_db_exist():
    config_system = {
        "app_code": os.getenv("APP_CODE"),
        "mongo_user": os.getenv("MONGO_USER"),
        "mongo_pass": os.getenv("MONGO_PASS"),
        "mongo_url": os.getenv("MONGO_URL"),
        "mongo_db": os.getenv("MONGO_DB"),
        "mongo_replica": os.getenv("MONGO_REPLICA"),
    }
    db_settings = DbSettings(**config_system)
    db = await connect_to_mongo(db_settings)
    env = OzonEnv()
    await env.init_orm(db=db)
    user = env.db.engine.get_collection('user')
    assert isinstance(user, AsyncIOMotorCollection)
    await env.close_env()
    assert db.client.is_primary
    await close_mongo_connection(db)
    with pytest.raises(Exception) as excinfo:
        assert db.client.is_primary
    assert str(excinfo.value) == 'Cannot use MongoClient after close'


@pytestmark
async def test_init_env():
    env = OzonEnv()
    env.use_cache = False
    await env.init_orm()
    await init_main_collections(env.db)
    user = env.db.engine.get_collection('user')
    users = await user.find({}).to_list(length=None)
    assert len(users) == 1
    stored_obj = await user.find_one({'uid': 'adminuser'})
    assert stored_obj['uid'] == "adminuser"
    settings = env.db.engine.get_collection('settings')
    query = {"rec_name": env.config_system.get("app_code")}
    set_stored_obj = await settings.find_one(query)
    assert set_stored_obj['rec_name'] == "test"
    await env.close_db()


@pytestmark
async def test_make_app_session():
    seed_env = OzonEnv()
    await seed_env.init_orm()
    await init_main_collections(seed_env.db)
    await seed_env.close_db()

    env = OzonEnv()
    assert env.cls_model.__name__ == "OzonModel"

    token = await get_keycloak_token(
        username="adminuser",
        password="adminpass",
    )

    res = await env.make_app_session(
        {"current_token": token}, redis_url="redis://localhost:10003"
    )
    assert res.fail is False
    assert len(env.models) == 5
    assert env.orm.user_session.get('uid') == "adminuser"
    assert env.orm.user_session.get(
        'create_datetime'
    ) == BasicModel.iso_to_utc("2022-08-05T05:10:02+02:00")
    assert env.orm.user_session.active is True
    assert env.orm.user_session.is_to_delete() is False
    assert env.orm.user_session.is_error() is False


@pytestmark
async def test_init_user():
    env = OzonEnv()
    assert env.cls_model.__name__ == "OzonModel"
    await env.login("testuser", "testpass")
    _user = await env.get('user').load({"uid": "testuser"})
    assert _user.uid == "testuser"
    assert env.orm.user_session.get('uid') == _user.uid
    assert env.orm.user_session.active is True
    assert env.orm.user_session.is_to_delete() is False
    assert env.orm.user_session.is_error() is False
    await env.close_db()


@pytestmark
async def test_make_app_session_error():
    env = OzonEnv()
    res = await env.make_app_session(
        {"current_token": "BA6B----"},
        use_cache=True,
        redis_url="redis://localhost:100013",
    )
    assert res.fail is True
    assert res.msg


@pytestmark
async def test_two_independent_connections_and_delayed_close():
    cfg1 = {
        "app_code": os.getenv("APP_CODE"),
        "mongo_user": os.getenv("MONGO_USER"),
        "mongo_pass": os.getenv("MONGO_PASS"),
        "mongo_url": os.getenv("MONGO_URL"),
        "mongo_db": os.getenv("MONGO_DB"),
        "mongo_replica": os.getenv("MONGO_REPLICA"),
        "models_folder": os.getenv("MODELS_FOLDER"),
    }
    cfg2 = cfg1.copy()
    cfg2["mongo_db"] = f"{cfg1['mongo_db']}_second"

    env1 = OzonEnv(cfg1)
    env2 = OzonEnv(cfg2)
    await env1.init_orm()
    await env2.init_orm()

    pong1 = await env1.db.engine.command("ping")
    pong2 = await env2.db.engine.command("ping")
    assert pong1.get("ok") == 1.0
    assert pong2.get("ok") == 1.0

    await env1.close_db()

    # db2 must remain usable after db1 close
    pong2_after_close_db1 = await env2.db.engine.command("ping")
    assert pong2_after_close_db1.get("ok") == 1.0

    await asyncio.sleep(3)
    await env2.close_db()

    with pytest.raises(Exception) as excinfo:
        assert env2.db.client.is_primary
    assert "Cannot use MongoClient after close" in str(excinfo.value)
