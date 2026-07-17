from types import SimpleNamespace

import pytest

from ozonenv.core.BaseModels import User
from ozonenv.core.OzonOrm import OzonOrm
from ozonenv.core.auth import TokenExpiredError

pytestmark = pytest.mark.asyncio


class _UserCollection:
    def __init__(self):
        self.inserted = None
        self.updated = None

    async def count_documents(self, query):
        return 0

    async def insert_one(self, data):
        self.inserted = data

    async def update_one(self, query, update):
        self.updated = (query, update)


class _Engine:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        assert name == "user"
        return self.collection


class _AuthManager:
    def __init__(self):
        self.refreshed_with = None
        self.verify_calls = 0

    async def verify(self, access_token):
        self.verify_calls += 1
        if self.verify_calls == 1:
            raise TokenExpiredError("expired")
        return SimpleNamespace()

    async def refresh(self, refresh_token):
        self.refreshed_with = refresh_token
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        }


class _AuthOrm(OzonOrm):
    def __init__(self, collection=None, auth_manager=None):
        self.db = (
            SimpleNamespace(engine=_Engine(collection)) if collection else None
        )
        self.env = SimpleNamespace(
            params={},
            get_user_auth_manager=lambda: auth_manager,
        )
        self.tz = "Europe/Rome"
        self.loaded_user_count = 0

    async def load_user_by_uid(self, uid):
        self.loaded_user_count += 1
        return User(uid=uid, check_fields=False)


async def test_persist_user_token_only_updates_login_metadata():
    collection = _UserCollection()
    orm = _AuthOrm(collection=collection)

    await orm.persist_user_token(
        "user-1",
        {"access_token": "access", "refresh_token": "refresh"},
    )

    query, update = collection.updated
    assert query == {"uid": "user-1"}
    assert set(update) == {"$set", "$unset"}
    assert set(update["$set"]) == {"last_login"}
    assert update["$unset"] == {"token": ""}


async def test_save_user_does_not_insert_token_data():
    collection = _UserCollection()
    orm = _AuthOrm(collection=collection)
    user = User(
        uid="user-1",
        rec_name="user-1",
        token={"access_token": "access", "refresh_token": "refresh"},
        check_fields=False,
    )

    await orm.save_user(user)

    assert "token" not in collection.inserted


async def test_expired_access_token_does_not_load_refresh_from_user():
    auth_manager = _AuthManager()
    orm = _AuthOrm(auth_manager=auth_manager)

    with pytest.raises(TokenExpiredError):
        await orm.authenticate_user_token("expired-access")

    assert orm.loaded_user_count == 0
    assert auth_manager.refreshed_with is None


async def test_request_refresh_token_is_used_without_database_credentials():
    auth_manager = _AuthManager()
    orm = _AuthOrm(auth_manager=auth_manager)

    verified = await orm.authenticate_user_token(
        {
            "access_token": "expired-access",
            "refresh_token": "request-refresh",
        }
    )

    assert orm.loaded_user_count == 0
    assert auth_manager.refreshed_with == "request-refresh"
    assert verified.access_token == "fresh-access"
    assert verified.refresh_token == "fresh-refresh"
    assert verified.token_data == {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
    }
