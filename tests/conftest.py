import pytest_asyncio

from tests.helpers.keycloak import get_keycloak_token


@pytest_asyncio.fixture
async def keycloak_user_token() -> str:
    return await get_keycloak_token(
        username="testuser",
        password="testpass",
    )


@pytest_asyncio.fixture
async def keycloak_admin_token() -> str:
    return await get_keycloak_token(
        username="adminuser",
        password="adminpass",
    )
