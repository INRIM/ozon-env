import os

import httpx


async def get_keycloak_token(
    username: str = "testuser",
    password: str = "testpass",
    client_id: str | None = None,
    client_secret: str | None = None,
    oauth_url: str | None = None,
) -> str:
    token_url = oauth_url or os.environ["OZON_OAUTH_URL"]

    data = {
        "grant_type": "password",
        "client_id": client_id or os.environ["OZON_CLIENT_ID"],
        "client_secret": client_secret or os.environ["OZON_CLIENT_SECRET"],
        "username": username,
        "password": password,
        "audience": client_id or os.environ["OZON_TOKEN_AUDIENCE"],
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(token_url, data=data)
        response.raise_for_status()
        payload = response.json()

    return payload["access_token"]


async def get_m2m_token(
    client_id: str | None = None,
    client_secret: str | None = None,
    oauth_url: str | None = None,
) -> str:
    token_url = oauth_url or os.environ["OZON_OAUTH_URL"]

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id or os.environ["OZON_M2M_CLIENT_ID"],
        "client_secret": client_secret or os.environ["OZON_M2M_CLIENT_SECRET"],
    }
    audience = os.environ.get("OZON_TOKEN_AUDIENCE", "")
    if audience:
        data["audience"] = audience
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(token_url, data=data)
        response.raise_for_status()
        payload = response.json()

    return payload["access_token"]
