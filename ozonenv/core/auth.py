from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import jwt
from jwt import PyJWKClient
from jwt.exceptions import ExpiredSignatureError
from starlette.concurrency import run_in_threadpool


class TokenVerificationError(Exception):
    pass


class TokenExpiredError(TokenVerificationError):
    pass


class TokenRefreshError(Exception):
    pass


@dataclass
class KeycloakAuthSettings:
    jwks_url: str = ""
    issuer: str = ""
    audience: str = ""
    oauth_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    algorithms: tuple[str, ...] = ("RS256",)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        prefix: str = "",
    ) -> KeycloakAuthSettings:
        def cfg(name: str, env_name: str = "", default=""):
            if config.get(name):
                return config.get(name)
            if env_name:
                return os.getenv(env_name, "")
            return default

        algorithms = cfg(
            "keycloak_algorithms", "OZON_KEYCLOAK_ALGORITHMS", "RS256"
        )
        parsed_algorithms = ("RS256",)
        if algorithms:
            parsed_algorithms = tuple(
                item.strip()
                for item in str(algorithms).split(",")
                if item.strip()
            ) or ("RS256",)
        return cls(
            jwks_url=cfg("keycloak_jwks_url", "OZON_KEYCLOAK_JWKS_URL"),
            issuer=cfg("keycloak_issuer", "OZON_KEYCLOAK_ISSUER"),
            audience=cfg("token_audience", "OZON_TOKEN_AUDIENCE"),
            oauth_url=cfg("oauth_url", "OZON_OAUTH_URL"),
            client_id=cfg("client_id", "OZON_CLIENT_ID"),
            client_secret=cfg("client_secret", "OZON_CLIENT_SECRET"),
            algorithms=parsed_algorithms,
        )


@dataclass
class VerifiedToken:
    claims: dict[str, Any]
    access_token: str
    refresh_token: str = ""
    token_data: dict[str, Any] = field(default_factory=dict)

    @property
    def user_id(self) -> str:
        return KeycloakAuthManager.extract_user_id(self.claims)

    @property
    def client_id(self) -> str:
        return KeycloakAuthManager.extract_client_id(self.claims)


class KeycloakAuthManager:
    def __init__(self, settings: KeycloakAuthSettings) -> None:
        self.settings = settings
        self._jwk_client: Optional[PyJWKClient] = None

    @staticmethod
    def strip_bearer_token(token: str) -> str:
        token = str(token or "").strip()
        if token.lower().startswith("bearer "):
            return token[7:].strip()
        return token

    @staticmethod
    def extract_user_id(claims: dict[str, Any]) -> str:
        return (
            claims.get("preferred_username")
            or claims.get("uid")
            or claims.get("sub")
            or claims.get("email")
            or ""
        )

    @staticmethod
    def extract_client_id(claims: dict[str, Any]) -> str:
        return claims.get("azp") or claims.get("client_id") or ""

    @staticmethod
    def decode_unverified(token: str) -> dict[str, Any]:
        try:
            return jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": False,
                    "verify_aud": False,
                    "verify_iss": False,
                },
                algorithms=["RS256", "HS256"],
            )
        except Exception:
            return {}

    async def verify(
        self,
        token: str,
        expected_client_id: str = "",
    ) -> VerifiedToken:
        return await run_in_threadpool(
            self._verify_sync,
            self.strip_bearer_token(token),
            expected_client_id,
        )

    def _verify_sync(
        self,
        token: str,
        expected_client_id: str = "",
    ) -> VerifiedToken:
        if not self.settings.jwks_url:
            raise TokenVerificationError("Missing Keycloak JWKS URL")
        if self._jwk_client is None:
            self._jwk_client = PyJWKClient(self.settings.jwks_url)
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(self.settings.algorithms),
                audience=self.settings.audience or None,
                issuer=self.settings.issuer or None,
                options={
                    "verify_aud": bool(self.settings.audience),
                    "verify_iss": bool(self.settings.issuer),
                },
            )
        except ExpiredSignatureError as exc:
            raise TokenExpiredError(str(exc)) from exc
        except Exception as exc:
            raise TokenVerificationError(str(exc)) from exc

        if expected_client_id:
            token_client_id = self.extract_client_id(claims)
            if token_client_id != expected_client_id:
                raise TokenVerificationError("Token client_id is not allowed")
        return VerifiedToken(claims=claims, access_token=token)

    async def refresh(self, refresh_token: str) -> dict[str, Any]:
        if not self.settings.oauth_url:
            raise TokenRefreshError("Missing OAuth token URL")
        if not self.settings.client_id or not self.settings.client_secret:
            raise TokenRefreshError(
                "Missing OAuth client credentials for refresh"
            )
        data = {
            "grant_type": "refresh_token",
            "client_id": self.settings.client_id,
            "client_secret": self.settings.client_secret,
            "refresh_token": refresh_token,
        }
        if self.settings.audience:
            data["audience"] = self.settings.audience
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(self.settings.oauth_url, data=data)
        try:
            response.raise_for_status()
        except Exception as exc:
            raise TokenRefreshError(str(exc)) from exc
        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise TokenRefreshError(
                "OAuth refresh response does not contain access_token"
            )
        return payload
