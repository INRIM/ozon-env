from aioredis.client import Redis
from typing import Any
import logging
from typing import Tuple
from .coder import PickleCoder

from aioredis import Redis


class RedisBackend:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.coder = PickleCoder

    async def get_with_ttl(self, app_code: str, key: str) -> Tuple[int, str]:
        async with self.redis.pipeline(transaction=True) as pipe:
            return await (pipe.ttl(
                f"{app_code}:{key}").get(f"{app_code}:{key}").execute())

    async def get(self, app_code: str, key: str) -> Any:
        if await self.redis.exists(f"{app_code}:{key}") == 0:
            return False
        return self.coder.decode(await self.redis.get(f"{app_code}:{key}"))

    async def set(self, app_code: str, key: str, value: Any, expire: int = 60):
        return await self.redis.set(
            f"{app_code}:{key}", PickleCoder.encode(value), ex=expire)

    async def clear(self, app_code: str = None, key: str = None) -> int:
        if app_code:
            lua = f"for i, name in ipairs(redis.call('KEYS'," \
                  f" '{app_code}:*')) do redis.call('DEL', name); end"
            return await self.redis.eval(lua, numkeys=0)
        elif key:
            return await self.redis.delete(key)


class OzonCache:
    client: Redis = None
    cache: RedisBackend = None


ioredis = OzonCache()


async def get_redis() -> Redis:
    return ioredis.client


async def get_cache() -> RedisBackend:
    return ioredis.cache
