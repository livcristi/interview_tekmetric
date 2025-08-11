import hashlib
import json
import logging
from typing import Dict, Optional

import redis

from src.core.interfaces import CacheRegister

logger = logging.getLogger(__name__)


class RedisCache(CacheRegister):
    """Redis cache implementation for classification results."""

    CACHE_PREFIX = "repairs_classification"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl_hours: int = 24,
        socket_connect_timeout: int = 5,
        socket_timeout: int = 5,
        retry_on_timeout: bool = True,
        max_connections: int = 50,
    ):
        self.default_ttl_hours = default_ttl_hours
        self.default_ttl_seconds = default_ttl_hours * 3600

        # Redis connection pool
        self.connection_pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            retry_on_timeout=retry_on_timeout,
            max_connections=max_connections,
            decode_responses=True,
        )

        self.redis_client = redis.Redis(connection_pool=self.connection_pool)

    def __make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        # Create a hash of the key to handle long sentences and special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{self.CACHE_PREFIX}:{key_hash}"

    async def get(self, key: str) -> Optional[Dict[str, str]]:
        """Get classification result from Redis cache."""
        try:
            redis_key = self.__make_key(key)
            value = self.redis_client.get(redis_key)

            if value is not None:
                result = self.__deserialize_value(value)
                logger.debug(f"Redis cache hit for key: {key}")
                return result
            else:
                logger.debug(f"Redis cache miss for key: {key}")
                return None

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis get error for key {key}", exc_info=e)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}", exc_info=e)
            return None

    async def set(
        self, key: str, value: Dict[str, str], ttl_hours: Optional[int] = None
    ) -> bool:
        """Store classification result in Redis cache."""
        try:
            redis_key = self.__make_key(key)
            serialized_value = self.__serialize_value(value)
            ttl_seconds = (ttl_hours or self.default_ttl_hours) * 3600

            result = self.redis_client.setex(redis_key, ttl_seconds, serialized_value)

            if result:
                logger.debug(f"Stored in Redis cache: {key} -> {value}")
                return True
            else:
                return False

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis set error for key {key}", exc_info=e)
            return False
        except Exception as e:
            logger.error(f"Error for key {key}", exc_info=e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete classification result from Redis cache."""
        try:
            redis_key = self.__make_key(key)
            result = self.redis_client.delete(redis_key)

            if result > 0:
                logger.debug(f"Deleted from Redis cache: {key}")
                return True
            else:
                return False

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis delete error for key {key}", exc_info=e)
            return False

    async def clear(self) -> bool:
        """Clear all cached classification results."""
        try:
            pattern = f"{self.CACHE_PREFIX}:*"
            keys = self.redis_client.keys(pattern)

            if keys:
                result = self.redis_client.delete(*keys)
                logger.info(f"Cleared {result} entries from Redis cache")
                return True
            else:
                logger.info("No entries to clear from Redis cache")
                return True

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis clear error", exc_info=e)
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis_key = self.__make_key(key)
            result = self.redis_client.exists(redis_key)
            return bool(result)

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis exists error for key {key}", exc_info=e)
            return False

    @staticmethod
    def __serialize_value(value: Dict[str, str]) -> str:
        """Serialize classification result to JSON string."""
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def __deserialize_value(value: str) -> Dict[str, str]:
        """Deserialize JSON string to classification result."""
        return json.loads(value)
