import logging

from src.cache.memory_cache import MemoryCache
from src.cache.redis_cache import RedisCache
from src.core.config import CacheConfig
from src.core.interfaces import CacheRegister

logger = logging.getLogger(__name__)


def get_cache_register(cache_config: CacheConfig) -> CacheRegister:
    if not cache_config.enabled:
        return None

    if cache_config.type == "redis":
        try:
            return RedisCache(cache_config.redis.host, cache_config.redis.port)
        except Exception as e:
            logger.error(
                "Failed to initialise Redis cache, will fallback to in-memory cache",
                exc_info=e,
            )
            return MemoryCache(
                cache_config.memory.max_size, cache_config.memory.ttl_hours
            )

    return MemoryCache(cache_config.memory.max_size, cache_config.memory.ttl_hours)
