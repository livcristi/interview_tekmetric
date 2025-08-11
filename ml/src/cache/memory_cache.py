import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from src.core.interfaces import CacheRegister

logger = logging.getLogger(__name__)


class MemoryCache(CacheRegister):
    """In-memory cache implementation with TTL support."""

    def __init__(self, max_size: int = 10000, default_ttl_hours: int = 24):
        self.max_size = max_size
        self.default_ttl_hours = default_ttl_hours
        self._cache: Dict[str, Tuple[Dict[str, str], datetime]] = {}
        self._access_times: Dict[str, datetime] = {}

    async def get(self, key: str) -> Optional[Dict[str, str]]:
        """Get classification result from memory cache."""
        self.__cleanup_expired()

        if key in self._cache:
            value, timestamp = self._cache[key]
            if not self.__is_expired(timestamp, self.default_ttl_hours):
                self._access_times[key] = datetime.now()
                logger.debug(f"Memory cache hit for key: {key}")
                return value
            else:
                # Remove expired entry
                self._cache.pop(key, None)
                self._access_times.pop(key, None)

        logger.debug(f"Memory cache miss for key: {key}")
        return None

    async def set(
        self, key: str, value: Dict[str, str], ttl_hours: Optional[int] = None
    ) -> bool:
        """Store classification result in memory cache."""
        try:
            self.__evict_if_needed()
            self.__cleanup_expired()

            timestamp = datetime.now()
            self._cache[key] = (value, timestamp)
            self._access_times[key] = timestamp

            logger.debug(f"Stored in memory cache: {key} -> {value}")
            return True

        except Exception as e:
            logger.error(f"Failed to store in memory cache", exc_info=e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete classification result from memory cache."""
        if key in self._cache:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            logger.debug(f"Deleted from memory cache: {key}")
            return True
        return False

    async def clear(self) -> bool:
        """Clear all cached classification results."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Cleared memory cache")
        return True

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            return not self.__is_expired(timestamp, self.default_ttl_hours)
        return False

    @staticmethod
    def __is_expired(timestamp: datetime, ttl_hours: int) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > timestamp + timedelta(hours=ttl_hours)

    def __evict_if_needed(self) -> None:
        """Evict the oldest entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove oldest accessed entries
            sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[: len(sorted_keys) // 2]]

            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)

            logger.info(f"Evicted {len(keys_to_remove)} entries from memory cache")

    def __cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = []

        for key, (value, timestamp) in self._cache.items():
            if self.__is_expired(timestamp, self.default_ttl_hours):
                expired_keys.append(key)

        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)

        if expired_keys:
            logger.info(
                f"Cleaned up {len(expired_keys)} expired entries from memory cache"
            )
