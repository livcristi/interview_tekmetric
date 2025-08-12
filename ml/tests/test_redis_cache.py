import json
import unittest
from unittest.mock import MagicMock, patch

from src.cache.redis_cache import RedisCache


class TestRedisCache(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Mock the Redis connection
        patcher = patch("src.cache.redis_cache.redis.Redis")
        self.addCleanup(patcher.stop)
        self.mock_redis_cls = patcher.start()
        self.mock_redis_client = MagicMock()
        self.mock_redis_cls.return_value = self.mock_redis_client

        self.cache = RedisCache()

    async def test_get_hit(self):
        key = "abc"
        value_dict = {"a": "1"}
        self.mock_redis_client.get.return_value = json.dumps(value_dict)
        result = await self.cache.get(key)
        self.assertEqual(result, value_dict)
        self.mock_redis_client.get.assert_called_once()

    async def test_get_miss(self):
        self.mock_redis_client.get.return_value = None
        result = await self.cache.get("abc")
        self.assertIsNone(result)

    async def test_set_success(self):
        self.mock_redis_client.setex.return_value = True
        result = await self.cache.set("abc", {"x": "y"})
        self.assertTrue(result)
        self.mock_redis_client.setex.assert_called_once()

    async def test_set_failure(self):
        self.mock_redis_client.setex.return_value = False
        result = await self.cache.set("abc", {"x": "y"})
        self.assertFalse(result)

    async def test_delete_found(self):
        self.mock_redis_client.delete.return_value = 1
        result = await self.cache.delete("abc")
        self.assertTrue(result)

    async def test_delete_not_found(self):
        self.mock_redis_client.delete.return_value = 0
        result = await self.cache.delete("abc")
        self.assertFalse(result)

    async def test_clear_with_keys(self):
        self.mock_redis_client.keys.return_value = ["k1", "k2"]
        self.mock_redis_client.delete.return_value = 2
        result = await self.cache.clear()
        self.assertTrue(result)

    async def test_clear_no_keys(self):
        self.mock_redis_client.keys.return_value = []
        result = await self.cache.clear()
        self.assertTrue(result)

    async def test_exists_true(self):
        self.mock_redis_client.exists.return_value = 1
        result = await self.cache.exists("abc")
        self.assertTrue(result)

    async def test_exists_false(self):
        self.mock_redis_client.exists.return_value = 0
        result = await self.cache.exists("abc")
        self.assertFalse(result)
