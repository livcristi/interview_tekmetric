import unittest
from unittest.mock import AsyncMock, MagicMock

from src.api.models import RepairResponse
from src.service.repair_service import RepairService


class TestRepairService(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Mock dependencies
        self.mock_cache = AsyncMock()
        self.mock_anomaly_detector = MagicMock()
        self.mock_classifier = MagicMock()

        self.service = RepairService(
            cache=self.mock_cache,
            anomaly_detector=self.mock_anomaly_detector,
            classifier=self.mock_classifier
        )

    async def test_classify_repair_from_cache(self):
        cached_value = {"section": "cached_section", "name": "cached_name"}
        expected_result = RepairResponse(**cached_value)
        self.mock_cache.get.return_value = cached_value

        result = await self.service.classify_repair("some text")

        self.assertEqual(result, expected_result)
        self.mock_cache.get.assert_awaited_once()
        self.mock_anomaly_detector.is_anomaly.assert_not_called()
        self.mock_classifier.predict.assert_not_called()

    async def test_classify_repair_anomaly(self):
        self.mock_cache.get.return_value = None
        self.mock_anomaly_detector.is_anomaly.return_value = True

        result = await self.service.classify_repair("some anomaly text")

        self.assertEqual(result.section, "unknown")
        self.assertEqual(result.name, "unknown")
        self.mock_classifier.predict.assert_not_called()
        self.mock_cache.set.assert_awaited_once()

    async def test_classify_repair_normal_prediction(self):
        self.mock_cache.get.return_value = None
        self.mock_anomaly_detector.is_anomaly.return_value = False
        self.mock_classifier.predict.return_value = ("pred_section", "pred_name")
        expected_result = RepairResponse(section="pred_section", name="pred_name")

        result = await self.service.classify_repair("some normal text")

        self.assertEqual(result, expected_result)
        self.mock_classifier.predict.assert_called_once()
        self.mock_cache.set.assert_awaited_once()

    async def test_classify_batch_repair_all_cached(self):
        self.mock_cache.get.side_effect = [
            {"section": "s1", "name": "n1"},
            {"section": "s2", "name": "n2"}
        ]

        result = await self.service.classify_batch_repair(["t1", "t2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].section, "s1")
        self.assertEqual(result[1].section, "s2")
        self.mock_anomaly_detector.is_anomaly.assert_not_called()

    async def test_classify_batch_repair_mixed_anomalies_and_predictions(self):
        # First item cached, second needs prediction
        self.mock_cache.get.side_effect = [
            {"section": "cached", "name": "cached"},
            None
        ]
        self.mock_anomaly_detector.is_anomaly.return_value = [True]
        self.mock_classifier.predict.return_value = [("sec_pred", "name_pred")]

        result = await self.service.classify_batch_repair(["t1", "t2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].section, "cached")
        self.assertEqual(result[1].section, "unknown")  # anomaly
        self.mock_cache.set.assert_awaited()

    async def test_classify_batch_repair_normal_predictions(self):
        self.mock_cache.get.side_effect = [None, None]
        self.mock_anomaly_detector.is_anomaly.return_value = [False, False]
        self.mock_classifier.predict.return_value = [
            ("sec1", "name1"),
            ("sec2", "name2")
        ]

        result = await self.service.classify_batch_repair(["t1", "t2"])

        self.assertEqual(result[0].section, "sec1")
        self.assertEqual(result[1].section, "sec2")
        self.mock_cache.set.assert_awaited()
