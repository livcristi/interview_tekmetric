import logging
import re
from typing import Optional, List

from src.api.models import RepairResponse, RepairBatchResponse
from src.core.interfaces import CacheRegister, AnomalyDetector, RepairClassifier

logger = logging.getLogger(__name__)


class RepairService:
    def __init__(
        self,
        cache: Optional[CacheRegister],
        anomaly_detector: AnomalyDetector,
        classifier: RepairClassifier,
    ):
        self.cache = cache
        self.anomaly_detector = anomaly_detector
        self.classifier = classifier

    async def classify_repair(self, text: str) -> RepairResponse:
        """Classifiers the received piece of repair text into a section and a name.
        If the text is an anomaly, both will be marked as 'unknown'"""
        logger.info(
            f"Requesting classify_repair with 1 pieces of text of length {len(text)}"
        )

        sanitized_text = self.__sanitize_text(text)
        try:
            cache_key = sanitized_text
            # Check if the item is in cache
            if self.cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    return RepairResponse(**cached)

            # Anomaly detection, followed by actual model prediction
            if self.anomaly_detector.is_anomaly(sanitized_text):
                result = RepairResponse(section="unknown", name="unknown")
            else:
                section, name = self.classifier.predict(sanitized_text)
                result = RepairResponse(section=section, name=name)

            # Save the item in cache at the end
            if self.cache:
                await self.cache.set(cache_key, result.model_dump())

            logger.info(
                f"Done classify_repair with 1 pieces of text of length {len(sanitized_text)}"
            )
            return result

        except Exception as e:
            logger.error(
                f"Unable to classify_repair on piece of text of length {len(sanitized_text)}",
                exc_info=e,
            )
            raise

    async def classify_batch_repair(self, texts: List[str]) -> RepairBatchResponse:
        """Classifiers each of the received pieces of repair text into a section and a name.
        If the text is an anomaly, both will be marked as 'unknown'"""
        logger.info(
            f"Requesting classify_batch_repair with {len(texts)} pieces of text"
        )

        try:
            results: RepairBatchResponse = [None] * len(texts)
            to_predict: List[str] = []
            predict_indices: List[int] = []

            # Check if some of the items are in cache
            if self.cache:
                for i, text in enumerate(texts):
                    sanitized_text = self.__sanitize_text(text)
                    cache_key = sanitized_text
                    cached = await self.cache.get(cache_key)
                    if cached:
                        results[i] = RepairResponse(**cached)
                    else:
                        to_predict.append(sanitized_text)
                        predict_indices.append(i)
            else:
                to_predict = texts
                predict_indices = list(range(len(texts)))

            # Anomaly detection
            if to_predict:
                anomalies = self.anomaly_detector.is_anomaly(to_predict)
                if isinstance(anomalies, bool):
                    anomalies = [anomalies] * len(to_predict)

                normal_texts: List[str] = []
                normal_indices: List[int] = []

                for idx, (text, is_anomaly) in enumerate(zip(to_predict, anomalies)):
                    sanitized_text = self.__sanitize_text(text)
                    if is_anomaly:
                        resp = RepairResponse(section="unknown", name="unknown")
                        results[predict_indices[idx]] = resp
                        if self.cache:
                            await self.cache.set(f"{sanitized_text}", resp.model_dump())
                    else:
                        normal_texts.append(sanitized_text)
                        normal_indices.append(predict_indices[idx])

                # Model prediction
                if normal_texts:
                    predictions = self.classifier.predict(normal_texts)
                    if isinstance(predictions[0], str):
                        # single tuple returned if the batch has only one element
                        predictions = [predictions]
                    for idx, (section, name) in zip(normal_indices, predictions):
                        resp = RepairResponse(section=section, name=name)
                        results[idx] = resp

                        # Save the items in cache at the end
                        if self.cache:
                            await self.cache.set(
                                f"{normal_texts.pop(0)}", resp.model_dump()
                            )

            logger.info(f"Done classify_batch_repair with {len(texts)} pieces of text")
            return results

        except Exception as e:
            logger.error(
                f"Unable to classify_batch_repair on batch with {len(texts)} pieces of text",
                exc_info=e,
            )
            raise

    @staticmethod
    def __sanitize_text(text: str) -> str:
        """Sanitize the text by removing unwanted characters and trailing whitespace"""
        unwanted_chars_pattern = r"[<>&?:\\[\]]"
        max_characters_limit = 256
        sanitized_text = re.sub(unwanted_chars_pattern, "", text)
        sanitized_text = sanitized_text.strip()[:max_characters_limit]

        return sanitized_text
