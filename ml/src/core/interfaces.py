from abc import abstractmethod, ABC
from typing import Any, Dict, List, Tuple, Optional


class ModelRepository(ABC):
    """Abstract base class for retrieving model data from repositories"""

    @abstractmethod
    def load_model(self, model_id: str) -> Any:
        """Loads a model given its id"""
        pass

    @abstractmethod
    def get_model_metadata(self, model_id: str) -> Dict:
        """Gets the metadata associated with a model given its id"""
        pass


class AnomalyDetector(ABC):
    """Abstract class for anomaly detection"""

    @abstractmethod
    def is_anomaly(self, query: str | List[str]) -> bool | List[bool]:
        """Determines if query/queries are anomalies based on similarity threshold."""
        pass


class RepairClassifier(ABC):
    """Abstract base class for repair classification"""

    @abstractmethod
    def predict(
        self, texts: str | List[str]
    ) -> Tuple[str, str] | List[Tuple[str, str]]:
        """
        Predict section and name for input text(s).

        Args:
            texts: Single text string or list of text strings

        Returns:
            Single (section, name) tuple or list of (section, name) tuples
        """
        pass


class CacheRegister(ABC):
    """Abstract class for caching classification results."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, str]]:
        """Get classification result from cache."""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: Dict[str, str], ttl_hours: Optional[int] = None
    ) -> bool:
        """Store classification result in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete classification result from cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached classification results."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
