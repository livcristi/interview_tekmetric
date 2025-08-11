from typing import List, Tuple

from src.core.interfaces import RepairClassifier, ModelRepository
from src.models.trained_classifier import TrainingRepairClassifier


class EmbeddingsRepairClassifier(RepairClassifier):
    """Simple classifier based on sentence embeddings"""

    def __init__(
        self, model_repository: ModelRepository, model_id: str, threshold: float = 0.5
    ):
        checkpoint = model_repository.load_model(model_id)

        config = checkpoint["model_config"]
        self.label_encoder = checkpoint["label_encoder"]

        # Recreate model
        self.model = TrainingRepairClassifier(
            embedding_model_name=config["embedding_model_name"],
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            threshold=threshold,
            label_encoder=self.label_encoder,
        )

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(
            f"Model loaded with {config['num_classes']} classes, threshold: {threshold}"
        )

    def set_threshold(self, threshold: float):
        """Update the confidence threshold"""
        self.model.threshold = threshold

    def predict(
        self, texts: str | List[str]
    ) -> Tuple[str, str] | List[Tuple[str, str]]:
        """Predict section and name for input text(s)"""
        return self.model.predict(texts)
