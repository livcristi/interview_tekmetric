from typing import List, override, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.core.interfaces import RepairClassifier


class TrainingRepairClassifier(nn.Module, RepairClassifier):
    """Repair classifier used during the training process"""

    def __init__(
        self,
        embedding_model_name="all-MiniLM-L6-v2",
        num_classes=None,
        hidden_dim=128,
        dropout=0.3,
        threshold=0.7,
        label_encoder=None,
    ):
        super(TrainingRepairClassifier, self).__init__()

        # Load pre-trained sentence transformer
        self.sentence_transformer = SentenceTransformer(embedding_model_name)

        # Freeze embeddings
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self.embedding_dim = (
            self.sentence_transformer.get_sentence_embedding_dimension()
        )

        # Single classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.threshold = threshold
        self.label_encoder = label_encoder

    def forward(self, texts):
        # Generate embeddings
        if isinstance(texts, list):
            embeddings = self.sentence_transformer.encode(
                texts, convert_to_tensor=True, device=next(self.parameters()).device
            )
            # Clone to make it a normal tensor for autograd, otherwise it causes errors
            embeddings = embeddings.clone().detach().requires_grad_(True)
        else:
            embeddings = texts

        # Classification
        logits = self.classifier(embeddings)
        return logits

    @override
    def predict(
        self, texts: str | List[str]
    ) -> Tuple[str, str] | List[Tuple[str, str]]:
        """Predict section and name for input text(s)"""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            probs = F.softmax(logits, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)

            results = []
            for i in range(len(texts)):
                if max_probs[i].item() < self.threshold:
                    results.append(("unknown", "unknown"))
                else:
                    combined_label = self.label_encoder.inverse_transform(
                        [predictions[i].cpu().numpy()]
                    )[0]
                    section, name = combined_label.split("|")
                    results.append((section, name))

        return results[0] if single_input else results
