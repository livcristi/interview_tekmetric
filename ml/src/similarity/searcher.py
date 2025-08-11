from typing import List, Literal, override

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from src.core.config import SimilarityConfig
from src.core.interfaces import AnomalyDetector


class SimilarityAnomalyDetector(AnomalyDetector):
    """Anomaly detector based on semantic similarity to known training examples."""

    def __init__(self, config: SimilarityConfig):
        self.model_name = config.model_name
        self.threshold = config.distance_threshold
        self.data_path = config.data_path
        self.metric: Literal["cosine", "euclidean"] = config.metric

        # Load model
        self.embedder = SentenceTransformer(self.model_name)

        # Load data
        self.__load_training_data()

    @override
    def is_anomaly(self, query: str | List[str]) -> bool | List[bool]:
        """Determines if query/queries are anomalies based on similarity threshold."""
        if isinstance(query, str):
            queries = [query]
            single_input = True
        else:
            queries = query
            single_input = False

        query_embs = self.embedder.encode(queries, convert_to_numpy=True)
        sims = self.__compute_similarity(query_embs)

        # For each query, check if max similarity >= threshold
        results = [float(np.max(row)) < self.threshold for row in sims]

        return results[0] if single_input else results

    def __load_training_data(self) -> None:
        """Reads dataset and precomputes embeddings for known samples."""
        with open(self.data_path, "r") as training_data_file:
            self.known_texts = [
                line_data.strip() for line_data in training_data_file.readlines()
            ]
        self.known_embeddings = self.embedder.encode(
            self.known_texts, convert_to_numpy=True
        )

    def __compute_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Computes similarity or distance to known embeddings."""
        if self.metric == "cosine":
            return cosine_similarity(query_embedding, self.known_embeddings)
        elif self.metric == "euclidean":
            distances = euclidean_distances(query_embedding, self.known_embeddings)
            return 1 / (1 + distances)  # scale to (0, 1]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
