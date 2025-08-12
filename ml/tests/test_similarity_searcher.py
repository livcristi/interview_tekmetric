import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np

from src.core.config import SimilarityConfig
from src.similarity.searcher import SimilarityAnomalyDetector


class TestSimilarityAnomalyDetector(unittest.TestCase):

    def setUp(self):
        # Patch SentenceTransformer, cosine_similarity, euclidean_distances, and open()
        patcher_model = patch("src.similarity.searcher.SentenceTransformer")
        self.mock_model_cls = patcher_model.start()
        self.addCleanup(patcher_model.stop)

        patcher_cosine = patch("src.similarity.searcher.cosine_similarity")
        self.mock_cosine = patcher_cosine.start()
        self.addCleanup(patcher_cosine.stop)

        patcher_euclid = patch("src.similarity.searcher.euclidean_distances")
        self.mock_euclid = patcher_euclid.start()
        self.addCleanup(patcher_euclid.stop)

        patcher_open = patch("builtins.open", mock_open(read_data="text1\ntext2\ntext3\n"))
        self.mock_open = patcher_open.start()
        self.addCleanup(patcher_open.stop)

        # Mock embedder.encode return
        self.mock_embedder_instance = MagicMock()
        self.mock_embedder_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.mock_model_cls.return_value = self.mock_embedder_instance

        # Prepare config
        self.config = SimilarityConfig(
            model_name="dummy-model",
            distance_threshold=0.5,
            data_path="dummy_path.txt",
            metric="cosine",
        )

    def test_initialization_loads_training_data_and_embeddings(self):
        detector = SimilarityAnomalyDetector(self.config)

        self.assertTrue(self.mock_open.called)
        args, kwargs = self.mock_open.call_args
        self.assertEqual(str(args[0]), "dummy_path.txt")
        self.assertEqual(args[1], "r")

        self.assertEqual(detector.known_texts, ["text1", "text2", "text3"])

        self.mock_embedder_instance.encode.assert_any_call(
            ["text1", "text2", "text3"], convert_to_numpy=True
        )

        self.assertTrue(hasattr(detector, "known_embeddings"))

    def test_is_anomaly_single_query(self):
        detector = SimilarityAnomalyDetector(self.config)

        # Setup mock cosine_similarity to return a matrix with max similarity > threshold
        self.mock_cosine.return_value = np.array([[0.6, 0.4, 0.3]])

        result = detector.is_anomaly("some query")
        self.assertFalse(result)  # max similarity 0.6 > threshold 0.5 => not anomaly

        # Test when max similarity < threshold
        self.mock_cosine.return_value = np.array([[0.4, 0.3, 0.2]])
        result = detector.is_anomaly("some query")
        self.assertTrue(result)

    def test_is_anomaly_batch_queries(self):
        detector = SimilarityAnomalyDetector(self.config)

        # Return a 2x3 similarity matrix (2 queries, 3 known embeddings)
        self.mock_cosine.return_value = np.array([
            [0.6, 0.4, 0.3],
            [0.4, 0.3, 0.2]
        ])

        results = detector.is_anomaly(["query1", "query2"])
        self.assertEqual(results, [False, True])  # first query not anomaly, second is
