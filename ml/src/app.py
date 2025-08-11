from fastapi import FastAPI

from fastapi import FastAPI

from core.config import AppConfig
from src.api.routes import create_router
from src.cache.cache import get_cache_register
from src.models.classifier import EmbeddingsRepairClassifier
from src.models.local_model_repository import LocalModelRepository
from src.service.repair_service import RepairService
from src.similarity.searcher import SimilarityAnomalyDetector


def create_app(config: AppConfig) -> FastAPI:
    # Loading the cache register
    cache_register = get_cache_register(config.cache)
    # Loading the detector
    detector = SimilarityAnomalyDetector(config.similarity)
    # Loading the model
    model_repository = LocalModelRepository()
    classifier = EmbeddingsRepairClassifier(
        model_repository, config.model.weights_path, config.model.softmax_threshold
    )
    # Creating the service instance for the API
    service_instance = RepairService(cache_register, detector, classifier)

    app = FastAPI(
        title="Car Repair Classifier",
        description="Simple FastAPI app for serving a car repair section classification",
        version="0.1.0",
    )
    app.include_router(create_router(service_instance))
    return app
