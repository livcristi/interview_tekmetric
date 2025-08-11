from typing import Dict, Any, override

import torch

from src.core.interfaces import ModelRepository


class LocalModelRepository(ModelRepository):
    @override
    def load_model(self, model_id: str) -> Any:
        return torch.load(model_id, map_location="cpu", weights_only=False)

    @override
    def get_model_metadata(self, model_id: str) -> Dict:
        return {"name": "embeddings_based_model_20250811"}
