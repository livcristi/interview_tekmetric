from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    weights_path: Path
    softmax_threshold: float


class SimilarityConfig(BaseModel):
    data_path: Path
    model_name: str
    distance_threshold: float
    metric: Literal["cosine", "euclidean"]


class RedisCacheConfig(BaseModel):
    host: str
    port: int
    ttl_hours: int


class MemoryCacheConfig(BaseModel):
    max_size: int
    ttl_hours: int


class CacheConfig(BaseModel):
    enabled: bool
    type: Literal["redis", "memory"]
    redis: Optional[RedisCacheConfig] = None
    memory: Optional[MemoryCacheConfig] = None


class ServerConfig(BaseModel):
    host: str
    port: int
    workers: int


class AppConfig(BaseModel):
    model: ModelConfig
    similarity: SimilarityConfig
    cache: CacheConfig
    server: ServerConfig


def load_config(path: Path) -> AppConfig:
    """Load YAML config file and return validated AppConfig."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
