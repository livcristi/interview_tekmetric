from http.client import HTTPException
from typing import Any

from fastapi import APIRouter

from src.api.models import (
    RepairResponse,
    RepairRequest,
    RepairBatchResponse,
    RepairBatchRequest,
)
from src.service.repair_service import RepairService


def create_router(service: RepairService) -> APIRouter:
    router = APIRouter()

    @router.post("/repairs", response_model=RepairResponse)
    async def classify_repair(request: RepairRequest) -> Any:
        try:
            return await service.classify_repair(request.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/repairs_batch", response_model=RepairBatchResponse)
    async def classify_batch_repair(request: RepairBatchRequest,) -> Any:
        try:
            return await service.classify_batch_repair(request.texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
