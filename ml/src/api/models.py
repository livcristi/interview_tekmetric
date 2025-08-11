from typing import List

from pydantic import BaseModel, RootModel


class RepairRequest(BaseModel):
    text: str


class RepairBatchRequest(BaseModel):
    texts: List[str]


class RepairResponse(BaseModel):
    section: str
    name: str


class RepairBatchResponse(RootModel):
    root: List[RepairResponse]
