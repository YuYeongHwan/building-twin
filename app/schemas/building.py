from datetime import datetime
from pydantic import BaseModel


class BuildingCreate(BaseModel):
    name: str
    address: str | None = None
    floor_count: int | None = None
    description: str | None = None


class BuildingResponse(BaseModel):
    id: int
    name: str
    address: str | None
    floor_count: int | None
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
