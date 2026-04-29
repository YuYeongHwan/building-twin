from datetime import datetime
from pydantic import BaseModel
from app.models.inspection import InspectionStatus


class InspectionCreate(BaseModel):
    building_id: int


class InspectionResponse(BaseModel):
    id: int
    building_id: int
    video_filename: str
    total_frames: int | None
    processed_frames: int | None
    total_windows: int
    status: InspectionStatus
    inspected_at: datetime

    model_config = {"from_attributes": True}
