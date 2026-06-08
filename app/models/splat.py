from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class SplatModel(Base):
    __tablename__ = "splat_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    building_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("buildings.id"), nullable=True
    )
    splat_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
