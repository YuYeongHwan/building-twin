from sqlalchemy import Integer, Float, String, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
import enum


class ContaminationGrade(str, enum.Enum):
    A = "A"  # 청결 (0~25%)
    B = "B"  # 경미 (25~50%)
    C = "C"  # 보통 (50~75%)
    D = "D"  # 심각 (75~100%)


class WindowResult(Base):
    __tablename__ = "window_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inspections.id"), nullable=False
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # 바운딩 박스 (픽셀 좌표)
    bbox_x: Mapped[int] = mapped_column(Integer)
    bbox_y: Mapped[int] = mapped_column(Integer)
    bbox_w: Mapped[int] = mapped_column(Integer)
    bbox_h: Mapped[int] = mapped_column(Integer)
    # 오염도 분석
    contamination_score: Mapped[float] = mapped_column(Float)  # 0.0 ~ 1.0
    grade: Mapped[ContaminationGrade] = mapped_column(SAEnum(ContaminationGrade))
    confidence: Mapped[float] = mapped_column(Float)  # 탐지 신뢰도
    crop_image_path: Mapped[str | None] = mapped_column(String(500))

    inspection: Mapped["Inspection"] = relationship("Inspection", back_populates="windows")  # noqa: F821
