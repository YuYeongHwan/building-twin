import os
import logging
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.building import Building
from app.models.inspection import Inspection, InspectionStatus
from app.schemas.inspection import InspectionResponse

router = APIRouter(prefix="/api/inspections", tags=["inspections"])
log = logging.getLogger(__name__)

# 프로젝트 루트: window_inspection/app/api/routes/ → 3단계 위
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_pipeline(inspection_id: int):
    from app.core.database import SessionLocal
    from ml.pipeline import InspectionPipeline

    db = SessionLocal()
    inspection = None
    try:
        inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
        if not inspection:
            log.error("Inspection id=%d 를 찾을 수 없습니다.", inspection_id)
            return

        # 절대 경로로 변환 — FastAPI working directory에 무관하게 동작
        video_path = str(_PROJECT_ROOT / "uploads" / inspection.video_filename)
        log.info("파이프라인 백그라운드 시작: inspection_id=%d, video=%s",
                 inspection_id, video_path)

        pipeline = InspectionPipeline()
        pipeline.process(inspection, video_path, db)

    except Exception as e:
        log.error("파이프라인 백그라운드 오류 (inspection_id=%d): %s",
                  inspection_id, e, exc_info=True)
        try:
            if inspection is not None:
                inspection.status = InspectionStatus.FAILED
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.get("/", response_model=list[InspectionResponse])
def list_inspections(building_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Inspection)
    if building_id:
        q = q.filter(Inspection.building_id == building_id)
    return q.order_by(Inspection.inspected_at.desc()).all()


@router.post("/", response_model=InspectionResponse, status_code=201)
async def create_inspection(
    background_tasks: BackgroundTasks,
    building_id: int = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    building = db.query(Building).filter(Building.id == building_id).first()
    if not building:
        raise HTTPException(status_code=404, detail="건물을 찾을 수 없습니다.")

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir / video.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    inspection = Inspection(
        building_id=building_id,
        video_filename=video.filename,
        status=InspectionStatus.PENDING,
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    background_tasks.add_task(_run_pipeline, inspection.id)
    return inspection


@router.get("/{inspection_id}", response_model=InspectionResponse)
def get_inspection(inspection_id: int, db: Session = Depends(get_db)):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="검사를 찾을 수 없습니다.")
    return inspection


@router.delete("/{inspection_id}", status_code=204)
def delete_inspection(inspection_id: int, db: Session = Depends(get_db)):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="검사를 찾을 수 없습니다.")
    db.delete(inspection)
    db.commit()
