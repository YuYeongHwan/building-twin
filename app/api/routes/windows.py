from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.window import WindowResult, ContaminationGrade
from app.schemas.window import WindowResultResponse, GradeSummary

router = APIRouter(prefix="/api/windows", tags=["windows"])

_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _ROOT / "data" / "results"


def _resolve_crop_image(raw_path: str) -> Optional[Path]:
    """crop_image_path 저장 형식이 여러 버전 섞여 있어 순서대로 시도한다.

    - 신규(inspection별 하위 폴더): "results/9/f00000_t001.jpg" → <ROOT>/results/9/...
    - 구버전(죽은 절대경로): "/results/images/window_024_crop.jpg" → 파일명만 추출해
      현재 이미지 폴더(data/results/)에서 찾는다.
    """
    if not raw_path:
        return None

    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(_ROOT / raw_path.lstrip("/"))
    else:
        candidates.append(_ROOT / raw_path)
    candidates.append(_RESULTS_DIR / p.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@router.get("/", response_model=list[WindowResultResponse])
def list_windows(
    inspection_id: Optional[int] = None,
    grade: Optional[ContaminationGrade] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    q = db.query(WindowResult)
    if inspection_id:
        q = q.filter(WindowResult.inspection_id == inspection_id)
    if grade:
        q = q.filter(WindowResult.grade == grade)
    return q.limit(limit).all()


@router.get("/summary/{inspection_id}", response_model=list[GradeSummary])
def grade_summary(inspection_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(WindowResult.grade, func.count(WindowResult.id).label("count"))
        .filter(WindowResult.inspection_id == inspection_id)
        .group_by(WindowResult.grade)
        .all()
    )
    total = sum(r.count for r in rows)
    return [
        GradeSummary(
            grade=r.grade,
            count=r.count,
            percentage=round(r.count / total * 100, 1) if total else 0.0,
        )
        for r in rows
    ]


@router.get("/{window_id}/image")
def get_window_image(window_id: int, db: Session = Depends(get_db)):
    win = db.query(WindowResult).filter(WindowResult.id == window_id).first()
    if not win or not win.crop_image_path:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    image_path = _resolve_crop_image(win.crop_image_path)
    if image_path is None:
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다.")
    return FileResponse(str(image_path), media_type="image/jpeg")
