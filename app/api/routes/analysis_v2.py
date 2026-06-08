import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.window import WindowResult, ContaminationGrade

router = APIRouter(prefix="/analysis", tags=["analysis"])

_ROOT         = Path(__file__).resolve().parents[3]
_RESULTS_FILE = _ROOT / "data" / "results.json"


@router.post("/result")
def save_analysis_result(
    inspection_id: int = Query(default=1, description="연결할 inspection ID"),
    db: Session = Depends(get_db),
):
    """
    data/results.json 을 읽어 window_results 테이블에 저장.
    같은 inspection_id + frame_number 가 이미 있으면 pollution_index / grade 만 갱신.
    """
    if not _RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="data/results.json 파일이 없습니다.")

    with open(_RESULTS_FILE, encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    saved = updated = 0
    for item in items:
        try:
            seq = int(item["window_id"].split("_")[1])
        except (IndexError, ValueError):
            seq = 0

        grade_str = item.get("grade", "D")
        try:
            grade = ContaminationGrade(grade_str)
        except ValueError:
            grade = ContaminationGrade.D

        pollution_index = float(item.get("pollution_index", 0.0))

        existing = (
            db.query(WindowResult)
            .filter(
                WindowResult.inspection_id == inspection_id,
                WindowResult.frame_number == seq,
            )
            .first()
        )
        if existing:
            existing.pollution_index = pollution_index
            existing.grade = grade
            existing.contamination_score = pollution_index
            updated += 1
        else:
            db.add(WindowResult(
                inspection_id=inspection_id,
                frame_number=seq,
                bbox_x=0, bbox_y=0, bbox_w=0, bbox_h=0,
                contamination_score=pollution_index,
                pollution_index=pollution_index,
                grade=grade,
                confidence=1.0,
            ))
            saved += 1

    db.commit()
    return {"saved": saved, "updated": updated, "total": saved + updated}


@router.get("/windows")
def get_all_windows(
    inspection_id: int = Query(default=None, description="특정 inspection만 조회"),
    db: Session = Depends(get_db),
):
    """window_results 전체(또는 특정 inspection) 반환."""
    q = db.query(WindowResult)
    if inspection_id:
        q = q.filter(WindowResult.inspection_id == inspection_id)
    windows = q.order_by(WindowResult.inspection_id, WindowResult.frame_number).all()

    return [
        {
            "id":                w.id,
            "window_id":         f"window_{w.frame_number:03d}",
            "inspection_id":     w.inspection_id,
            "frame_number":      w.frame_number,
            "bbox":              [w.bbox_x, w.bbox_y, w.bbox_w, w.bbox_h],
            "contamination_score": w.contamination_score,
            "pollution_index":   w.pollution_index,
            "grade":             w.grade.value if w.grade else "D",
            "confidence":        w.confidence,
        }
        for w in windows
    ]
