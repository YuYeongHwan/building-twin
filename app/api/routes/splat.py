from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["models"])

_ROOT = Path(__file__).resolve().parents[3]
_SPLAT_DIR = _ROOT / "data" / "splat"


@router.get("/models/splat")
def serve_splat():
    """data/splat/ 에서 .splat 파일을 찾아 반환."""
    candidate = _SPLAT_DIR / "output.splat"
    if not candidate.exists():
        found = sorted(_SPLAT_DIR.rglob("*.splat")) if _SPLAT_DIR.exists() else []
        if not found:
            raise HTTPException(status_code=404, detail=".splat 파일이 없습니다. 파이프라인을 먼저 실행하세요.")
        candidate = found[0]
    return FileResponse(
        str(candidate),
        media_type="application/octet-stream",
        filename="model.splat",
    )
