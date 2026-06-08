from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/list")
async def list_models():
    """사용 가능한 3DGS / 탐지 모델 목록 반환."""
    return {"models": []}


@router.get("/{model_id}/info")
async def model_info(model_id: str):
    """모델 메타데이터 반환."""
    return {"model_id": model_id}
