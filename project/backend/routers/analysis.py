from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/run/{inspection_id}")
async def run_analysis(inspection_id: str):
    """지정 검사 ID에 대해 오염도 분석 파이프라인 실행."""
    return JSONResponse({"inspection_id": inspection_id, "status": "started"})


@router.get("/result/{inspection_id}")
async def get_result(inspection_id: str):
    """분석 결과 (등급, 오염 비율, 히트맵 경로) 반환."""
    return {"inspection_id": inspection_id, "results": []}
