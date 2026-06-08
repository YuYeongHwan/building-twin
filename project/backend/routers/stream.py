from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """영상 파일 업로드 및 raw_video 저장."""
    return JSONResponse({"filename": file.filename, "status": "uploaded"})


@router.get("/status/{job_id}")
async def stream_status(job_id: str):
    """업로드/전처리 작업 진행 상태 반환."""
    return {"job_id": job_id, "status": "pending"}
