from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import buildings, inspections, windows, pages, dashboard
from app.api.routes import splat, analysis_v2, mesh, lidar

_ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="Building Twin System",
    description="드론/스마트폰 촬영 영상 기반 건물 창문 오염도 분석 시스템",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 정적 파일 마운트 ──────────────────────────────────────────
# 크롭 이미지: /static/results/<filename> → data/results/<filename>
# NOTE: /static/results 를 /static 보다 먼저 마운트해야 prefix 충돌 없음
(_ROOT / "data" / "results").mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/results",
    StaticFiles(directory=str(_ROOT / "data" / "results")),
    name="result-images",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ODM(OpenDroneMap) 결과물 폴더: mesh/texture/좌표 파일을 직접 URL로 접근 가능하게 함
(_ROOT / "data" / "odm").mkdir(parents=True, exist_ok=True)
app.mount("/odm", StaticFiles(directory=str(_ROOT / "data" / "odm")), name="odm")

# ── 라우터 ───────────────────────────────────────────────────
app.include_router(pages.router)
app.include_router(buildings.router)
app.include_router(inspections.router)
app.include_router(windows.router)
app.include_router(dashboard.router)
app.include_router(splat.router)
app.include_router(analysis_v2.router)
app.include_router(mesh.router)
app.include_router(lidar.router)


# ── .splat 서빙 (직접 엔드포인트) ────────────────────────────
@app.get("/models/splat")
async def get_splat():
    """data/splat/output.splat 파일을 스트림으로 반환."""
    splat_path = _ROOT / "data" / "splat" / "output.splat"
    if not splat_path.exists():
        raise HTTPException(status_code=404, detail=".splat 파일이 없습니다. 파이프라인을 먼저 실행하세요.")
    return FileResponse(
        str(splat_path),
        media_type="application/octet-stream",
        filename="output.splat",
    )


# ── 시작 이벤트 ───────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
