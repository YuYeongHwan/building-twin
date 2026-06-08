from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers import stream, analysis, model

app = FastAPI(
    title="Building Twin System",
    description="드론/스마트폰 촬영 영상 기반 건물 창문 오염도 분석 시스템 (캡스톤)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="project/data"), name="data")

app.include_router(stream.router, prefix="/stream", tags=["stream"])
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(model.router, prefix="/model", tags=["model"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
