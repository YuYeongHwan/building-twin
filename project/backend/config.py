import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    DATA_DIR: str = "project/data"
    RAW_VIDEO_DIR: str = "project/data/raw_video"
    FRAMES_DIR: str = "project/data/frames"
    SPLAT_DIR: str = "project/data/splat"

    YOLO_MODEL_PATH: str = "ml/weights/yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    FRAME_SAMPLE_RATE: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
