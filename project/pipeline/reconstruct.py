"""
3D 재구성 모듈.
frames → sparse (COLMAP SfM) → splat (3D Gaussian Splatting).
실제 실행은 외부 바이너리(COLMAP, gaussian-splatting)를 subprocess로 호출.
"""

import subprocess
import os

from backend.config import settings


def run_colmap(frames_dir: str, sparse_dir: str | None = None) -> str:
    """
    COLMAP feature_extractor + exhaustive_matcher + mapper 를 순서대로 실행.
    Returns:
        sparse_dir : SfM 결과가 저장된 경로
    """
    out = sparse_dir or settings.SPLAT_DIR.replace("splat", "sparse")
    os.makedirs(out, exist_ok=True)

    steps = [
        ["colmap", "feature_extractor",
         "--database_path", f"{out}/database.db",
         "--image_path", frames_dir],
        ["colmap", "exhaustive_matcher",
         "--database_path", f"{out}/database.db"],
        ["colmap", "mapper",
         "--database_path", f"{out}/database.db",
         "--image_path", frames_dir,
         "--output_path", out],
    ]
    for cmd in steps:
        subprocess.run(cmd, check=True)

    return out


def run_gaussian_splatting(sparse_dir: str, splat_dir: str | None = None) -> str:
    """
    gaussian-splatting train.py를 호출해 .splat 파일 생성.
    Returns:
        splat_dir : 결과 저장 경로
    """
    out = splat_dir or settings.SPLAT_DIR
    os.makedirs(out, exist_ok=True)

    subprocess.run(
        ["python", "train.py", "-s", sparse_dir, "-m", out],
        check=True,
    )
    return out
