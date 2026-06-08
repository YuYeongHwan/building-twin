"""
영상 전처리 모듈.
raw_video → frames 폴더로 프레임 추출.
"""

import cv2
import os
from pathlib import Path

from backend.config import settings


def extract_frames(video_path: str, output_dir: str | None = None,
                   sample_rate: int | None = None) -> list[str]:
    """
    영상에서 프레임을 추출해 output_dir에 저장하고 저장된 경로 목록을 반환.

    Args:
        video_path  : 입력 영상 경로
        output_dir  : 프레임 저장 디렉터리 (None 이면 settings.FRAMES_DIR 사용)
        sample_rate : 매 N번째 프레임만 저장 (None 이면 settings.FRAME_SAMPLE_RATE 사용)
    """
    out_dir  = output_dir or settings.FRAMES_DIR
    rate     = sample_rate or settings.FRAME_SAMPLE_RATE
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    saved, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % rate == 0:
            name = f"frame_{frame_idx:06d}.jpg"
            path = os.path.join(out_dir, name)
            cv2.imwrite(path, frame)
            saved.append(path)
        frame_idx += 1

    cap.release()
    return saved
