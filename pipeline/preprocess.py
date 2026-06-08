"""
영상 전처리 모듈.

입력  : data/raw_video/ 안의 영상 파일 경로
출력  : data/frames/<영상명>/ 에 정제된 이미지 저장

처리 흐름:
  1. 초당 2프레임 추출 (cv2.VideoCapture)
  2. 블러 프레임 제거 (Laplacian variance < 100)
  3. CLAHE 조명 정규화 (LAB L-채널)
  4. 결과 로그 출력

실행: python pipeline/preprocess.py
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
WATCH_DIR  = ROOT / "data" / "raw_video"
FRAMES_DIR = ROOT / "data" / "frames"
TARGET_FPS = 2
BLUR_THRESHOLD = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 핵심 함수
# ---------------------------------------------------------------------------

def is_sharp(img: np.ndarray, threshold: int = BLUR_THRESHOLD) -> bool:
    """Laplacian variance 가 threshold 초과인 경우만 선명한 프레임으로 판단."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """BGR 이미지의 L-채널에 CLAHE 를 적용해 조명 정규화 후 BGR 반환."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def run_preprocess(video_path: str) -> int:
    """
    영상을 전처리해 data/frames/<영상명>/ 에 저장한다.

    Returns:
        저장된 프레임 수
    """
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    out_dir = FRAMES_DIR / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 매 N번째 프레임마다 1장 샘플링해 TARGET_FPS 에 근사
    sample_interval = max(1, round(src_fps / TARGET_FPS))

    total = 0       # 전체 디코딩 프레임
    blurry = 0      # 블러로 제외된 프레임
    saved = 0       # 최종 저장 프레임

    log.info(f"전처리 시작: {src.name}  (원본 {src_fps:.1f} fps → 샘플 간격 {sample_interval})")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        if not is_sharp(frame):
            blurry += 1
            frame_idx += 1
            continue

        normalized = apply_clahe(frame)
        out_path = out_dir / f"frame_{saved:06d}.jpg"
        cv2.imwrite(str(out_path), normalized)
        saved += 1
        frame_idx += 1

    cap.release()

    log.info(
        f"전처리 완료: {src.name}\n"
        f"  전체 프레임    : {total}\n"
        f"  블러 제외      : {blurry}\n"
        f"  최종 저장      : {saved}  →  {out_dir}"
    )
    return saved


# ---------------------------------------------------------------------------
# 단독 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        videos = sorted(WATCH_DIR.glob("*.mp4")) + sorted(WATCH_DIR.glob("*.mov"))
        if not videos:
            log.error(f"data/raw_video/ 에 .mp4 / .mov 파일이 없습니다. ({WATCH_DIR})")
            sys.exit(1)
        target = str(videos[0])
        log.info(f"처리 대상 자동 선택: {Path(target).name}")

    count = run_preprocess(target)
    log.info(f"완료 — 저장된 프레임 수: {count}")
