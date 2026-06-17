"""
창문 검출 모듈 (흰색 A4 용지 + 검정 폼보드 모형 전용).

HSV 색공간에서 흰색 영역을 추출한 뒤 Contour 검출로 창문을 식별한다.
YOLO는 실제 건물 학습 모델이므로 사용하지 않는다.

반환:
  [{"window_id": "window_001", "bbox": [x1, y1, x2, y2]}, ...]

시각화:
  data/results/detected_windows.jpg  (초록 BBox + window_id 표시)

실행: python pipeline/detect.py [이미지_경로]
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"

# 흰색 HSV 범위 (흰색 A4 용지)
LOWER_WHITE  = np.array([0,   0, 200])
UPPER_WHITE  = np.array([180, 30, 255])

MIN_AREA     = 500   # 노이즈 제거용 최소 면적 (px²)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 흰색 영역 기반 창문 검출
# ---------------------------------------------------------------------------

def detect_windows(image: np.ndarray) -> list[dict]:
    """
    HSV 흰색 마스크 → Contour → BBox 순서로 창문을 검출한다.

    Args:
        image: BGR 이미지 (cv2.imread 결과)

    Returns:
        [{"window_id": "window_001", "bbox": [x1, y1, x2, y2]}, ...]
    """
    hsv        = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    # 모폴로지로 작은 노이즈 제거 후 구멍 메우기
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    win_num    = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue                # 노이즈 제거

        x, y, w, h = cv2.boundingRect(cnt)
        wid = f"window_{win_num:03d}"
        detections.append({"window_id": wid, "bbox": [x, y, x + w, y + h]})
        win_num += 1

    # y 좌표 기준 위→아래 정렬
    detections.sort(key=lambda d: d["bbox"][1])

    log.info(f"창문 검출 완료: {len(detections)}개  [HSV 흰색 검출]")
    return detections


# ---------------------------------------------------------------------------
# 시각화 저장
# ---------------------------------------------------------------------------

def save_visualization(image: np.ndarray, detections: list[dict],
                       out_path: Path = None) -> None:
    """BBox + window_id 를 초록색으로 그려 저장."""
    if out_path is None:
        out_path = RESULTS_DIR / "detected_windows.jpg"

    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, det["window_id"],
                    (x1 + 4, max(y1 + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    log.info(f"시각화 저장: {out_path}")


# ---------------------------------------------------------------------------
# 단독 실행
# ---------------------------------------------------------------------------

def run_detect(image_path: str) -> list[dict]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    log.info(f"검출 대상: {Path(image_path).name}")
    dets = detect_windows(img)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        log.info(f"  {d['window_id']}  bbox=[{x1},{y1},{x2},{y2}]")
    save_visualization(img, dets)
    return dets


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        frames_root = ROOT / "data" / "frames"
        images = sorted(frames_root.rglob("*.jpg"))
        if not images:
            log.error(f"data/frames/ 에 이미지가 없습니다. ({frames_root})")
            sys.exit(1)
        target = str(images[0])
        log.info(f"처리 대상 자동 선택: {Path(target).name}")

    dets = run_detect(target)
    log.info(f"검출 완료 — {len(dets)}개 창문")
