"""
창문 검출 모듈.

1차: YOLO v8 nano (신뢰도 0.3 이상)
2차: YOLO 결과 0개 또는 실패 시 → Canny Edge + Contour (플랜 B)

반환:
  [{"window_id": "window_001", "bbox": [x1, y1, x2, y2], "mask": ndarray}, ...]

실행: python pipeline/detect.py [이미지_경로]
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT           = Path(__file__).resolve().parents[1]
YOLO_MODEL     = ROOT / "yolov8n.pt"
CONF_THRESHOLD = 0.3

# Canny 필터 상수 (기존 project/pipeline/detect.py 기준)
MIN_AREA       = 4500
ASPECT_MIN     = 0.5
ASPECT_MAX     = 2.5
NMS_THRESH     = 0.05
BORDER_PAD     = 20
DUP_XY         = 30
MIN_BRIGHT     = 80
GLASS_MIN_RATIO = 0.20
BRICK_RATIO_MAX = 0.30
EDGE_DENSITY_MAX = 0.15
COLOR_VAR_MAX_V  = 65
COLOR_VAR_MAX_S  = 55

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1차: YOLO v8
# ---------------------------------------------------------------------------

def detect_with_yolo(image: np.ndarray,
                     model_path: str = str(YOLO_MODEL),
                     conf: float = CONF_THRESHOLD) -> list[list[int]]:
    """YOLO v8 nano 로 창문 검출. [x1, y1, x2, y2] 리스트 반환."""
    from ultralytics import YOLO          # ImportError 는 호출 측에서 처리
    model   = YOLO(model_path)
    results = model(image, conf=conf, verbose=False)
    boxes   = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2])
    return boxes


# ---------------------------------------------------------------------------
# 2차 폴백: Canny Edge + Contour (기존 detect.py 로직)
# ---------------------------------------------------------------------------

def _in_bounds(x, y, bw, bh, img_w, img_h) -> bool:
    return (x >= BORDER_PAD and y >= BORDER_PAD
            and x + bw <= img_w - BORDER_PAD
            and y + bh <= img_h - BORDER_PAD)


def _brightness_ok(roi_hsv: np.ndarray) -> bool:
    return roi_hsv.size > 0 and float(np.mean(roi_hsv[:, :, 2])) >= MIN_BRIGHT


def _has_grid(roi_bgr: np.ndarray) -> bool:
    rh, rw = roi_bgr.shape[:2]
    if rh < 10 or rw < 10:
        return False
    gray   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    my, mx = max(1, rh // 10), max(1, rw // 10)
    border = np.concatenate([
        gray[:my, :].flatten(), gray[-my:, :].flatten(),
        gray[my:-my, :mx].flatten(), gray[my:-my, -mx:].flatten(),
    ])
    hsv_roi  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    interior = hsv_roi[my:-my, mx:-mx]
    if interior.size == 0:
        return False
    blue_mask  = cv2.inRange(interior, (90, 20, 60), (130, 255, 255))
    blue_ratio = float(np.count_nonzero(blue_mask)) / interior[:, :, 0].size
    return float(np.mean(border)) >= 160 and blue_ratio >= 0.05


def _is_glass(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    total   = roi_hsv.shape[0] * roi_hsv.shape[1]
    neutral = (roi_hsv[:, :, 1] < 80) & (roi_hsv[:, :, 2] > 80)
    blue    = ((roi_hsv[:, :, 0] >= 90) & (roi_hsv[:, :, 0] <= 130)
               & (roi_hsv[:, :, 2] > 50))
    return float(np.count_nonzero(neutral | blue)) / total >= GLASS_MIN_RATIO


def _is_brick(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    total = roi_hsv.shape[0] * roi_hsv.shape[1]
    lo    = cv2.inRange(roi_hsv, (0,   60, 0), (20,  255, 255))
    hi    = cv2.inRange(roi_hsv, (160, 60, 0), (180, 255, 255))
    return float(np.count_nonzero(cv2.bitwise_or(lo, hi))) / total >= BRICK_RATIO_MAX


def _is_complex(roi_bgr: np.ndarray) -> bool:
    if roi_bgr.size == 0:
        return False
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    return float(np.count_nonzero(edges)) / edges.size >= EDGE_DENSITY_MAX


def _is_uniform(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    return (float(np.std(roi_hsv[:, :, 2])) <= COLOR_VAR_MAX_V
            and float(np.std(roi_hsv[:, :, 1])) <= COLOR_VAR_MAX_S)


def _area_score(bw, bh, median_area) -> float:
    if median_area == 0:
        return 0.0
    ratio = (bw * bh) / median_area
    if ratio < 0.25 or ratio > 4.0:
        return 0.0
    return 1.0 - abs(1.0 - ratio) / max(1.0, ratio)


def _nms(boxes: list[tuple]) -> list[tuple]:
    if not boxes:
        return []
    areas  = [w * h for _, _, w, h in boxes]
    median = float(np.median(areas))
    boxes  = sorted(boxes, key=lambda b: _area_score(b[2], b[3], median), reverse=True)
    kept   = []
    for x1, y1, w1, h1 in boxes:
        dominated = any(
            (lambda ix=max(0, min(x1+w1, x2+w2) - max(x1, x2)),
                   iy=max(0, min(y1+h1, y2+h2) - max(y1, y2)):
             (ix * iy) / (w1*h1 + w2*h2 - ix*iy) > NMS_THRESH
             if (w1*h1 + w2*h2 - ix*iy) > 0 else False)()
            for x2, y2, w2, h2 in kept
        )
        if not dominated:
            kept.append((x1, y1, w1, h1))
    return kept


def _dedup(boxes: list[tuple]) -> list[tuple]:
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dup = False
        for i, (x2, y2, w2, h2) in enumerate(kept):
            if abs(x1 - x2) <= DUP_XY and abs(y1 - y2) <= DUP_XY:
                if w1 * h1 > w2 * h2:
                    kept[i] = box
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


def detect_with_canny(image: np.ndarray) -> list[list[int]]:
    """Canny + Contour 기반 창문 검출. [x1, y1, x2, y2] 리스트 반환."""
    img_h, img_w = image.shape[:2]
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []

    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw * bh < MIN_AREA:
            continue
        aspect = bw / bh if bh > 0 else 0
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        if not _in_bounds(x, y, bw, bh, img_w, img_h):
            continue
        roi_hsv = hsv[y: y + bh, x: x + bw]
        roi_bgr = image[y: y + bh, x: x + bw]
        if not _brightness_ok(roi_hsv) and not _has_grid(roi_bgr):
            continue
        if not _is_glass(roi_hsv):
            continue
        if _is_brick(roi_hsv) or _is_complex(roi_bgr) or not _is_uniform(roi_hsv):
            continue
        candidates.append((x, y, bw, bh))

    kept = _dedup(_nms(candidates))
    kept.sort(key=lambda b: b[1])
    return [[x, y, x + w, y + h] for x, y, w, h in kept]


# ---------------------------------------------------------------------------
# 마스크 생성
# ---------------------------------------------------------------------------

def make_mask(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    """bbox 영역을 255 로 채운 단채널 바이너리 마스크 반환."""
    x1, y1, x2, y2 = bbox
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


# ---------------------------------------------------------------------------
# 통합 검출 진입점
# ---------------------------------------------------------------------------

def detect_windows(image: np.ndarray,
                   model_path: str = str(YOLO_MODEL)) -> list[dict]:
    """
    YOLO → (실패/0개 시) Canny+Contour 순서로 창문 검출.

    Returns:
        [{"window_id": "window_001", "bbox": [x1, y1, x2, y2], "mask": ndarray}, ...]
    """
    bboxes: list[list[int]] = []
    method = "YOLO"

    try:
        bboxes = detect_with_yolo(image, model_path)
        if not bboxes:
            raise ValueError("검출 결과 0개")
    except Exception as exc:
        log.warning(f"YOLO 검출 실패 ({exc})  →  Canny+Contour 방식으로 전환")
        bboxes = detect_with_canny(image)
        method = "Canny+Contour"

    log.info(f"창문 검출 완료: {len(bboxes)}개  [{method}]")

    return [
        {
            "window_id": f"window_{i + 1:03d}",
            "bbox":      bbox,
            "mask":      make_mask(image, bbox),
        }
        for i, bbox in enumerate(bboxes)
    ]


# ---------------------------------------------------------------------------
# 단독 실행
# ---------------------------------------------------------------------------

def run_detect(image_path: str) -> list[dict]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    log.info(f"검출 대상: {Path(image_path).name}")
    detections = detect_windows(img)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        log.info(f"  {d['window_id']}  bbox=[{x1},{y1},{x2},{y2}]")
    return detections


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

    run_detect(target)
