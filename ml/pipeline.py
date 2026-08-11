"""
영상 처리 파이프라인: 비디오 → 프레임 샘플링 → 창문 탐지 → 오염도 분석 → DB 저장

실행 예:
    python ml/pipeline.py --video ./test_video.MOV --building_id 1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 로거 설정 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 탐지 파라미터 ─────────────────────────────────────────────────
MIN_AREA    = 4500    # 1.5배 상향 — 작은 구조물 제거
ASPECT_MIN  = 0.5
ASPECT_MAX  = 2.5
NMS_THRESH  = 0.05
BORDER_PAD  = 20
DUP_XY      = 30
MIN_BRIGHT  = 80

# ── 추가 필터 파라미터 ────────────────────────────────────────────
GLASS_PIXEL_MIN_RATIO = 0.20
BRICK_RATIO_MAX       = 0.30
EDGE_DENSITY_MAX      = 0.15
COLOR_VAR_MAX_V       = 65
COLOR_VAR_MAX_S       = 55

# ── 트래킹 파라미터 ───────────────────────────────────────────────
TRACK_IOU_THRESH = 0.30   # 같은 창문으로 판단할 IoU
FRAME_SAMPLE_RATE = 30    # 매 N프레임마다 처리

# ── 흰 프레임 사전 필터 파라미터 ─────────────────────────────────
FRAME_WHITE_V_MIN     = 185   # 밝기(V) 하한
FRAME_WHITE_S_MAX     = 95    # 채도(S) 상한
FRAME_SKY_CUTOFF      = 0.15  # 상단(하늘) 제외 비율
FRAME_BOTTOM_CUTOFF   = 0.90  # 하단(바닥) 제외 비율
FRAME_MIN_WHITE_RATIO = 0.01  # 프레임 전체 대비 흰 픽셀 최소 비율 (미만이면 창문 없는 프레임)
CROP_MIN_WHITE_RATIO  = 0.05  # 창문 crop 대비 흰 픽셀 최소 비율 (미만이면 외벽 오탐으로 판단)


# ════════════════════════════════════════════════════════════════
# 탐지 함수 (detector.py 로직 인라인)
# ════════════════════════════════════════════════════════════════

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def _is_in_bounds(x, y, bw, bh, img_w, img_h):
    return (x >= BORDER_PAD and y >= BORDER_PAD
            and x + bw <= img_w - BORDER_PAD
            and y + bh <= img_h - BORDER_PAD)


def _brightness_ok(region_hsv):
    return region_hsv.size > 0 and float(np.mean(region_hsv[:, :, 2])) >= MIN_BRIGHT


def _has_grid_pattern(region_bgr):
    if region_bgr.size == 0:
        return False
    rh, rw = region_bgr.shape[:2]
    if rh < 10 or rw < 10:
        return False
    gray   = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    my, mx = max(1, rh // 10), max(1, rw // 10)
    border = np.concatenate([
        gray[:my, :].flatten(), gray[-my:, :].flatten(),
        gray[my:-my, :mx].flatten(), gray[my:-my, -mx:].flatten(),
    ])
    hsv_roi  = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    interior = hsv_roi[my:-my, mx:-mx]
    if interior.size == 0:
        return False
    blue_mask  = cv2.inRange(interior, (90, 20, 60), (130, 255, 255))
    blue_ratio = float(np.count_nonzero(blue_mask)) / interior[:, :, 0].size
    return float(np.mean(border)) >= 160 and blue_ratio >= 0.05


def _is_glass_color(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    total   = roi_hsv.shape[0] * roi_hsv.shape[1]
    neutral = (roi_hsv[:, :, 1] < 80) & (roi_hsv[:, :, 2] > 80)
    blue    = ((roi_hsv[:, :, 0] >= 90) & (roi_hsv[:, :, 0] <= 130)
               & (roi_hsv[:, :, 2] > 50))
    return float(np.count_nonzero(neutral | blue)) / total >= GLASS_PIXEL_MIN_RATIO


def _is_brick(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    total = roi_hsv.shape[0] * roi_hsv.shape[1]
    lo = cv2.inRange(roi_hsv, (0,   60, 0), (20,  255, 255))
    hi = cv2.inRange(roi_hsv, (160, 60, 0), (180, 255, 255))
    return float(np.count_nonzero(cv2.bitwise_or(lo, hi))) / total >= BRICK_RATIO_MAX


def _is_complex_structure(roi_bgr: np.ndarray) -> bool:
    if roi_bgr.size == 0:
        return False
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    return float(np.count_nonzero(edges)) / edges.size >= EDGE_DENSITY_MAX


def _is_color_uniform(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    return (float(np.std(roi_hsv[:, :, 2])) <= COLOR_VAR_MAX_V
            and float(np.std(roi_hsv[:, :, 1])) <= COLOR_VAR_MAX_S)


def has_windows(img: np.ndarray) -> bool:
    """프레임에 흰 창틀 프레임 픽셀이 충분히 있는지 확인해 창문 없는 프레임을 걸러낸다."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    lower_white = np.array([0, 0, FRAME_WHITE_V_MIN])
    upper_white = np.array([180, FRAME_WHITE_S_MAX, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 하늘 영역 제거
    mask_white[:int(h * FRAME_SKY_CUTOFF), :] = 0
    # 바닥 영역 제거
    mask_white[int(h * FRAME_BOTTOM_CUTOFF):, :] = 0

    white_ratio = cv2.countNonZero(mask_white) / (h * w)

    # 흰색 픽셀이 기준치 미만이면 창문 없는 프레임으로 판단
    return white_ratio > FRAME_MIN_WHITE_RATIO


def is_valid_window(crop_img: np.ndarray) -> bool:
    """창문 영역 내 흰색 프레임 비율이 낮으면 외벽 오탐으로 판단해 제외한다."""
    if crop_img.size == 0:
        return False
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, FRAME_WHITE_V_MIN])
    upper_white = np.array([180, FRAME_WHITE_S_MAX, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    ratio = cv2.countNonZero(mask) / (crop_img.shape[0] * crop_img.shape[1])
    return ratio > CROP_MIN_WHITE_RATIO


def _area_score(bw, bh, median_area):
    if median_area == 0:
        return 0.0
    ratio = (bw * bh) / median_area
    if ratio < 0.25 or ratio > 4.0:
        return 0.0
    return 1.0 - abs(1.0 - ratio) / max(1.0, ratio)


def _nms(boxes):
    if not boxes:
        return []
    areas  = [bw * bh for _, _, bw, bh in boxes]
    median = float(np.median(areas))
    boxes  = sorted(boxes, key=lambda b: _area_score(b[2], b[3], median), reverse=True)
    kept   = []
    for box in boxes:
        x1, y1, w1, h1 = box
        if not any(_iou(box, k) > NMS_THRESH for k in kept):
            kept.append(box)
    return kept


def _dedup(boxes):
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dup = False
        for i, k in enumerate(kept):
            x2, y2, w2, h2 = k
            if abs(x1-x2) <= DUP_XY and abs(y1-y2) <= DUP_XY:
                if w1*h1 > w2*h2:
                    kept[i] = box
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


def detect_windows(frame: np.ndarray) -> list[tuple]:
    """창문 박스 [(x,y,w,h), ...] 반환."""
    img_h, img_w = frame.shape[:2]
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        if not _is_in_bounds(x, y, bw, bh, img_w, img_h):
            continue
        roi_hsv = hsv[y:y+bh, x:x+bw]
        roi_bgr = frame[y:y+bh, x:x+bw]
        if not _brightness_ok(roi_hsv) and not _has_grid_pattern(roi_bgr):
            continue
        if not _is_glass_color(roi_hsv):
            continue
        if _is_brick(roi_hsv):
            continue
        if _is_complex_structure(roi_bgr):
            continue
        if not _is_color_uniform(roi_hsv):
            continue
        candidates.append((x, y, bw, bh))

    boxes = _nms(candidates)
    boxes = _dedup(boxes)
    boxes.sort(key=lambda b: b[1])
    return boxes


# ════════════════════════════════════════════════════════════════
# 엣지 기반 창문 검출 (색상이 아닌 형태로 검출)
#
# 흐린 날씨에서는 창문 유리가 회색빛으로 변해 색상(HSV) 기반 검출이
# 불안정하고, 흰 프레임 밝기만으로는 화강암 외벽의 밝은 부분과 구분이
# 어렵다. 대신 Canny 엣지 + 폴리곤 컨투어의 사각형 형태(면적/종횡비/
# solidity)와 내부 격자 엣지 밀도로 창문을 식별한다.
# ════════════════════════════════════════════════════════════════

def detect_windows_edge(img: np.ndarray) -> list[dict]:
    """Canny 엣지 + Contour 형태 필터로 창문을 검출한다.

    Returns:
        [{"bbox": [x1, y1, x2, y2], "area": float, "crop": np.ndarray}, ...]
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 하늘 영역 제거
    sky_cut = int(h * 0.12)
    bottom_cut = int(h * 0.92)
    roi = gray[sky_cut:bottom_cut, :]

    # 2. CLAHE로 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)

    # 2-1. 가우시안 블러로 화강암 벽의 미세 텍스처 노이즈 완화
    #      (원래는 3. Canny 뒤에 dilate로 엣지를 이어 붙였으나, 화강암
    #      텍스처+격자+전선이 조밀해 dilate가 ROI 전체를 하나의 컨투어로
    #      뭉개버리는 문제가 있어 dilate 대신 블러로 엣지 밀도 자체를 낮춘다)
    roi_eq = cv2.GaussianBlur(roi_eq, (5, 5), 0)

    # 3. Canny 엣지 검출 (실제 창문 검출률을 높이기 위해 임계값 완화)
    edges = cv2.Canny(roi_eq, 30, 90)

    # 4. Contour 검출 (dilate 없이 원본 엣지에서 바로 검출)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    windows = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        roi_h, roi_w = roi.shape[:2]

        # 면적 필터
        min_area = (roi_h * roi_w) * 0.001
        max_area = (roi_h * roi_w) * 0.20
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # 실제 이미지 좌표로 변환
        y_real = y + sky_cut

        # 가로세로 비율 (이 건물 창문은 거의 정사각형~가로 1.5배 수준)
        aspect = bw / bh
        if aspect < 0.6 or aspect > 2.0:
            continue

        # 최소 크기
        if bw < 25 or bh < 25:
            continue

        # 좌우 끝 제거
        if x < w * 0.03 or (x + bw) > w * 0.97:
            continue

        # 채움 비율 (사각형에 가까울수록 창문)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.2:
            continue

        # 크롭 이미지에서 창문 유효성 검사
        crop = img[y_real:y_real+bh, x:x+bw]
        if crop.size == 0:
            continue

        # 크롭 내부 엣지 밀도 확인
        # (창문은 격자 패턴이 있어서 엣지가 일정 수준 이상)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_edges = cv2.Canny(crop_gray, 30, 90)
        edge_density = cv2.countNonZero(crop_edges) / (bw * bh)
        if edge_density < 0.03 or edge_density > 0.4:
            continue

        # 창문 내부 격자 패턴 확인
        # 실제 창문은 내부에 세로/가로 격자(창살)가 있어 수평/수직 엣지가
        # 균등하게 분포하지만, 전선은 대각선 엣지만 있어 한쪽으로 치우친다.
        sobel_h = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_v = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
        h_score = np.mean(np.abs(sobel_h))
        v_score = np.mean(np.abs(sobel_v))
        if h_score == 0 or v_score == 0:
            continue
        hv_ratio = min(h_score, v_score) / max(h_score, v_score)
        if hv_ratio < 0.15:
            continue

        # 창문 내부 밝기 균일성 확인
        # 실제 창문 유리는 내부가 비교적 균일한 색상이지만, 외벽 화강암은
        # 밝기 편차가 크다. (검출률을 높이기 위해 85로 추가 완화)
        std_brightness = np.std(crop_gray)
        if std_brightness > 85:
            continue

        windows.append({
            "bbox": [x, y_real, x+bw, y_real+bh],
            "area": area,
            "crop": crop
        })

    # 겹치는 박스 제거 (NMS)
    windows = remove_overlapping(windows, iou_threshold=0.3)
    return windows


def remove_overlapping(windows: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    if not windows:
        return []

    boxes = [[w["bbox"][0], w["bbox"][1],
              w["bbox"][2], w["bbox"][3]] for w in windows]
    areas = [w["area"] for w in windows]

    keep = []
    idxs = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        remove = []
        for j in idxs:
            xi1 = max(boxes[i][0], boxes[j][0])
            yi1 = max(boxes[i][1], boxes[j][1])
            xi2 = min(boxes[i][2], boxes[j][2])
            yi2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            union = areas[i] + areas[j] - inter
            if union > 0 and inter/union > iou_threshold:
                remove.append(j)
        idxs = [j for j in idxs if j not in remove]

    return [windows[i] for i in keep]


# ════════════════════════════════════════════════════════════════
# YOLO 기반 창문 검출 (이 건물 전용 파인튜닝 모델)
# ════════════════════════════════════════════════════════════════

YOLO_WINDOW_MODEL_PATH = "data/yolo_runs/window_detector_v2/weights/best.pt"
_yolo_window_model = None  # 프레임마다 재로드하지 않도록 모듈 레벨에 캐싱


def _get_yolo_window_model(model_path: str = YOLO_WINDOW_MODEL_PATH):
    global _yolo_window_model
    if _yolo_window_model is None:
        from ultralytics import YOLO
        _yolo_window_model = YOLO(model_path)
    return _yolo_window_model


def detect_windows_yolo(img: np.ndarray, model_path: str = YOLO_WINDOW_MODEL_PATH) -> list[dict]:
    """파인튜닝된 YOLO 모델로 창문을 검출한다.

    crop 저장은 하지 않는다 — 실제 crop 파일 저장은 파이프라인의
    _save_results()가 담당하며, 여기서 자체적으로 저장하면 win_idx가
    프레임마다 1로 리셋돼 다른 프레임/inspection의 crop 파일을 덮어쓰는
    문제가 생긴다.

    Returns:
        [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...]
    """
    if not os.path.exists(model_path):
        log.warning(f"YOLO 모델 없음: {model_path}")
        return []

    model = _get_yolo_window_model(model_path)
    results = model(img, conf=0.25, verbose=False)

    windows = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            windows.append({"bbox": [x1, y1, x2, y2], "confidence": conf})

    return windows


def get_cached_model(model_path: str = YOLO_WINDOW_MODEL_PATH):
    """캐싱된 YOLO 모델을 반환. 파일이 없으면 None."""
    if not os.path.exists(model_path):
        log.warning(f"YOLO 모델 없음: {model_path}")
        return None
    return _get_yolo_window_model(model_path)


def detect_windows_combined(img: np.ndarray) -> list[dict]:
    """YOLO(우선) + Canny/Contour(보완) 병합 검출.

    YOLO가 놓친 영역을 classical CV(엣지+형태 필터)로 보완하고,
    IoU 기반으로 겹치는 박스를 제거한다(YOLO 결과 우선 유지).

    crop은 반환 딕셔너리에 포함하되 파일로 저장하지 않는다 — 실제 저장은
    파이프라인의 _save_results()가 담당한다 (detect_windows_yolo와 동일한 이유).

    Returns:
        [{"window_id": str, "bbox": [x1,y1,x2,y2], "confidence": float,
          "source": "yolo"|"canny", "crop": np.ndarray}, ...]
    """
    h, w = img.shape[:2]
    all_boxes = []

    # ── 1. YOLO 검출 ──
    model = get_cached_model(YOLO_WINDOW_MODEL_PATH)
    if model:
        results = model(img, conf=0.20, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                all_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "source": "yolo"
                })

    # ── 2. Canny + Contour 보완 검출 ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sky_cut = int(h * 0.12)
    bottom_cut = int(h * 0.85)
    roi = gray[sky_cut:bottom_cut, :]
    roi_h, roi_w = roi.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)
    blurred = cv2.GaussianBlur(roi_eq, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = roi_h * roi_w * 0.001
        max_area = roi_h * roi_w * 0.12
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        y_real = y + sky_cut

        aspect = bw / bh
        if aspect < 0.6 or aspect > 2.0:
            continue
        if bw < 40 or bh < 40:
            continue
        if x < w * 0.15 or (x + bw) > w * 0.92:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        if area / hull_area < 0.62:
            continue

        crop = img[y_real:y_real+bh, x:x+bw]
        if crop.size == 0:
            continue

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_edges = cv2.Canny(crop_gray, 30, 90)
        edge_density = cv2.countNonZero(crop_edges) / (bw * bh)
        if edge_density < 0.03 or edge_density > 0.4:
            continue

        std_brightness = np.std(crop_gray)
        if std_brightness > 85:
            continue

        # 창문 유리(하늘색/청록색) vs 콘크리트 벽(무채색) 구분
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_s = np.mean(crop_hsv[:, :, 1])
        mean_v = np.mean(crop_hsv[:, :, 2])
        if mean_s < 40:
            continue   # 채도가 너무 낮으면 콘크리트 벽으로 판단
        if mean_v < 80:
            continue   # 명도가 너무 낮으면 어두운 벽으로 판단

        # 창문 내부 색상 확인 (하늘색/청록색은 B가 R보다 높음, 콘크리트는 R≈G≈B 또는 R>B)
        mean_b = np.mean(crop[:, :, 0])
        mean_r = np.mean(crop[:, :, 2])
        if mean_b < mean_r * 1.05:
            continue

        sobel_h = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_v = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
        h_score = np.mean(np.abs(sobel_h))
        v_score = np.mean(np.abs(sobel_v))
        if h_score == 0 or v_score == 0:
            continue
        if min(h_score, v_score) / max(h_score, v_score) < 0.15:
            continue

        all_boxes.append({
            "bbox": [x, y_real, x+bw, y_real+bh],
            "conf": 0.5,
            "source": "canny"
        })

    # ── 3. NMS로 중복 제거 (YOLO 우선) ──
    if not all_boxes:
        return []

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        xi1 = max(ax1, bx1); yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2); yi2 = min(ay2, by2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        a_area = (ax2-ax1) * (ay2-ay1)
        b_area = (bx2-bx1) * (by2-by1)
        union = a_area + b_area - inter
        return inter/union if union > 0 else 0

    all_boxes.sort(key=lambda x: (
        0 if x["source"] == "yolo" else 1, -x["conf"]
    ))

    keep = []
    for box in all_boxes:
        overlap = False
        for kept in keep:
            if iou(box, kept) > 0.3:
                overlap = True
                break
        if not overlap:
            keep.append(box)

    # ── 4. 결과 반환 ──
    windows = []
    for i, box in enumerate(keep):
        x1, y1, x2, y2 = box["bbox"]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        windows.append({
            "window_id": f"window_{i+1:03d}",
            "bbox": [x1, y1, x2, y2],
            "confidence": box["conf"],
            "source": box["source"],
            "crop": crop
        })

    return windows


# ════════════════════════════════════════════════════════════════
# 오염도 분석 함수 (grader.py 로직 인라인)
# ════════════════════════════════════════════════════════════════

GRADE_THRESHOLDS = [
    ("A",  0,  10),
    ("B", 10,  30),
    ("C", 30,  60),
    ("D", 60, 100),
]


def _contamination_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv         = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    dirty_brown = cv2.inRange(hsv, ( 15, 40,   0), ( 30, 255, 255))
    dirty_dust  = cv2.inRange(hsv, (  0,  0,   0), (180,  30, 149))
    dirty_moss  = cv2.inRange(hsv, ( 35, 40,   0), ( 85, 255, 255))
    contamination = cv2.bitwise_or(dirty_brown, cv2.bitwise_or(dirty_dust, dirty_moss))

    exclude_glass = cv2.inRange(hsv, (100, 50,   0), (130, 255, 255))
    exclude_frame = cv2.inRange(hsv, (  0,  0, 200), (180,  30, 255))
    exclude = cv2.bitwise_or(exclude_glass, exclude_frame)
    contamination = cv2.bitwise_and(contamination, cv2.bitwise_not(exclude))

    kernel = np.ones((5, 5), np.uint8)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_CLOSE, kernel)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_OPEN,  kernel)
    return contamination


def analyze_window(crop_bgr: np.ndarray) -> tuple[str, float]:
    """(grade, contamination_score 0.0~1.0) 반환."""
    mask  = _contamination_mask(crop_bgr)
    pct   = float(np.count_nonzero(mask)) / mask.size * 100
    grade = next((g for g, lo, hi in GRADE_THRESHOLDS if lo <= pct < hi), "D")
    return grade, round(pct / 100.0, 4)


# ════════════════════════════════════════════════════════════════
# 창문 트래커 — 프레임 간 동일 창문 식별
# ════════════════════════════════════════════════════════════════

class WindowTracker:
    def __init__(self, iou_thresh: float = TRACK_IOU_THRESH):
        self.tracks: list[dict] = []
        self.next_id = 1
        self.iou_thresh = iou_thresh

    def update(self, boxes: list[tuple], frame_idx: int) -> list[tuple[int, tuple]]:
        """
        boxes: [(x,y,w,h), ...]
        반환: [(track_id, bbox), ...]  — 이번 프레임에서 매칭된/새로운 트랙 목록
        """
        matched_track_ids = set()
        result = []

        for box in boxes:
            best_iou, best_idx = 0.0, -1
            for i, track in enumerate(self.tracks):
                iou = _iou(box, track["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= self.iou_thresh and best_idx not in matched_track_ids:
                matched_track_ids.add(best_idx)
                self.tracks[best_idx]["bbox"]        = box
                self.tracks[best_idx]["last_frame"]  = frame_idx
                self.tracks[best_idx]["frames_seen"] += 1
                result.append((self.tracks[best_idx]["id"], box))
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks.append({
                    "id": track_id, "bbox": box,
                    "last_frame": frame_idx, "frames_seen": 1,
                })
                result.append((track_id, box))

        return result

    @property
    def unique_count(self) -> int:
        return len(self.tracks)


# ════════════════════════════════════════════════════════════════
# DB 저장
# ════════════════════════════════════════════════════════════════

def _save_results(db, inspection_id: int, building_id: int,
                  frame_idx: int, track_id: int, box: tuple,
                  grade: str, score: float, crop: np.ndarray,
                  results_dir: Path, img_h: int) -> None:
    from app.models.window import WindowResult, ContaminationGrade

    x, y, bw, bh = box
    crop_path = results_dir / f"f{frame_idx:05d}_t{track_id:03d}.jpg"
    cv2.imwrite(str(crop_path), crop)

    wr = WindowResult(
        inspection_id=inspection_id,
        frame_number=frame_idx,
        bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh,
        contamination_score=score,
        grade=ContaminationGrade(grade),
        confidence=1.0,
        crop_image_path=str(crop_path),
    )
    db.add(wr)


def _create_window_records(db, building_id: int, tracker: WindowTracker,
                           img_h: int) -> None:
    from app.models.window import Window

    for track in tracker.tracks:
        x, y, bw, bh = track["bbox"]
        cy_ratio = (y + bh / 2) / img_h if img_h > 0 else 0.5
        # 상/중/하 3구역으로 층수 추정 (1=상, 2=중, 3=하)
        floor_est = 1 if cy_ratio < 0.33 else (2 if cy_ratio < 0.66 else 3)

        win = Window(
            building_id=building_id,
            floor=floor_est,
            position_x=float(x), position_y=float(y),
            width=float(bw), height=float(bh),
        )
        db.add(win)


# ════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ════════════════════════════════════════════════════════════════

def run_pipeline(video_path: str, building_id: int) -> None:
    from app.core.database import SessionLocal, init_db
    from app.models.building import Building
    from app.models.inspection import Inspection, InspectionStatus

    # ── 영상 파일 확인 ────────────────────────────────────────────
    if not os.path.exists(video_path):
        log.error("영상 파일을 찾을 수 없습니다: %s", video_path)
        sys.exit(1)

    # ── DB 초기화 및 연결 ─────────────────────────────────────────
    try:
        init_db()
        db = SessionLocal()
    except Exception as e:
        log.error("DB 연결 실패: %s", e)
        sys.exit(1)

    try:
        # 건물 존재 확인
        building = db.get(Building, building_id)
        if building is None:
            log.error("building_id=%d 가 DB에 없습니다.", building_id)
            sys.exit(1)

        log.info("건물: %s (id=%d)", building.name, building_id)

        # Inspection 레코드 생성
        inspection = Inspection(
            building_id=building_id,
            video_filename=os.path.basename(video_path),
            status=InspectionStatus.PROCESSING,
        )
        db.add(inspection)
        db.commit()
        db.refresh(inspection)
        log.info("Inspection 생성 (id=%d)", inspection.id)

        results_dir = Path("results") / str(inspection.id)
        results_dir.mkdir(parents=True, exist_ok=True)

        # ── 영상 열기 ─────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        inspection.total_frames = total_frames
        db.commit()

        log.info("영상 정보: 총 %d 프레임 / %.1f fps", total_frames, fps)

        tracker        = WindowTracker()
        frame_idx      = 0
        sampled_count  = 0
        total_detected = 0
        img_h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        grade_counts   = {"A": 0, "B": 0, "C": 0, "D": 0}

        # ── 프레임 루프 ───────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLE_RATE == 0:
                if not has_windows(frame):
                    log.info(f"창문 없는 프레임 스킵: frame_idx={frame_idx}")
                    tracked = []
                else:
                    edge_results = detect_windows_combined(frame)
                    boxes = [(r["bbox"][0], r["bbox"][1],
                              r["bbox"][2] - r["bbox"][0],
                              r["bbox"][3] - r["bbox"][1]) for r in edge_results]
                    tracked = tracker.update(boxes, frame_idx)

                    for track_id, box in tracked:
                        x, y, bw, bh = box
                        crop = frame[y:y+bh, x:x+bw]
                        if not is_valid_window(crop):
                            continue                 # 외벽 오탐 — 유효한 창문 아님

                        grade, score = analyze_window(crop)
                        grade_counts[grade] += 1
                        total_detected += 1

                        _save_results(db, inspection.id, building_id,
                                      frame_idx, track_id, box,
                                      grade, score, crop, results_dir, img_h)

                    elapsed = frame_idx / fps
                    log.info("[%6.1fs | 프레임 %5d] 창문 %d개 감지",
                             elapsed, frame_idx, len(tracked))

                inspection.processed_frames = sampled_count + 1
                inspection.total_windows    = total_detected
                db.commit()
                sampled_count += 1

            frame_idx += 1

        cap.release()

        # ── 고유 창문 Window 레코드 저장 ─────────────────────────
        _create_window_records(db, building_id, tracker, img_h)
        inspection.status = InspectionStatus.COMPLETED
        db.commit()

        # ── 요약 리포트 ───────────────────────────────────────────
        _print_summary(building.name, inspection.id, sampled_count,
                       tracker, grade_counts)

    except Exception as e:
        log.error("파이프라인 오류: %s", e, exc_info=True)
        try:
            from app.models.inspection import InspectionStatus
            inspection.status = InspectionStatus.FAILED
            db.commit()
        except Exception:
            pass
        sys.exit(1)
    finally:
        db.close()


def _print_summary(building_name: str, inspection_id: int,
                   sampled_frames: int, tracker: WindowTracker,
                   grade_counts: dict) -> None:
    total = sum(grade_counts.values())
    log.info("")
    log.info("═" * 52)
    log.info("  분석 완료 — %s  (inspection #%d)", building_name, inspection_id)
    log.info("  처리 프레임: %d  /  고유 창문: %d개  /  탐지 누적: %d건",
             sampled_frames, tracker.unique_count, total)
    log.info("  ─────────────────────────────────────────────")
    labels = {"A": "청결(0~10%)", "B": "보통(10~30%)",
              "C": "오염(30~60%)", "D": "심각(60%+)"}
    for g in "ABCD":
        cnt  = grade_counts[g]
        pct  = cnt / total * 100 if total > 0 else 0
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        log.info("  %s등급 %-12s [%s] %3d건 (%4.1f%%)",
                 g, labels[g], bar, cnt, pct)
    log.info("═" * 52)


# ════════════════════════════════════════════════════════════════
# 웹 API 백그라운드 태스크용 클래스
# ════════════════════════════════════════════════════════════════

class InspectionPipeline:
    """웹 API 백그라운드 태스크에서 호출하는 파이프라인."""

    def process(self, inspection, video_path: str, db) -> None:
        from app.models.inspection import InspectionStatus
        from app.models.building import Building

        if not os.path.exists(video_path):
            log.error("영상 파일을 찾을 수 없습니다: %s", video_path)
            inspection.status = InspectionStatus.FAILED
            db.commit()
            return

        building = db.get(Building, inspection.building_id)
        if building is None:
            log.error("building_id=%d 가 DB에 없습니다.", inspection.building_id)
            inspection.status = InspectionStatus.FAILED
            db.commit()
            return

        inspection.status = InspectionStatus.PROCESSING
        db.commit()
        log.info("파이프라인 시작: inspection_id=%d, video=%s", inspection.id, video_path)

        try:
            results_dir = Path("results") / str(inspection.id)
            results_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            img_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            inspection.total_frames = total_frames
            db.commit()

            log.info("영상 정보: 총 %d 프레임 / %.1f fps", total_frames, fps)

            tracker        = WindowTracker()
            frame_idx      = 0
            sampled_count  = 0
            total_detected = 0
            grade_counts   = {"A": 0, "B": 0, "C": 0, "D": 0}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SAMPLE_RATE == 0:
                    if not has_windows(frame):
                        log.info(f"창문 없는 프레임 스킵: frame_idx={frame_idx}")
                        tracked = []
                    else:
                        edge_results = detect_windows_combined(frame)
                        boxes = [(r["bbox"][0], r["bbox"][1],
                                  r["bbox"][2] - r["bbox"][0],
                                  r["bbox"][3] - r["bbox"][1]) for r in edge_results]
                        tracked = tracker.update(boxes, frame_idx)

                        for track_id, box in tracked:
                            x, y, bw, bh = box
                            crop = frame[y:y+bh, x:x+bw]
                            if not is_valid_window(crop):
                                continue             # 외벽 오탐 — 유효한 창문 아님

                            grade, score = analyze_window(crop)
                            grade_counts[grade] += 1
                            total_detected += 1
                            _save_results(db, inspection.id, inspection.building_id,
                                          frame_idx, track_id, box,
                                          grade, score, crop, results_dir, img_h)

                        elapsed = frame_idx / fps
                        log.info("[%6.1fs | 프레임 %5d] 창문 %d개 감지",
                                 elapsed, frame_idx, len(tracked))

                    inspection.processed_frames = sampled_count + 1
                    inspection.total_windows    = total_detected
                    db.commit()
                    sampled_count += 1

                frame_idx += 1

            cap.release()

            _create_window_records(db, inspection.building_id, tracker, img_h)
            inspection.status = InspectionStatus.COMPLETED
            db.commit()
            log.info("파이프라인 완료: inspection_id=%d → COMPLETED", inspection.id)

            _print_summary(building.name, inspection.id, sampled_count,
                           tracker, grade_counts)

        except Exception as e:
            log.error("파이프라인 오류 (inspection_id=%d): %s", inspection.id, e, exc_info=True)
            try:
                inspection.status = InspectionStatus.FAILED
                db.commit()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════
# CLI 진입점
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="창문 오염도 검사 파이프라인")
    parser.add_argument("--video",       required=True, help="분석할 영상 파일 경로")
    parser.add_argument("--building_id", required=True, type=int, help="DB buildings.id")
    args = parser.parse_args()

    run_pipeline(args.video, args.building_id)
