"""
창문 탐지 모듈.
Canny + contour 기반으로 이미지에서 창문 영역 [(x, y, w, h), ...] 반환.
"""

import cv2
import numpy as np

MIN_AREA    = 4500
ASPECT_MIN  = 0.5
ASPECT_MAX  = 2.5
NMS_THRESH  = 0.05
BORDER_PAD  = 20
DUP_XY      = 30
MIN_BRIGHT  = 80

GLASS_PIXEL_MIN_RATIO = 0.20
BRICK_RATIO_MAX       = 0.30
EDGE_DENSITY_MAX      = 0.15
COLOR_VAR_MAX_V       = 65
COLOR_VAR_MAX_S       = 55


def is_in_bounds(x: int, y: int, bw: int, bh: int, img_w: int, img_h: int) -> bool:
    return (x >= BORDER_PAD and y >= BORDER_PAD
            and x + bw <= img_w - BORDER_PAD
            and y + bh <= img_h - BORDER_PAD)


def brightness_ok(region_hsv: np.ndarray) -> bool:
    if region_hsv.size == 0:
        return False
    return float(np.mean(region_hsv[:, :, 2])) >= MIN_BRIGHT


def has_grid_pattern(region_bgr: np.ndarray) -> bool:
    if region_bgr.size == 0:
        return False
    rh, rw = region_bgr.shape[:2]
    if rh < 10 or rw < 10:
        return False
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    margin_y = max(1, rh // 10)
    margin_x = max(1, rw // 10)
    border = np.concatenate([
        gray[:margin_y, :].flatten(),
        gray[-margin_y:, :].flatten(),
        gray[margin_y:-margin_y, :margin_x].flatten(),
        gray[margin_y:-margin_y, -margin_x:].flatten(),
    ])
    hsv_roi = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    interior = hsv_roi[margin_y:-margin_y, margin_x:-margin_x]
    if interior.size == 0:
        return False
    blue_mask  = cv2.inRange(interior, (90, 20, 60), (130, 255, 255))
    blue_ratio = float(np.count_nonzero(blue_mask)) / interior[:, :, 0].size
    return float(np.mean(border)) >= 160 and blue_ratio >= 0.05


def is_glass_color(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    total   = roi_hsv.shape[0] * roi_hsv.shape[1]
    neutral = (roi_hsv[:, :, 1] < 80) & (roi_hsv[:, :, 2] > 80)
    blue    = ((roi_hsv[:, :, 0] >= 90) & (roi_hsv[:, :, 0] <= 130)
               & (roi_hsv[:, :, 2] > 50))
    return float(np.count_nonzero(neutral | blue)) / total >= GLASS_PIXEL_MIN_RATIO


def is_brick(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    total = roi_hsv.shape[0] * roi_hsv.shape[1]
    lo    = cv2.inRange(roi_hsv, (0,   60, 0), (20,  255, 255))
    hi    = cv2.inRange(roi_hsv, (160, 60, 0), (180, 255, 255))
    return float(np.count_nonzero(cv2.bitwise_or(lo, hi))) / total >= BRICK_RATIO_MAX


def is_complex_structure(roi_bgr: np.ndarray) -> bool:
    if roi_bgr.size == 0:
        return False
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    return float(np.count_nonzero(edges)) / edges.size >= EDGE_DENSITY_MAX


def is_color_uniform(roi_hsv: np.ndarray) -> bool:
    if roi_hsv.size == 0:
        return False
    return (float(np.std(roi_hsv[:, :, 2])) <= COLOR_VAR_MAX_V
            and float(np.std(roi_hsv[:, :, 1])) <= COLOR_VAR_MAX_S)


def _area_score(bw: int, bh: int, median_area: float) -> float:
    area = bw * bh
    if median_area == 0:
        return 0.0
    ratio = area / median_area
    if ratio < 0.25 or ratio > 4.0:
        return 0.0
    return 1.0 - abs(1.0 - ratio) / max(1.0, ratio)


def nms(boxes: list[tuple], iou_thresh: float) -> list[tuple]:
    if not boxes:
        return []
    areas  = [bw * bh for _, _, bw, bh in boxes]
    median = float(np.median(areas))
    boxes  = sorted(boxes, key=lambda b: _area_score(b[2], b[3], median), reverse=True)
    kept   = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for k in kept:
            x2, y2, w2, h2 = k
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            if union > 0 and inter / union > iou_thresh:
                dominated = True
                break
        if not dominated:
            kept.append(box)
    return kept


def dedup_by_position(boxes: list[tuple]) -> list[tuple]:
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        duplicate = False
        for i, k in enumerate(kept):
            x2, y2, w2, h2 = k
            if abs(x1 - x2) <= DUP_XY and abs(y1 - y2) <= DUP_XY:
                if w1 * h1 > w2 * h2:
                    kept[i] = box
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def detect_windows(frame: np.ndarray) -> list[tuple]:
    """Canny + contour + 다중 필터로 창문 박스 [(x, y, w, h), ...] 반환."""
    img_h, img_w = frame.shape[:2]
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

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
        if not is_in_bounds(x, y, bw, bh, img_w, img_h):
            continue

        roi_hsv = hsv[y:y + bh, x:x + bw]
        roi_bgr = frame[y:y + bh, x:x + bw]

        if not brightness_ok(roi_hsv) and not has_grid_pattern(roi_bgr):
            continue
        if not is_glass_color(roi_hsv):
            continue
        if is_brick(roi_hsv):
            continue
        if is_complex_structure(roi_bgr):
            continue
        if not is_color_uniform(roi_hsv):
            continue

        candidates.append((x, y, bw, bh))

    boxes = nms(candidates, NMS_THRESH)
    boxes = dedup_by_position(boxes)
    boxes.sort(key=lambda b: b[1])
    return boxes
