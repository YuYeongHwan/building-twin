"""
오염도 분석 모듈.
창문 크롭 이미지를 받아 오염 마스크 생성, 등급 산출, 히트맵 반환.
"""

import cv2
import numpy as np

GRADE_THRESHOLDS = [
    ("A",  0,  10),
    ("B", 10,  30),
    ("C", 30,  60),
    ("D", 60, 100),
]

GRADE_COLORS = {
    "A": (0, 255,   0),
    "B": (0, 255, 255),
    "C": (0, 165, 255),
    "D": (0,   0, 255),
}


def build_contamination_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """
    창문 크롭에서 오염 픽셀 마스크(0/255) 생성.

    오염 픽셀 (양성):
      - 갈색/황색 얼룩 : H 15-30,  S > 40
      - 회색/검은 먼지 : S < 30,   V < 150
      - 녹색 이끼      : H 35-85,  S > 40

    제외 (음성 — 정상 창문):
      - 파란 유리      : H 100-130, S > 50  (하늘·반사)
      - 흰색 창틀      : S < 30,   V > 200
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    dirty_brown = cv2.inRange(hsv, ( 15, 40,   0), ( 30, 255, 255))
    dirty_dust  = cv2.inRange(hsv, (  0,  0,   0), (180,  30, 149))
    dirty_moss  = cv2.inRange(hsv, ( 35, 40,   0), ( 85, 255, 255))

    contamination = cv2.bitwise_or(dirty_brown, cv2.bitwise_or(dirty_dust, dirty_moss))

    exclude_glass = cv2.inRange(hsv, (100, 50,   0), (130, 255, 255))
    exclude_frame = cv2.inRange(hsv, (  0,  0, 200), (180,  30, 255))
    exclude = cv2.bitwise_or(exclude_glass, exclude_frame)
    contamination = cv2.bitwise_and(contamination, cv2.bitwise_not(exclude))

    kernel        = np.ones((5, 5), np.uint8)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_CLOSE, kernel)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_OPEN,  kernel)

    return contamination


def make_heatmap(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """오염 마스크를 JET 컬러맵으로 변환 후 원본에 반투명 합성."""
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(crop_bgr, 0.5, colored, 0.5, 0)


def assign_grade(pct: float) -> str:
    for grade, lo, hi in GRADE_THRESHOLDS:
        if lo <= pct < hi:
            return grade
    return "D"


def analyze_window(crop_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Returns:
        grade   : A/B/C/D
        pct     : 오염 픽셀 비율 (0~100)
        heatmap : 히트맵 이미지
    """
    mask    = build_contamination_mask(crop_bgr)
    pct     = float(np.count_nonzero(mask)) / mask.size * 100
    grade   = assign_grade(pct)
    heatmap = make_heatmap(crop_bgr, mask)
    return grade, pct, heatmap


def draw_grade_box(canvas: np.ndarray, x: int, y: int,
                   bw: int, bh: int, idx: int, grade: str, pct: float) -> None:
    color = GRADE_COLORS[grade]
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)

    label = f"#{idx} {grade} {pct:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ly = y - 4 if y - th - 8 >= 0 else y + th + 6
    cv2.rectangle(canvas, (x, ly - th - 4), (x + tw + 6, ly + 2), color, -1)
    cv2.putText(canvas, label, (x + 3, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
