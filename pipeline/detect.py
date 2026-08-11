"""
창문 검출 모듈 (실제 건물 촬영 영상 전용).

대상 건물 특성:
  - 외벽: 회색 화강암 + 진한 네이비 컬러 패널
  - 창문 프레임: 흰색
  - 창문 유리: 하늘색/청록색 계열이지만 반사·그림자에 따라 밝기가 매우 넓게 변함
  - 전선이 건물 앞을 가로지름

실측 결과, 이 건물 사진은 유리·프레임·패널·벽이 모두 비슷한 Hue(색상)를 가져
(흐린 날씨/화이트밸런스로 전체적으로 푸르스름한 색조) Hue 기반으로는 유리를
구분할 수 없었다. 대신 "흰색 프레임"만 밝기(V)·채도(S)로 안정적으로 구분되므로,
프레임의 사각 테두리를 검출한 뒤 안쪽을 채워 창문 영역(창틀+유리)으로 잡는다.
화강암 벽의 반짝이는 노이즈와 구분하기 위해 컨투어의 사각형 근사(폴리곤 근사)와
바운딩박스 대비 채움 비율(extent)로 필터링한다.

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

# 흰색 창틀 프레임 (HSV) — 밝고(V 높음) 채도 낮음(S 낮음)
FRAME_V_MIN = 185
FRAME_S_MAX = 95

SKY_CUTOFF_RATIO    = 0.15   # 상단 15%는 하늘로 간주해 마스킹 제외
BOTTOM_CUTOFF_RATIO = 0.90   # 하단 10%는 바닥(타일 등)으로 간주해 제거
SIDE_MARGIN_RATIO   = 0.05   # 좌우 끝 5%는 측면 잡음으로 간주해 제거

MIN_AREA_RATIO = 0.001   # 전체 이미지 면적의 0.1% 미만이면 노이즈로 제거
MAX_AREA_RATIO = 0.03    # 전체 이미지 면적의 3% 초과면 벽 노이즈 덩어리로 간주해 제거
MIN_SIDE_PX    = 40      # 가로/세로 최소 픽셀
MIN_ASPECT     = 0.4     # 창문은 정사각형~가로형이 대부분
MAX_ASPECT     = 2.2
MIN_EXTENT     = 0.5     # 바운딩박스 대비 컨투어 채움 비율(사각형에 가까울수록 1에 근접)
MIN_POLY_SIDES = 4        # 폴리곤 근사 꼭짓점 최소/최대 개수 (사각형 판별용)
MAX_POLY_SIDES = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 흰 프레임 테두리 기반 창문 검출
# ---------------------------------------------------------------------------

def detect_windows(image: np.ndarray) -> list[dict]:
    """
    흰색 프레임 테두리 마스크 → 모폴로지로 사각형 채움 → Contour → 사각형 근사
    필터로 창문을 검출한다.

    Args:
        image: BGR 이미지 (cv2.imread 결과)

    Returns:
        [{"window_id": "window_001", "bbox": [x1, y1, x2, y2]}, ...]
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    s_ch = hsv[:, :, 1].astype(np.int16)
    v_ch = hsv[:, :, 2].astype(np.int16)

    mask = ((v_ch >= FRAME_V_MIN) & (s_ch <= FRAME_S_MAX)).astype(np.uint8) * 255

    # 하늘(sky) 영역 제거 - 상단 15% 마스킹
    sky_cutoff = int(h * SKY_CUTOFF_RATIO)
    mask[:sky_cutoff, :] = 0

    # 화강암 벽의 미세한 반짝임 노이즈 제거 후, 프레임 테두리를 사각형으로 채움
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area      = h * w
    min_area      = img_area * MIN_AREA_RATIO
    max_area      = img_area * MAX_AREA_RATIO
    bottom_cutoff = int(h * BOTTOM_CUTOFF_RATIO)
    left_cutoff   = int(w * SIDE_MARGIN_RATIO)
    right_cutoff  = int(w * (1 - SIDE_MARGIN_RATIO))
    detections    = []
    win_num       = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue                     # 노이즈 또는 벽 노이즈 덩어리

        x, y, bw, bh = cv2.boundingRect(cnt)

        if y > bottom_cutoff:
            continue                     # 이미지 하단(바닥 타일 등) 제거

        if x < left_cutoff or (x + bw) > right_cutoff:
            continue                     # 이미지 좌우 끝(측면 잡음) 제거

        if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX:
            continue                     # 창문이라기엔 너무 작음

        aspect_ratio = bw / bh
        if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
            continue                     # 전선 등 가늘고 긴 형태 배제

        # 사각형 판별: 폴리곤 근사 꼭짓점 개수 + 바운딩박스 대비 채움 비율
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) < MIN_POLY_SIDES or len(approx) > MAX_POLY_SIDES:
            continue

        extent = area / (bw * bh)
        if extent < MIN_EXTENT:
            continue                     # 화강암 노이즈 등 불규칙한 형태 배제

        wid = f"window_{win_num:03d}"
        detections.append({"window_id": wid, "bbox": [x, y, x + bw, y + bh]})
        win_num += 1

    # y 좌표 기준 위→아래 정렬
    detections.sort(key=lambda d: d["bbox"][1])

    log.info(f"창문 검출 완료: {len(detections)}개  [흰 프레임 테두리 검출]")
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
