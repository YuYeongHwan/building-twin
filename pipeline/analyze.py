"""
오염도 분석 모듈.

기존 HSV 오염도 분석 + pollution_index (0~1) 수치 추가.
기준 이미지(baseline)가 없으면 첫 분석 결과를 자동 저장.
결과는 data/results.json 에 누적 저장.

pollution_index = (채도편차 * 0.6 + 명도편차 * 0.4) / 255
grade: A 0.0~0.1 / B 0.1~0.3 / C 0.3~0.6 / D 0.6~1.0

반환: {"window_id": ..., "pollution_index": ..., "grade": ...}

실행: python pipeline/analyze.py [이미지_경로]
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT         = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "data" / "baseline"
RESULTS_FILE = ROOT / "data" / "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HSV 오염도 분석 (기존 project/pipeline/analyze.py 로직)
# ---------------------------------------------------------------------------

_DIRTY_RANGES = [
    ((15,  40,   0), (30,  255, 255)),   # 갈색/황색 얼룩
    (( 0,   0,   0), (180,  30, 149)),   # 회색/검은 먼지
    ((35,  40,   0), (85,  255, 255)),   # 녹색 이끼
]
_CLEAN_RANGES = [
    ((100, 50, 0), (130, 255, 255)),     # 파란 유리 (하늘·반사)
    ((  0,  0, 200), (180, 30, 255)),    # 흰색 창틀
]
_MORPH_KERNEL = np.ones((5, 5), np.uint8)

GRADE_COLORS = {
    "A": (0, 255,   0),
    "B": (0, 255, 255),
    "C": (0, 165, 255),
    "D": (0,   0, 255),
}


def build_contamination_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """오염 픽셀 바이너리 마스크(0/255) 생성."""
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    dirt = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
    for lo, hi in _DIRTY_RANGES:
        dirt = cv2.bitwise_or(dirt, cv2.inRange(hsv, lo, hi))

    clean = np.zeros_like(dirt)
    for lo, hi in _CLEAN_RANGES:
        clean = cv2.bitwise_or(clean, cv2.inRange(hsv, lo, hi))

    mask = cv2.bitwise_and(dirt, cv2.bitwise_not(clean))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    return mask


def make_heatmap(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """오염 마스크를 JET 컬러맵으로 원본에 반투명 합성."""
    return cv2.addWeighted(crop_bgr, 0.5,
                           cv2.applyColorMap(mask, cv2.COLORMAP_JET), 0.5, 0)


def contamination_pct(mask: np.ndarray) -> float:
    """오염 픽셀 비율 (0~100)."""
    return float(np.count_nonzero(mask)) / mask.size * 100


# ---------------------------------------------------------------------------
# pollution_index 계산
# ---------------------------------------------------------------------------

def compute_pollution_index(crop_bgr: np.ndarray) -> float:
    """
    HSV 채도/명도 편차 기반 오염 지수 (0~1).

    index = (std_S * 0.6 + std_V * 0.4) / 255
    """
    hsv   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    std_s = float(np.std(hsv[:, :, 1]))
    std_v = float(np.std(hsv[:, :, 2]))
    index = (std_s * 0.6 + std_v * 0.4) / 255.0
    return round(min(1.0, max(0.0, index)), 4)


def assign_grade(index: float) -> str:
    if index < 0.1:
        return "A"
    if index < 0.3:
        return "B"
    if index < 0.6:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Baseline 관리
# ---------------------------------------------------------------------------

def save_baseline(window_id: str, crop_bgr: np.ndarray) -> None:
    """기준 이미지가 없으면 첫 분석 결과를 baseline으로 저장."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{window_id}_baseline.jpg"
    if not path.exists():
        cv2.imwrite(str(path), crop_bgr)
        log.info(f"기준 이미지 저장: {path.name}")


# ---------------------------------------------------------------------------
# 결과 JSON 저장
# ---------------------------------------------------------------------------

def save_results(results: list[dict]) -> None:
    """data/results.json 에 window_id 기준으로 upsert."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["window_id"]] = item
    for r in results:
        existing[r["window_id"]] = r
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(existing.values()), f, indent=2, ensure_ascii=False)
    log.info(f"결과 저장: {RESULTS_FILE}  ({len(existing)}개 창문)")


# ---------------------------------------------------------------------------
# 창문 단위 분석
# ---------------------------------------------------------------------------

def analyze_window_crop(window_id: str, crop_bgr: np.ndarray) -> dict:
    """
    단일 창문 크롭을 분석하고 결과 딕셔너리 반환.

    Returns:
        {"window_id": ..., "pollution_index": ..., "grade": ...}
    """
    save_baseline(window_id, crop_bgr)

    mask  = build_contamination_mask(crop_bgr)
    pct   = contamination_pct(mask)
    index = compute_pollution_index(crop_bgr)
    grade = assign_grade(index)

    return {
        "window_id":       window_id,
        "pollution_index": index,
        "grade":           grade,
        "contamination_pct": round(pct, 2),
    }


# ---------------------------------------------------------------------------
# 배치 분석 진입점
# ---------------------------------------------------------------------------

def run_analyze(detections: list[dict], image: np.ndarray) -> list[dict]:
    """
    detect.py 결과 목록을 받아 각 창문을 분석하고 results.json 에 저장.

    Args:
        detections : detect_windows() 반환값
        image      : 원본 BGR 이미지

    Returns:
        분석 결과 리스트
    """
    results = []
    for det in detections:
        wid        = det["window_id"]
        x1, y1, x2, y2 = det["bbox"]
        crop       = image[y1:y2, x1:x2]
        if crop.size == 0:
            log.warning(f"{wid}: 크롭 이미지 비어 있음, 건너뜀")
            continue
        result = analyze_window_crop(wid, crop)
        results.append(result)
        log.info(
            f"  {wid}  pollution_index={result['pollution_index']:.4f}"
            f"  grade={result['grade']}"
            f"  contamination={result['contamination_pct']:.1f}%"
        )

    if results:
        avg = sum(r["pollution_index"] for r in results) / len(results)
        grade_dist = {}
        for r in results:
            grade_dist[r["grade"]] = grade_dist.get(r["grade"], 0) + 1
        dist_str = "  ".join(f"{g}:{n}" for g, n in sorted(grade_dist.items()))
        log.info(f"분석 완료  {len(results)}개 창문 · 평균 index={avg:.4f}  [{dist_str}]")

    save_results(results)
    return results


# ---------------------------------------------------------------------------
# 단독 실행
# ---------------------------------------------------------------------------

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

    # detect → analyze 연계 실행
    sys.path.insert(0, str(ROOT / "pipeline"))
    from detect import detect_windows  # noqa: E402

    img = cv2.imread(target)
    if img is None:
        log.error(f"이미지를 열 수 없습니다: {target}")
        sys.exit(1)

    log.info(f"분석 대상: {Path(target).name}")
    dets    = detect_windows(img)
    results = run_analyze(dets, img)

    for r in results:
        print(r)
