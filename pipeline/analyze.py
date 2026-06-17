"""
오염도 분석 모듈 (흰색 A4 용지 모형 전용).

흰색에 가까울수록 오염도 낮음(A등급), 어두울수록 높음(D등급).
HSV V채널 평균값만으로 오염 지수를 단순 계산한다.

pollution_index = 1.0 - (mean_V / 255)
grade: A 0.0~0.1 / B 0.1~0.3 / C 0.3~0.6 / D 0.6~1.0

출력:
  - data/results/<window_id>_crop.jpg  (창문 크롭 이미지)
  - data/results.json                  (분석 결과 누적)
  - data/baseline/<window_id>_baseline.jpg (최초 1회만)

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
RESULTS_DIR  = ROOT / "data" / "results"
RESULTS_FILE = ROOT / "data" / "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 오염 지수 계산
# ---------------------------------------------------------------------------

def compute_pollution_index(crop_bgr: np.ndarray) -> float:
    """
    HSV V채널 평균 기반 오염 지수 (0~1).

    흰 A4용지 기준: 밝을수록 깨끗(index ≈ 0), 어두울수록 오염(index ≈ 1).
    index = 1.0 - (mean_V / 255)
    """
    hsv    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    index  = 1.0 - (mean_v / 255.0)
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
# 크롭 이미지 저장
# ---------------------------------------------------------------------------

def save_crop(window_id: str, crop_bgr: np.ndarray) -> str:
    """창문 크롭 이미지를 data/results/<window_id>_crop.jpg 에 저장."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{window_id}_crop.jpg"
    cv2.imwrite(str(path), crop_bgr)
    return str(path)


# ---------------------------------------------------------------------------
# 결과 JSON 저장
# ---------------------------------------------------------------------------

def save_results(results: list[dict]) -> None:
    """data/results.json 에 window_id 기준으로 upsert."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                for item in json.load(f):
                    existing[item["window_id"]] = item
        except (json.JSONDecodeError, KeyError):
            pass
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
        {
          "window_id": ...,
          "pollution_index": ...,
          "grade": ...,
          "crop_image_path": "data/results/<window_id>_crop.jpg"
        }
    """
    save_baseline(window_id, crop_bgr)
    crop_path = save_crop(window_id, crop_bgr)
    index     = compute_pollution_index(crop_bgr)
    grade     = assign_grade(index)

    return {
        "window_id":       window_id,
        "pollution_index": index,
        "grade":           grade,
        "crop_image_path": crop_path,
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
