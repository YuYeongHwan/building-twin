"""
building-twin 전체 파이프라인 자동 실행 스크립트.

STEP 1  영상 파일 자동 감지     (data/raw_video/)
STEP 2  영상 전처리              (preprocess.run_preprocess)
STEP 3  3D 재구성 — COLMAP      (reconstruct.run_colmap)
STEP 4  3D 재구성 — 3DGS        (reconstruct.run_3dgs)
STEP 5  창문 검출                (detect.detect_windows)
STEP 6  오염도 분석              (analyze.run_analyze → data/results.json)
STEP 7  분석 결과 DB 저장        (POST /analysis/result)
STEP 8  웹 서버 시작 + 브라우저  (uvicorn main:app)

실행: python pipeline/run_all.py
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_VIDEO_DIR  = ROOT / "data" / "raw_video"
FRAMES_DIR     = ROOT / "data" / "frames"
SPARSE_DIR     = ROOT / "data" / "sparse"
DATA_DIR       = ROOT / "data"
RESULTS_FILE   = ROOT / "data" / "results.json"
DASHBOARD_URL  = "http://localhost:8000/dashboard"
API_RESULT_URL = "http://localhost:8000/analysis/result"
UVICORN_CMD    = ["uvicorn", "main:app", "--reload"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def step_header(n: int, name: str) -> float:
    print(f"\n===== STEP {n}: {name} 시작 =====")
    return time.time()


def step_done(t0: float) -> None:
    elapsed = time.time() - t0
    print(f"  완료 — 소요 시간: {elapsed:.1f}초")


# ---------------------------------------------------------------------------
# reconstruct 함수 (COLMAP + 3DGS 외부 바이너리 호출)
# ---------------------------------------------------------------------------

class reconstruct:
    @staticmethod
    def run_colmap(frames_dir: str, sparse_dir: str) -> None:
        """COLMAP feature_extractor → exhaustive_matcher → mapper 순서 실행."""
        Path(sparse_dir).mkdir(parents=True, exist_ok=True)
        db = str(Path(sparse_dir) / "database.db")
        steps = [
            ["colmap", "feature_extractor",
             "--database_path", db, "--image_path", frames_dir],
            ["colmap", "exhaustive_matcher",
             "--database_path", db],
            ["colmap", "mapper",
             "--database_path", db,
             "--image_path", frames_dir,
             "--output_path", sparse_dir],
        ]
        for cmd in steps:
            log.info(f"  $ {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(ROOT))

    @staticmethod
    def run_3dgs(data_dir: str) -> None:
        """3D Gaussian Splatting 학습 실행 (train.py 호출)."""
        sparse_dir = str(Path(data_dir) / "sparse")
        splat_dir  = str(Path(data_dir) / "splat")
        Path(splat_dir).mkdir(parents=True, exist_ok=True)
        cmd = ["python", "train.py", "-s", sparse_dir, "-m", splat_dir]
        log.info(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ROOT))


# ---------------------------------------------------------------------------
# 파이프라인 헬퍼
# ---------------------------------------------------------------------------

def find_latest_video() -> Path:
    """data/raw_video/ 에서 가장 최근 수정된 .mp4/.mov 파일 반환."""
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    videos = [
        p for p in RAW_VIDEO_DIR.iterdir()
        if p.suffix.lower() in {".mp4", ".mov"}
    ]
    if not videos:
        raise FileNotFoundError(
            f"data/raw_video/ 에 .mp4/.mov 파일이 없습니다.\n경로: {RAW_VIDEO_DIR}"
        )
    return max(videos, key=lambda p: p.stat().st_mtime)


def detect_all_frames(frames_root: Path) -> list[tuple]:
    """frames_root 하위 전체 이미지에서 창문 검출."""
    from pipeline.detect import detect_windows
    images = sorted(frames_root.rglob("*.jpg")) + sorted(frames_root.rglob("*.png"))
    if not images:
        log.warning(f"프레임 이미지가 없습니다: {frames_root}")
        return []
    results = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        dets = detect_windows(frame)
        if dets:
            results.append((img_path, frame, dets))
    return results


def post_analysis_result() -> bool:
    """POST /analysis/result 로 results.json → DB 저장 요청."""
    try:
        req = urllib.request.Request(
            API_RESULT_URL,
            method="POST",
            data=b"",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
            log.info(f"  DB 저장 완료: {body}")
            return True
    except urllib.error.URLError as e:
        log.warning(f"  API 호출 실패 (서버 미실행?): {e.reason}")
        return False


def wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """서버가 응답할 때까지 최대 timeout 초 대기."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()

    # ── STEP 1: 영상 파일 자동 감지 ───────────────────────────────────────
    t = step_header(1, "영상 파일 자동 감지")
    try:
        video_path = find_latest_video()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    log.info(f"  대상 영상: {video_path.name}  ({video_path.stat().st_size // 1024} KB)")
    step_done(t)

    # ── STEP 2: 전처리 ────────────────────────────────────────────────────
    t = step_header(2, "영상 전처리 (프레임 추출 + 블러 필터링 + CLAHE)")
    from pipeline.preprocess import run_preprocess
    frame_count = run_preprocess(str(video_path))
    log.info(f"  저장된 프레임 수: {frame_count}")
    step_done(t)

    # ── STEP 3: COLMAP ────────────────────────────────────────────────────
    t = step_header(3, "3D 재구성 — COLMAP SfM")
    frames_dir = str(FRAMES_DIR / video_path.stem)
    try:
        reconstruct.run_colmap(frames_dir, str(SPARSE_DIR))
        log.info(f"  COLMAP 완료 → {SPARSE_DIR}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.error("COLMAP 실패: 폼보드 모형 촬영 후 다시 시도하세요.")
        sys.exit(1)
    step_done(t)

    # ── STEP 4: 3D Gaussian Splatting ────────────────────────────────────
    t = step_header(4, "3D 재구성 — 3D Gaussian Splatting")
    try:
        reconstruct.run_3dgs(str(DATA_DIR))
        log.info(f"  3DGS 완료 → {DATA_DIR / 'splat'}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning(f"  3DGS 실패 (이후 단계 계속 진행): {e}")
    step_done(t)

    # ── STEP 5: 창문 검출 ─────────────────────────────────────────────────
    t = step_header(5, "창문 검출 (YOLO v8 → Canny+Contour 폴백)")
    detections_per_frame = detect_all_frames(FRAMES_DIR)
    total_wins = sum(len(d) for _, _, d in detections_per_frame)
    log.info(f"  검출 완료: {total_wins}개 창문  ({len(detections_per_frame)}개 프레임에서 발견)")
    step_done(t)

    # ── STEP 6: 오염도 분석 ───────────────────────────────────────────────
    t = step_header(6, "오염도 분석 (HSV Pollution Index → data/results.json)")
    from pipeline.analyze import run_analyze
    all_results: list[dict] = []
    for _img_path, frame, dets in detections_per_frame:
        all_results.extend(run_analyze(dets, frame))

    if all_results:
        avg_idx = sum(r["pollution_index"] for r in all_results) / len(all_results)
        grade_dist: dict[str, int] = {}
        for r in all_results:
            grade_dist[r["grade"]] = grade_dist.get(r["grade"], 0) + 1
        dist_str = "  ".join(f"{g}:{n}" for g, n in sorted(grade_dist.items()))
        log.info(
            f"  분석 완료: {len(all_results)}개 창문 · "
            f"평균 pollution_index={avg_idx:.4f}  [{dist_str}]"
        )
    else:
        log.info("  분석 완료 (검출된 창문 없음)")
    log.info(f"  결과 저장: {RESULTS_FILE}")
    step_done(t)

    # ── STEP 7: 분석 결과 DB 저장 ─────────────────────────────────────────
    t = step_header(7, "분석 결과 DB 저장 (POST /analysis/result)")
    if not post_analysis_result():
        log.info("  서버 미실행 — STEP 8에서 서버 시작 후 자동 재시도합니다.")
    step_done(t)

    # ── STEP 8: 웹 서버 시작 + 브라우저 열기 ──────────────────────────────
    t = step_header(8, "웹 서버 시작 및 대시보드 열기")
    log.info(f"  $ {' '.join(UVICORN_CMD)}")
    server = subprocess.Popen(UVICORN_CMD, cwd=str(ROOT))

    log.info("  서버 준비 대기 중 (최대 15초)...")
    if wait_for_server(DASHBOARD_URL):
        # STEP 7이 실패했던 경우 재시도
        if RESULTS_FILE.exists():
            post_analysis_result()
        webbrowser.open(DASHBOARD_URL)
        log.info(f"  브라우저 열기: {DASHBOARD_URL}")
    else:
        log.error("  서버 응답 없음 — 수동으로 브라우저를 열어주세요: " + DASHBOARD_URL)
    step_done(t)

    # ── 완료 요약 ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print(f"전체 파이프라인 완료 — 총 소요 시간: {total_elapsed:.1f}초")
    print("=" * 60)

    try:
        log.info("서버 실행 중 (종료: Ctrl+C)")
        server.wait()
    except KeyboardInterrupt:
        log.info("종료 중...")
        server.terminate()
        server.wait()
        log.info("서버 종료 완료.")


if __name__ == "__main__":
    main()
