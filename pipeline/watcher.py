"""
data/raw_video/ 폴더를 감시하다가 새 .mp4 / .mov 파일이 감지되면
전체 파이프라인을 자동 실행합니다.

실행: python pipeline/watcher.py
"""

import os
import sys
import time
import logging
from pathlib import Path

# project root 와 project/ 를 sys.path 에 등록해 pipeline 모듈을 임포트할 수 있게 함
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pipeline.preprocess import extract_frames
from pipeline.reconstruct import run_colmap, run_gaussian_splatting
from pipeline.detect import detect_windows
from pipeline.analyze import analyze_window

WATCH_DIR   = ROOT / "data" / "raw_video"
FRAMES_DIR  = ROOT / "data" / "frames"
SPARSE_DIR  = ROOT / "data" / "sparse"
SPLAT_DIR   = ROOT / "data" / "splat"
STABLE_SECS = 3          # 파일 크기가 이 초 동안 변하지 않으면 복사 완료로 판단

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_processing: set[str] = set()


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def wait_until_stable(path: Path, interval: float = STABLE_SECS) -> bool:
    """파일 크기가 interval 초 동안 변하지 않으면 True 반환."""
    prev_size = -1
    while True:
        try:
            curr_size = path.stat().st_size
        except FileNotFoundError:
            return False
        if curr_size == prev_size:
            return True
        prev_size = curr_size
        time.sleep(interval)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def run_pipeline(video_path: Path) -> None:
    key = str(video_path)
    if key in _processing:
        return
    _processing.add(key)

    try:
        log.info("=" * 60)
        log.info(f"파이프라인 시작: {video_path.name}")
        log.info("=" * 60)

        # ── 1단계: 전처리 ──────────────────────────────────────────
        log.info("[1/4] 전처리 시작  (프레임 추출 · OpenCV)")
        frames_out = str(FRAMES_DIR / video_path.stem)
        try:
            frames = extract_frames(str(video_path), frames_out)
            log.info(f"[1/4] 전처리 완료  — {len(frames)}개 프레임 추출 → {frames_out}")
        except Exception as exc:
            log.error(f"[1/4] 전처리 실패: {exc}")
            return

        # ── 2단계: 3D 재구성 ───────────────────────────────────────
        log.info("[2/4] 3D 재구성 시작  (COLMAP → 3D Gaussian Splatting)")
        sparse_out = str(SPARSE_DIR / video_path.stem)
        splat_out  = str(SPLAT_DIR  / video_path.stem)
        try:
            sparse_dir = run_colmap(frames_out, sparse_out)
            run_gaussian_splatting(sparse_dir, splat_out)
            log.info(f"[2/4] 3D 재구성 완료  — splat 저장 경로: {splat_out}")
        except Exception as exc:
            log.error(f"[2/4] 3D 재구성 실패: {exc}")
            return

        # ── 3단계: 창문 검출 ───────────────────────────────────────
        log.info("[3/4] 창문 검출 시작  (YOLO v8 → SAM 2)")
        detections: list[tuple[str, object, list]] = []
        try:
            for fp in frames:
                frame = cv2.imread(fp)
                if frame is None:
                    continue
                boxes = detect_windows(frame)
                if boxes:
                    detections.append((fp, frame, boxes))
            total_windows = sum(len(b) for _, _, b in detections)
            log.info(f"[3/4] 창문 검출 완료  — 총 {total_windows}개 창문 검출 "
                     f"({len(detections)}/{len(frames)} 프레임에서 발견)")
        except Exception as exc:
            log.error(f"[3/4] 창문 검출 실패: {exc}")
            return

        # ── 4단계: 오염도 분석 ─────────────────────────────────────
        log.info("[4/4] 오염도 분석 시작  (HSV 기반 Pollution Index)")
        try:
            results: list[tuple[str, float]] = []
            grade_count: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}

            for fp, frame, boxes in detections:
                for x, y, w, h in boxes:
                    crop = frame[y: y + h, x: x + w]
                    if crop.size == 0:
                        continue
                    grade, pct, _ = analyze_window(crop)
                    results.append((grade, pct))
                    grade_count[grade] = grade_count.get(grade, 0) + 1

            if results:
                avg_pct = sum(p for _, p in results) / len(results)
                grade_summary = "  ".join(
                    f"{g}:{cnt}" for g, cnt in sorted(grade_count.items()) if cnt
                )
                log.info(
                    f"[4/4] 오염도 분석 완료  — "
                    f"{len(results)}개 창문 · 평균 오염도 {avg_pct:.1f}%  "
                    f"[등급 분포: {grade_summary}]"
                )
            else:
                log.info("[4/4] 오염도 분석 완료  — 분석 가능한 창문 없음")
        except Exception as exc:
            log.error(f"[4/4] 오염도 분석 실패: {exc}")
            return

        log.info("=" * 60)
        log.info(f"파이프라인 완료: {video_path.name}")
        log.info("=" * 60)

    finally:
        _processing.discard(key)


# ---------------------------------------------------------------------------
# Watchdog 핸들러
# ---------------------------------------------------------------------------

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in {".mp4", ".mov"}:
            return

        log.info(f"새 파일 감지: {path.name}  — 복사 완료 대기 중 ({STABLE_SECS}초 안정화) ...")
        if wait_until_stable(path):
            run_pipeline(path)
        else:
            log.warning(f"파일을 찾을 수 없음 (전송 중 삭제된 것 같음): {path.name}")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    log.info("building-twin 파이프라인 와처 시작")
    log.info(f"감시 디렉터리: {WATCH_DIR}")
    log.info(".mp4 / .mov 파일을 data/raw_video/ 에 넣으면 파이프라인이 자동 실행됩니다.")
    log.info("종료: Ctrl+C")
    log.info("-" * 60)

    handler  = VideoHandler()
    observer = Observer()
    observer.schedule(handler, str(WATCH_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("종료 신호 수신 ...")
        observer.stop()

    observer.join()
    log.info("와처 종료 완료.")


if __name__ == "__main__":
    main()
