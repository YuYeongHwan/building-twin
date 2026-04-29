"""
영상 처리 파이프라인: 비디오 → 프레임 샘플링 → 창문 탐지 → 오염도 분석 → DB 저장
"""
import cv2
import os
from pathlib import Path
from sqlalchemy.orm import Session

from ml.detector import WindowDetector
from ml.grader import ContaminationGrader
from app.models.inspection import Inspection, InspectionStatus
from app.models.window import WindowResult, ContaminationGrade
from app.core.config import settings


class InspectionPipeline:
    def __init__(self):
        self.detector = WindowDetector(
            model_path=settings.YOLO_MODEL_PATH,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        )
        self.grader = ContaminationGrader()

    def process(self, inspection: Inspection, video_path: str, db: Session) -> None:
        inspection.status = InspectionStatus.PROCESSING
        db.commit()

        try:
            self._run(inspection, video_path, db)
            inspection.status = InspectionStatus.COMPLETED
        except Exception as e:
            inspection.status = InspectionStatus.FAILED
            raise e
        finally:
            db.commit()

    def _run(self, inspection: Inspection, video_path: str, db: Session) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        inspection.total_frames = total_frames
        db.commit()

        results_dir = Path("results") / str(inspection.id)
        results_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        processed = 0
        window_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % settings.FRAME_SAMPLE_RATE == 0:
                windows = self.detector.detect(frame)

                for w_idx, win in enumerate(windows):
                    grade_result = self.grader.analyze(win.crop)
                    x, y, w, h = win.bbox

                    crop_path = results_dir / f"f{frame_idx}_w{w_idx}.jpg"
                    cv2.imwrite(str(crop_path), win.crop)

                    window_result = WindowResult(
                        inspection_id=inspection.id,
                        frame_number=frame_idx,
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        contamination_score=grade_result.score,
                        grade=ContaminationGrade(grade_result.grade),
                        confidence=win.confidence,
                        crop_image_path=str(crop_path),
                    )
                    db.add(window_result)
                    window_count += 1

                processed += 1
                inspection.processed_frames = processed
                inspection.total_windows = window_count
                db.commit()

            frame_idx += 1

        cap.release()
