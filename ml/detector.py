"""
YOLOv8 기반 창문 탐지 모듈.
pretrained 모델로 먼저 테스트하고, 이후 창문 데이터셋으로 fine-tuning 가능.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class DetectedWindow:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    crop: np.ndarray


class WindowDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None

        if YOLO_AVAILABLE:
            model_file = Path(model_path)
            if model_file.exists():
                self.model = YOLO(str(model_file))
            else:
                # weights 없으면 yolov8n 자동 다운로드
                Path("ml/weights").mkdir(parents=True, exist_ok=True)
                self.model = YOLO("yolov8n.pt")
                self.model.save(str(model_file))

    def detect(self, frame: np.ndarray) -> list[DetectedWindow]:
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_fallback(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[DetectedWindow]:
        results = self.model(frame, verbose=False)[0]
        windows = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                windows.append(DetectedWindow(bbox=(x, y, w, h), confidence=conf, crop=crop))

        return windows

    def _detect_fallback(self, frame: np.ndarray) -> list[DetectedWindow]:
        """YOLO 없을 때 OpenCV 기반 단순 사각형 탐지 (테스트용)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        windows = []
        h_frame, w_frame = frame.shape[:2]
        min_area = (w_frame * h_frame) * 0.005  # 전체 화면의 0.5% 이상

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h if h > 0 else 0
                if 0.4 < aspect < 2.5:
                    crop = frame[y:y+h, x:x+w]
                    if crop.size > 0:
                        windows.append(
                            DetectedWindow(bbox=(x, y, w, h), confidence=0.5, crop=crop)
                        )

        return windows
