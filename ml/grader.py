"""
창문 오염도 등급 산정 모듈.

오염도 점수(0.0~1.0) → 등급 변환:
  A (청결):  0.00 ~ 0.25
  B (경미):  0.25 ~ 0.50
  C (보통):  0.50 ~ 0.75
  D (심각):  0.75 ~ 1.00

오염도 점수 계산 방법:
  1. 밝기 균일도: 오염된 창문은 얼룩으로 인해 밝기 분포가 불균일
  2. 채도 분석: 오염물(먼지, 얼룩)은 채도가 낮은 회색/갈색 계열
  3. 엣지 밀도: 오염 패턴은 불규칙한 텍스처를 유발
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class GradeResult:
    score: float        # 0.0 ~ 1.0 (높을수록 오염)
    grade: str          # A / B / C / D
    brightness_score: float
    texture_score: float
    color_score: float


GRADE_THRESHOLDS = [
    (0.25, "A"),
    (0.50, "B"),
    (0.75, "C"),
    (1.01, "D"),
]


def score_to_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score < threshold:
            return grade
    return "D"


class ContaminationGrader:
    def analyze(self, crop: np.ndarray) -> GradeResult:
        if crop is None or crop.size == 0:
            return GradeResult(score=0.0, grade="A",
                               brightness_score=0.0, texture_score=0.0, color_score=0.0)

        crop_resized = cv2.resize(crop, (64, 64))
        brightness_score = self._brightness_nonuniformity(crop_resized)
        texture_score = self._texture_complexity(crop_resized)
        color_score = self._color_contamination(crop_resized)

        # 가중 평균
        total_score = (
            brightness_score * 0.40
            + texture_score * 0.35
            + color_score * 0.25
        )
        total_score = float(np.clip(total_score, 0.0, 1.0))

        return GradeResult(
            score=total_score,
            grade=score_to_grade(total_score),
            brightness_score=brightness_score,
            texture_score=texture_score,
            color_score=color_score,
        )

    def _brightness_nonuniformity(self, img: np.ndarray) -> float:
        """밝기 표준편차를 정규화한 점수. 불균일할수록 높음."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        std = np.std(gray)
        # 표준편차 0~80 범위를 0~1로 정규화
        return float(np.clip(std / 80.0, 0.0, 1.0))

    def _texture_complexity(self, img: np.ndarray) -> float:
        """Laplacian 분산 기반 텍스처 복잡도. 오염 얼룩은 텍스처를 증가시킴."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 분산 0~500 범위를 0~1로 정규화
        return float(np.clip(lap_var / 500.0, 0.0, 1.0))

    def _color_contamination(self, img: np.ndarray) -> float:
        """
        오염된 창문은 탁한 갈색/회색을 띰.
        HSV 채도(S) 평균이 낮고 명도(V) 평균도 낮은 경우 오염으로 판단.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
        s_mean = np.mean(hsv[:, :, 1]) / 255.0  # 0~1
        v_mean = np.mean(hsv[:, :, 2]) / 255.0  # 0~1

        # 채도 낮고 명도 낮으면 오염 가능성 높음
        contamination = (1.0 - s_mean) * 0.5 + (1.0 - v_mean) * 0.5
        return float(np.clip(contamination, 0.0, 1.0))
