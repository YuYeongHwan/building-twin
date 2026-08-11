# building-twin

3D Gaussian Splatting 및 SAM 2를 활용한 디지털 트윈 기반  
건물 외벽 오염도 분석 시스템

## 프로젝트 개요

드론 또는 스마트폰으로 촬영한 건물 외벽 영상을 기반으로  
3D 디지털 트윈 모델을 생성하고, AI가 창문을 자동 식별하여  
오염 수치를 정량화하는 통합 관제 웹 플랫폼입니다.

## 주요 기능

- **창문 자동 검출**: YOLO v8 파인튜닝 + Canny 엣지 병합 방식
- **오염도 분석**: HSV 색공간 기반 Pollution Index 산출 (0~1, A~D 등급)
- **3D 모델링**: OpenDroneMap 기반 건물 외벽 3D Mesh 생성
- **웹 대시보드**: 창문별 오염도 등급 시각화 및 필터링
- **3D 뷰어**: Three.js + OBJLoader 기반 웹 렌더링
- **LiDAR 안전 가이드**: 드론 접근 시 5방향 거리 기반 안전 등급 판단
- **허프 변환 알고리즘**: 수평·수직 선분 기반 창문 영역 검출 연구

## 기술 스택

| 구분 | 기술 |
|---|---|
| 창문 검출 | YOLO v8, OpenCV (Canny + Contour) |
| 오염도 분석 | OpenCV HSV, Python |
| 3D 모델링 | OpenDroneMap, Three.js, OBJLoader |
| 백엔드 | FastAPI, MySQL |
| 프론트엔드 | HTML, JavaScript, Three.js |
| 안전 가이드 | LiDAR 시뮬레이션 (Python) |

## 파이프라인

```
영상 입력 (스마트폰/드론)
  ↓
프레임 추출 및 전처리 (OpenCV, CLAHE)
  ↓
3D 모델 생성 (OpenDroneMap → .obj)
  ↓
창문 자동 검출 (YOLO v8 + Canny 병합)
  ↓
오염도 분석 (HSV Pollution Index)
  ↓
웹 대시보드 시각화 (FastAPI + Three.js)
```

## 프로젝트 구조

```
building-twin/
├── pipeline/
│   ├── preprocess.py      # 프레임 추출 및 전처리
│   ├── detect.py          # Canny 기반 창문 검출
│   ├── analyze.py         # HSV 오염도 분석
│   ├── reconstruct.py     # COLMAP + 3DGS 파이프라인
│   ├── odm_runner.py      # OpenDroneMap 실행
│   ├── hough_detect.py    # 허프 변환 창문 검출
│   ├── lidar_safety.py    # LiDAR 안전 접근 가이드
│   ├── watcher.py         # 영상 파일 감시 자동 실행
│   └── run_all.py         # 전체 파이프라인 실행
├── ml/
│   └── pipeline.py        # YOLO v8 기반 창문 검출 파이프라인
├── app/
│   └── api/routes/        # FastAPI 라우터
│       ├── analysis_v2.py
│       ├── mesh.py
│       ├── splat.py
│       └── lidar.py
├── backend/
│   └── main.py            # FastAPI 앱 진입점
├── templates/
│   ├── dashboard.html     # 오염도 대시보드
│   └── viewer.html        # 3D 뷰어
├── data/
│   ├── raw_video/         # 입력 영상
│   ├── frames/             # 추출된 프레임
│   ├── results/            # 분석 결과 이미지
│   ├── splat/               # 3DGS 결과물
│   ├── odm/                 # ODM 3D Mesh 결과물
│   └── yolo_runs/          # YOLO 학습 결과
│       ├── window_detector_v2/   # 최종 채택 모델
│       └── window_detector_v3/   # 실험 모델
└── requirements.txt
```

## 실행 방법

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 영상 파일 넣기
`data/raw_video/` 폴더에 영상 파일(.mp4, .mov) 넣기

### 3. 전체 파이프라인 실행
```bash
python pipeline/run_all.py
```

### 4. 웹 대시보드 접속
```bash
python -m uvicorn backend.main:app --reload
```
- 대시보드: http://localhost:8000/dashboard
- 3D 뷰어: http://localhost:8000/viewer
- LiDAR 상태: http://localhost:8000/lidar/status

### 5. LiDAR 안전 가이드 단독 실행
```bash
python pipeline/lidar_safety.py
```

## 오염도 등급 기준

| 등급 | 오염 지수 | 상태 |
|---|---|---|
| A | 0.0 ~ 0.1 | 청결 |
| B | 0.1 ~ 0.3 | 양호 |
| C | 0.3 ~ 0.6 | 주의 |
| D | 0.6 ~ 1.0 | 청소 필요 |

## YOLO 모델 학습 과정

| 버전 | 데이터 | 결과 |
|---|---|---|
| v1 | 공개 데이터셋 387장 | 도메인 차이로 검출 실패 |
| v2 | 실제 건물 70장 자동 라벨링 | 37개 검출 / 신뢰도 0.48~0.60 ✅ |
| v3 | v2 + 수동 라벨 10장 | 라벨 밀도 불균형으로 채택 제외 |

## LiDAR 안전 등급 기준

| 등급 | 거리 기준 | 조치 |
|---|---|---|
| 🟢 SAFE | 150cm 이상 | 정상 접근 가능 |
| 🟡 WARNING | 80~150cm | 속도 감소 후 서행 |
| 🔴 DANGER | 30~80cm | 즉시 후퇴 |
| 🚨 CRITICAL | 30cm 미만 | 긴급 정지 |

## 팀 구성

| 역할 | 담당 업무 |
|---|---|
| AI & Data | YOLO v8 학습, 오염도 분석 알고리즘 |
| 3D & Engine | OpenDroneMap, 3DGS 파이프라인 |
| Full-stack | FastAPI 백엔드, 웹 대시보드, 3D 뷰어 |

## GitHub

https://github.com/YuYeongHwan/building-twin
