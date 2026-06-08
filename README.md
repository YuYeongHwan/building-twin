# building-twin

3D Gaussian Splatting 및 SAM 2를 활용한 디지털 트윈 기반 건물 외벽 오염도 분석 시스템

## 프로젝트 개요
드론/스마트폰 카메라로 촬영한 건물 외벽 영상을 기반으로 3D 디지털 트윈 모델을 생성하고,
AI가 유리창 영역을 자동 식별하여 오염 수치를 정량화하는 통합 관제 웹 플랫폼

## 기술 스택
- 3D Reconstruction: 3D Gaussian Splatting, COLMAP
- Object Detection: YOLO v8, SAM 2
- Image Processing: OpenCV
- Backend: FastAPI, MySQL
- Frontend: HTML, JavaScript, gsplat.js

## 파이프라인
1. 영상 입력 (스마트폰/드론)
2. 프레임 추출 및 전처리 (OpenCV)
3. 3D 모델 생성 (COLMAP → 3DGS)
4. 창문 검출 (YOLO v8 → SAM 2)
5. 오염도 분석 (HSV 기반 Pollution Index)
6. 웹 시각화 (3D 뷰어 + 히트맵 대시보드)

## 실행 방법
1. 의존성 설치
```
pip install -r requirements.txt
```

2. 영상 파일을 data/raw_video/ 에 넣기

3. 전체 파이프라인 실행
```
python pipeline/run_all.py
```

4. 웹 대시보드 접속
```
http://localhost:8000/dashboard
```

## 팀 구성
- AI & Data: YOLO, SAM 2 연동 및 오염도 분석 알고리즘
- 3D & Engine: COLMAP, 3DGS 파이프라인 및 모델 최적화
- Full-stack: FastAPI 백엔드, 웹 대시보드 및 3D 시각화
