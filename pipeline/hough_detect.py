"""
허프 변환 기반 창문 검출 모듈.

허프 변환(HoughLinesP)으로 창문의 수직/수평 선분을 검출하고,
인접한 수평선-수직선 교차점으로 창문 영역을 사각형으로 특정한다.

실행: python pipeline/hough_detect.py
"""

import cv2
import numpy as np
import os


def detect_windows_hough(image_path: str) -> list:
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 전처리
    # 하늘/바닥 영역 제거
    roi = gray[int(h*0.12):int(h*0.90), :]
    roi_h, roi_w = roi.shape

    # CLAHE 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi)

    # 가우시안 블러로 노이즈 제거 (화강암 미세 질감을 더 부드럽게)
    blurred = cv2.GaussianBlur(roi_eq, (7,7), 0)

    # 2. Canny 엣지 검출 (임계값 상향 — 약한 화강암 텍스처 엣지 제거)
    edges = cv2.Canny(blurred, 60, 150)

    # 3. 허프 변환으로 직선 검출
    # HoughLinesP: 확률적 허프 변환 (더 빠르고 정확)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,              # 거리 해상도 (픽셀)
        theta=np.pi/180,    # 각도 해상도 (라디안)
        threshold=50,       # 최소 투표 수
        minLineLength=60,   # 최소 선분 길이 (짧은 화강암 질감 선분 제거)
        maxLineGap=10       # 선분 간 최대 간격
    )

    if lines is None:
        return []

    # 4. 수평선/수직선 분리
    # 화강암 질감 선분은 짧고 산발적, 실제 창문 수직 프레임은 길고 연속적이므로
    # 수직선은 각도(거의 완전한 수직)와 길이 조건을 모두 만족해야 한다.
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))

        # 수평선: 각도 -15 ~ 15도
        if -15 <= angle <= 15:
            horizontal_lines.append(line[0])
        # 수직선: 각도 82~90 또는 -90~-82도 + 길이 80px 이상
        elif abs(angle) >= 82:
            if length >= 80:
                vertical_lines.append(line[0])

    # 5. 수평선/수직선 교차점으로 창문 후보 사각형 생성
    windows = []
    win_idx = 1

    # 수평선 y좌표 클러스터링
    h_ys = sorted(set([
        int((l[1]+l[3])/2) for l in horizontal_lines
    ]))
    # 수직선 x좌표 클러스터링
    v_xs = sorted(set([
        int((l[0]+l[2])/2) for l in vertical_lines
    ]))

    # 인접한 좌표 병합 (30픽셀 이내)
    def cluster_coords(coords, gap=30):
        if not coords: return []
        clusters = [[coords[0]]]
        for c in coords[1:]:
            if c - clusters[-1][-1] < gap:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [int(np.mean(cl)) for cl in clusters]

    h_ys = cluster_coords(h_ys, gap=30)
    v_xs = cluster_coords(v_xs, gap=50)   # 인접한 화강암 질감 선분들을 하나로 묶음

    # 인접한 수평선 쌍 + 수직선 쌍으로 사각형 생성
    for i in range(len(h_ys)-1):
        for j in range(len(v_xs)-1):
            y1 = h_ys[i]
            y2 = h_ys[i+1]
            x1 = v_xs[j]
            x2 = v_xs[j+1]

            bw = x2 - x1
            bh = y2 - y1

            # 크기 필터
            if bw < 40 or bh < 40:
                continue
            if bw > roi_w * 0.4 or bh > roi_h * 0.4:
                continue

            # 비율 필터 (0.6~2.0)
            aspect = bw / bh
            if aspect < 0.6 or aspect > 2.0:
                continue

            # 실제 이미지 좌표로 변환
            y1_real = y1 + int(h*0.12)
            y2_real = y2 + int(h*0.12)

            # 크롭 이미지 저장
            crop = img[y1_real:y2_real, x1:x2]
            if crop.size == 0:
                continue

            window_id = f"window_hough_{win_idx:03d}"
            os.makedirs("data/results", exist_ok=True)
            crop_path = f"data/results/{window_id}_crop.jpg"
            cv2.imwrite(crop_path, crop)

            windows.append({
                "window_id": window_id,
                "bbox": [x1, y1_real, x2, y2_real],
                "crop_image_path": crop_path
            })
            win_idx += 1

    return windows


def visualize_hough(image_path: str, output_path: str):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roi = gray[int(h*0.12):int(h*0.90), :]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi)
    blurred = cv2.GaussianBlur(roi_eq, (7,7), 0)
    edges = cv2.Canny(blurred, 60, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 50,
        minLineLength=60, maxLineGap=10
    )

    result = img.copy()
    offset_y = int(h*0.12)

    n_horizontal = 0
    n_vertical = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))

            # 수평선 → 파란색
            if -15 <= angle <= 15:
                n_horizontal += 1
                cv2.line(result,
                    (x1, y1+offset_y),
                    (x2, y2+offset_y),
                    (255, 0, 0), 2)
            # 수직선 → 빨간색 (거의 완전한 수직 + 길이 80px 이상만)
            elif abs(angle) >= 82 and length >= 80:
                n_vertical += 1
                cv2.line(result,
                    (x1, y1+offset_y),
                    (x2, y2+offset_y),
                    (0, 0, 255), 2)

    # 검출된 창문 → 초록색
    windows = detect_windows_hough(image_path)
    for w_info in windows:
        x1, y1, x2, y2 = w_info['bbox']
        cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(result, w_info['window_id'],
            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0,255,0), 1)

    # 범례
    cv2.putText(result, "Blue: Horizontal lines",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(result, "Red: Vertical lines",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(result, "Green: Detected Windows",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imwrite(output_path, result)
    print(f"저장: {output_path}")
    print(f"수평선: {n_horizontal}개")
    print(f"수직선: {n_vertical}개")
    print(f"검출 창문: {len(windows)}개")
    return windows


if __name__ == "__main__":
    import sys
    from pathlib import Path

    frames = sorted(Path("data/frames").rglob("*.jpg"))
    if not frames:
        print("data/frames/ 에 이미지 없음")
        sys.exit(1)

    # 5번째 프레임으로 테스트
    test_path = str(frames[min(4, len(frames)-1)])

    print(f"테스트 이미지: {test_path}")
    windows = visualize_hough(
        test_path,
        "data/results/hough_result.jpg"
    )

    print(f"\n검출된 창문 목록:")
    for w in windows:
        print(f"  {w['window_id']}: {w['bbox']}")
