"""
LiDAR 센서 데이터 기반 드론 안전 접근 가이드.

실제 LiDAR 센서가 없는 경우 시뮬레이션 모드로 동작하며, 드론이 건물
외벽에 접근할 때 전/좌/우/상/하 거리를 바탕으로 안전 등급(SAFE/WARNING/
DANGER/CRITICAL)과 권장 조치를 판단한다.

실행: python pipeline/lidar_safety.py
"""

import numpy as np
import json
import time
import os
from datetime import datetime

# 안전 거리 기준 (단위: cm)
SAFE_DISTANCE = 150      # 150cm 이상: 안전
WARNING_DISTANCE = 80    # 80~150cm: 주의
DANGER_DISTANCE = 80     # 80cm 미만: 위험 (즉시 후퇴)
CRITICAL_DISTANCE = 30   # 30cm 미만: 긴급 정지

class LidarSafetyGuide:
    """
    LiDAR 센서 데이터 기반 드론 안전 접근 가이드 시스템
    실제 LiDAR 없는 경우 시뮬레이션 모드로 동작
    """

    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.history = []
        self.current_distance = None
        self.alert_log = []

    def read_lidar(self, lidar_device=None) -> dict:
        """
        LiDAR 센서에서 거리 데이터 읽기
        simulation_mode=True 이면 시뮬레이션 데이터 반환
        """
        if self.simulation_mode:
            return self._simulate_lidar()
        else:
            # 실제 LiDAR 연동 코드 (하드웨어 파트와 연동)
            # 예: RPLiDAR, TFmini, Benewake 등
            return self._read_real_lidar(lidar_device)

    def _simulate_lidar(self) -> dict:
        """
        시뮬레이션: 드론이 건물에 접근하는 시나리오
        실제 구현 시 이 함수를 실제 센서 읽기로 교체
        """
        # 5개 방향 거리 측정 시뮬레이션 (단위: cm)
        # 전방, 좌, 우, 상, 하
        base_dist = np.random.uniform(50, 300)
        data = {
            'timestamp': datetime.now().isoformat(),
            'front': round(base_dist + np.random.uniform(-10, 10), 1),
            'left':  round(base_dist + np.random.uniform(-20, 20) + 30, 1),
            'right': round(base_dist + np.random.uniform(-20, 20) + 30, 1),
            'up':    round(np.random.uniform(100, 500), 1),
            'down':  round(np.random.uniform(50, 200), 1),
            'mode':  'simulation'
        }
        return data

    def _read_real_lidar(self, device) -> dict:
        """실제 LiDAR 센서 연동 (하드웨어 파트 구현 필요)"""
        # TODO: 실제 센서 드라이버 연동
        # 예시:
        # import serial
        # ser = serial.Serial('/dev/ttyUSB0', 115200)
        # raw = ser.read(9)
        # distance = (raw[3] + raw[4] * 256) / 100
        raise NotImplementedError("실제 LiDAR 드라이버 연동 필요")

    def analyze_safety(self, lidar_data: dict) -> dict:
        """
        LiDAR 데이터를 분석하여 안전 상태 판단
        """
        front = lidar_data.get('front', 999)
        left  = lidar_data.get('left', 999)
        right = lidar_data.get('right', 999)

        # 전방 최소 거리 기준으로 판단
        min_distance = min(front, left, right)
        self.current_distance = min_distance

        # 안전 등급 판단
        if min_distance >= SAFE_DISTANCE:
            status = 'SAFE'
            status_kr = '안전'
            color = 'green'
            action = '정상 접근 가능'
        elif min_distance >= WARNING_DISTANCE:
            status = 'WARNING'
            status_kr = '주의'
            color = 'yellow'
            action = '속도 감소 후 서행 접근'
        elif min_distance >= CRITICAL_DISTANCE:
            status = 'DANGER'
            status_kr = '위험'
            color = 'red'
            action = '즉시 후퇴'
        else:
            status = 'CRITICAL'
            status_kr = '긴급'
            color = 'red'
            action = '긴급 정지 및 후퇴'

        result = {
            'timestamp': lidar_data.get('timestamp'),
            'distances': {
                'front': front,
                'left':  left,
                'right': right,
                'up':    lidar_data.get('up', 999),
                'down':  lidar_data.get('down', 999),
                'minimum': round(min_distance, 1)
            },
            'safety': {
                'status': status,
                'status_kr': status_kr,
                'color': color,
                'action': action
            },
            'thresholds': {
                'safe':     SAFE_DISTANCE,
                'warning':  WARNING_DISTANCE,
                'danger':   DANGER_DISTANCE,
                'critical': CRITICAL_DISTANCE
            }
        }

        # 이력 저장
        self.history.append(result)

        # 위험/긴급 상황 로그
        if status in ['DANGER', 'CRITICAL']:
            self.alert_log.append({
                'time': result['timestamp'],
                'status': status,
                'distance': min_distance,
                'action': action
            })

        return result

    def get_approach_guide(self, lidar_data: dict) -> str:
        """
        조종사에게 텍스트로 접근 가이드 제공
        """
        result = self.analyze_safety(lidar_data)
        dist = result['distances']['minimum']
        status = result['safety']['status']
        action = result['safety']['action']

        guide = f"""
=== 안전 미세 접근 가이드 (Eye 역할) ===
시각: {result['timestamp']}
────────────────────────────
전방 거리: {result['distances']['front']} cm
좌측 거리: {result['distances']['left']} cm
우측 거리: {result['distances']['right']} cm
최소 거리: {dist} cm
────────────────────────────
상태: [{status}] {result['safety']['status_kr']}
조치: {action}
────────────────────────────
안전 기준:
  🟢 안전:   {SAFE_DISTANCE}cm 이상
  🟡 주의:   {WARNING_DISTANCE}~{SAFE_DISTANCE}cm
  🔴 위험:   {CRITICAL_DISTANCE}~{WARNING_DISTANCE}cm
  🚨 긴급:   {CRITICAL_DISTANCE}cm 미만
"""
        return guide

    def save_log(self, output_path='data/lidar_log.json'):
        """분석 결과 JSON으로 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_readings': len(self.history),
                'alert_count': len(self.alert_log),
                'alerts': self.alert_log,
                'history': self.history[-50:]  # 최근 50개
            }, f, ensure_ascii=False, indent=2)
        print(f"로그 저장: {output_path}")


def run_simulation(n=20):
    """
    드론 접근 시나리오 시뮬레이션 실행
    300cm → 점점 가까워지는 시나리오
    """
    guide = LidarSafetyGuide(simulation_mode=True)

    print("=== LiDAR 안전 접근 시뮬레이션 시작 ===")
    print(f"총 {n}회 측정\n")

    # 점점 가까워지는 시나리오 (300cm → 20cm)
    distances = np.linspace(300, 20, n)

    for i, target_dist in enumerate(distances):
        # 실제 센서처럼 약간의 노이즈 추가
        noise = np.random.uniform(-5, 5)
        dist = max(10, target_dist + noise)

        lidar_data = {
            'timestamp': datetime.now().isoformat(),
            'front': round(dist, 1),
            'left':  round(dist + np.random.uniform(10, 30), 1),
            'right': round(dist + np.random.uniform(10, 30), 1),
            'up':    round(np.random.uniform(100, 300), 1),
            'down':  round(np.random.uniform(50, 150), 1),
            'mode':  'simulation'
        }

        result = guide.analyze_safety(lidar_data)
        dist_val = result['distances']['minimum']
        status = result['safety']['status']
        action = result['safety']['action']

        # 상태별 이모지
        emoji = {'SAFE':'🟢','WARNING':'🟡','DANGER':'🔴','CRITICAL':'🚨'}
        print(f"[{i+1:02d}] {emoji.get(status,'?')} "
              f"거리: {dist_val:6.1f}cm | "
              f"{status:8s} | {action}")

        time.sleep(0.1)

    print("\n=== 시뮬레이션 완료 ===")
    print(f"총 측정: {len(guide.history)}회")
    print(f"경고 발생: {len(guide.alert_log)}회")

    guide.save_log()
    return guide


if __name__ == "__main__":
    guide = run_simulation(n=20)
    print(guide.get_approach_guide(guide.read_lidar()))
