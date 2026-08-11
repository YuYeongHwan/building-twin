"""
OpenDroneMap(ODM) Docker 컨테이너를 실행해 드론/스마트폰 사진을 3D Mesh로 재구성한다.

입력 : data/odm/project/images/ 안의 사진
출력 : data/odm/project/odm_texturing/odm_textured_model.obj  (텍스처 Mesh)
       data/odm/project/odm_meshing/odm_mesh.ply              (원본 Mesh, 텍스처 없음)
       data/odm/project/odm_georeferencing/                   (좌표 정보)

ODM 공식 Docker 사용법(https://docs.opendronemap.org/installation.html):
  프로젝트 폴더 구조가 <project-path>/<project-name>/images/ 여야 하므로
  data/odm 를 /datasets 에 마운트하고 프로젝트 이름을 "project" 로 지정한다.

실행: python pipeline/odm_runner.py
"""

import os
import subprocess
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
ODM_DIR     = ROOT / "data" / "odm"
PROJECT_DIR = ODM_DIR / "project"
IMAGES_DIR  = PROJECT_DIR / "images"
# GPS 좌표(EXIF)가 없는 데이터는 ODM이 "_geo" 접미사 파일만 생성한다.
MESH_PATH_CANDIDATES = [
    PROJECT_DIR / "odm_texturing" / "odm_textured_model.obj",
    PROJECT_DIR / "odm_texturing" / "odm_textured_model_geo.obj",
]
RAW_MESH_PATH = PROJECT_DIR / "odm_meshing" / "odm_mesh.ply"


def run_odm() -> None:
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("*")):
        print(f"이미지가 없습니다: {IMAGES_DIR}")
        print("드론/스마트폰 사진을 data/odm/project/images/ 에 넣은 뒤 다시 실행하세요.")
        return

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{ODM_DIR}:/dataset",
        "opendronemap/odm",
        "--project-path", "/dataset",
        "project",
        "--feature-quality", "ultra",
        "--pc-quality", "ultra",
        "--mesh-size", "300000",
        "--ignore-gsd",
        "--auto-boundary",
        "--pc-filter", "5",        # 배경 잡음 제거 강도
        "--dem-resolution", "1",
        "--orthophoto-resolution", "1",
        "--min-num-features", "16000",
    ]

    print("===== ODM 고품질 재실행 =====")
    print(f"이미지 수: {len(list(IMAGES_DIR.glob('*')))}장")
    print("예상 소요 시간: 30분~1시간")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print("===== 완료 =====")

    if result.returncode != 0:
        print(f"ODM 실행 실패 (exit code {result.returncode})")
        return

    mesh_path = next((p for p in MESH_PATH_CANDIDATES if p.exists()), None)
    if mesh_path is not None:
        size_mb = mesh_path.stat().st_size / 1024 / 1024
        print(f"Mesh 생성 성공: {mesh_path} ({size_mb:.1f}MB)")
    elif RAW_MESH_PATH.exists():
        size_mb = RAW_MESH_PATH.stat().st_size / 1024 / 1024
        print(f"텍스처링 없는 원본 Mesh만 생성됨: {RAW_MESH_PATH} ({size_mb:.1f}MB)")
    else:
        print("Mesh 생성 실패: 결과물 없음")


if __name__ == "__main__":
    run_odm()
