"""
COLMAP SfM 재구성 + 3DGS 학습 모듈.

입력  : data/frames/ 안의 정제된 이미지
출력  :
  - data/sparse/    (COLMAP SfM: cameras.bin / images.bin / points3D.bin)
  - data/splat/     (3DGS: point_cloud.ply + output.splat)

3DGS 학습:
  - CUDA 사용 가능: gaussian-splatting/train.py 실행 (7000 iterations)
  - CUDA 없음 (Apple Silicon 등): COLMAP 포인트 클라우드를 3DGS PLY로
    초기화 후 저장 (학습 없음, 즉각 결과)

실행: python pipeline/reconstruct.py
"""

import logging
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "data" / "frames"
SPARSE_DIR = ROOT / "data" / "sparse"
RECON_DIR  = SPARSE_DIR / "sparse" / "0"
SPLAT_DIR  = ROOT / "data" / "splat"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SH_C0 = 0.28209479177387814

# SIMPLE_PINHOLE=0, PINHOLE=1, SIMPLE_RADIAL=2, RADIAL=3, OPENCV=4 …
_CAM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


# ---------------------------------------------------------------------------
# COLMAP 바이너리 파서
# ---------------------------------------------------------------------------

def _read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid    = struct.unpack("<I", f.read(4))[0]
            mid    = struct.unpack("<i", f.read(4))[0]
            w, h   = struct.unpack("<QQ", f.read(16))
            np_    = _CAM_PARAMS.get(mid, 4)
            params = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cameras[cid] = {"model": mid, "w": int(w), "h": int(h), "params": params}
    return cameras


def _read_images_bin(path: Path) -> dict:
    images = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid  = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))   # w, x, y, z
            tvec = struct.unpack("<3d", f.read(24))
            cid  = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            n2d  = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2d)  # skip 2D points (2*float64 xy + int64 point3D_id)
            images[iid] = {"qvec": qvec, "tvec": tvec, "cam_id": cid, "name": name}
    return images


def _read_points3d_bin(path: Path):
    """Returns (xyz [N,3] float32, rgb [N,3] uint8)."""
    xyzs, rgbs = [], []
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                           # point3D_id (uint64)
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)                           # reprojection error (float64)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)               # track: (image_id uint32, point2D_idx uint32) × N
            xyzs.append(xyz)
            rgbs.append(rgb)
    return np.array(xyzs, dtype=np.float32), np.array(rgbs, dtype=np.uint8)


def _count_images(images_bin: Path) -> int:
    with images_bin.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def _count_points3d(points3d_bin: Path) -> int:
    with points3d_bin.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


# ---------------------------------------------------------------------------
# 3DGS 유틸
# ---------------------------------------------------------------------------

def _init_gaussians_from_colmap(xyz: np.ndarray, rgb: np.ndarray):
    """
    COLMAP 포인트 클라우드 → 3DGS 초기 파라미터

    Returns:
        log_scales    [N, 3] float32
        quats         [N, 4] float32  (w, x, y, z)
        raw_opacities [N]    float32  (logit)
        sh_dc         [N, 3] float32  (SH DC 계수)
    """
    from scipy.spatial import KDTree

    N = len(xyz)

    # Scale: K-최근접 이웃 평균 거리의 절반
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=4)           # 자기 자신(0) + 3개 이웃
    mean_nn = np.maximum(np.mean(dists[:, 1:], axis=1), 1e-6)
    log_scales = np.log(mean_nn[:, None] * 0.5 * np.ones((1, 3), dtype=np.float32))

    # Rotation: identity quaternion (w=1)
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # Opacity: logit(0.1)
    raw_opacities = np.full(N, math.log(0.1 / 0.9), dtype=np.float32)

    # Color: RGB → SH DC
    sh_dc = (rgb.astype(np.float32) / 255.0 - 0.5) / _SH_C0

    return log_scales, quats, raw_opacities, sh_dc


def _export_3dgs_ply(path: Path, xyz, log_scales, quats, raw_opacities, sh_dc) -> None:
    """3DGS 표준 PLY 포맷으로 저장."""
    N = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)

    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(45)]
    props += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"

    normals  = np.zeros((N, 3), dtype=np.float32)
    f_rest   = np.zeros((N, 45), dtype=np.float32)

    with path.open("wb") as f:
        f.write(header.encode())
        for i in range(N):
            f.write(struct.pack("<3f", *xyz[i].tolist()))
            f.write(struct.pack("<3f", *normals[i].tolist()))
            f.write(struct.pack("<3f", *sh_dc[i].tolist()))
            f.write(struct.pack("<45f", *f_rest[i].tolist()))
            f.write(struct.pack("<f",  raw_opacities[i]))
            f.write(struct.pack("<3f", *log_scales[i].tolist()))
            f.write(struct.pack("<4f", *quats[i].tolist()))

    log.info(f"PLY 저장: {path}  ({N:,} Gaussians, {path.stat().st_size / 1024:.0f} KB)")


def _read_3dgs_ply(ply_path: Path) -> dict:
    """3DGS PLY 파일 → 속성 dict."""
    with ply_path.open("rb") as f:
        props, n_vert = [], 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        dtype = np.dtype([(p, np.float32) for p in props])
        raw   = np.frombuffer(f.read(n_vert * dtype.itemsize), dtype=dtype)
    return {p: raw[p].copy() for p in props}


# ---------------------------------------------------------------------------
# COLMAP 실행
# ---------------------------------------------------------------------------

def run_colmap(image_dir: str = str(FRAMES_DIR),
               output_dir: str = str(SPARSE_DIR)) -> bool:
    """
    COLMAP automatic_reconstructor 를 실행한다.

    Returns:
        성공 여부 (True / False)
    """
    img_path  = Path(image_dir)
    work_path = Path(output_dir)

    if not img_path.exists() or not any(img_path.rglob("*.jpg")):
        log.error(f"프레임 디렉터리가 비어 있습니다: {img_path}")
        log.error("먼저 pipeline/preprocess.py 를 실행해 프레임을 추출하세요.")
        return False

    work_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "automatic_reconstructor",
        "--image_path",     str(img_path),
        "--workspace_path", str(work_path),
    ]

    log.info(f"COLMAP 시작: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    images_bin   = work_path / "sparse" / "0" / "images.bin"
    points3d_bin = work_path / "sparse" / "0" / "points3D.bin"

    if result.returncode != 0 or not images_bin.exists():
        log.error("COLMAP 실패: 등록된 이미지가 부족합니다. 촬영을 다시 시도하세요.")
        return False

    num_images = _count_images(images_bin)
    num_points = _count_points3d(points3d_bin) if points3d_bin.exists() else 0

    log.info("COLMAP 재구성 성공!")
    log.info(f"  등록된 이미지 수 : {num_images}")
    log.info(f"  3D 포인트 수     : {num_points}")
    return True


# ---------------------------------------------------------------------------
# 3DGS 학습
# ---------------------------------------------------------------------------

def _prepare_gs_dirs() -> Path:
    """
    gaussian-splatting train.py 가 기대하는 디렉터리 구조를 만든다.

    data/
      images/            ← data/frames 심볼릭 링크
      sparse/0/          ← data/sparse/sparse/0 심볼릭 링크
    """
    gs_data = ROOT / "data"

    # data/images → data/frames
    img_link = gs_data / "images"
    if not img_link.exists():
        img_link.symlink_to(FRAMES_DIR)
        log.info(f"심볼릭 링크 생성: {img_link} → {FRAMES_DIR}")

    # data/sparse/0 → data/sparse/sparse/0
    sparse_0 = SPARSE_DIR / "0"
    sparse_src = SPARSE_DIR / "sparse" / "0"
    if not sparse_0.exists() and sparse_src.exists():
        sparse_0.symlink_to(sparse_src)
        log.info(f"심볼릭 링크 생성: {sparse_0} → {sparse_src}")

    return gs_data


def run_3dgs(data_dir: str = str(ROOT / "data"),
             iterations: int = 7000) -> str:
    """
    3DGS 학습을 실행하고 PLY 파일 경로를 반환한다.

    CUDA 가능:  gaussian-splatting/train.py 실행
    CUDA 없음:  COLMAP 포인트 클라우드를 3DGS PLY 로 초기화 (학습 없이 즉각 저장)

    Returns:
        PLY 파일 절대 경로 (실패 시 빈 문자열)
    """
    import torch
    has_cuda = torch.cuda.is_available()

    SPLAT_DIR.mkdir(parents=True, exist_ok=True)
    ply_path = SPLAT_DIR / "point_cloud.ply"

    # ------------------------------------------------------------------
    # A. CUDA 사용 가능: 공식 gaussian-splatting train.py 실행
    # ------------------------------------------------------------------
    if has_cuda:
        gs_train = ROOT / "gaussian-splatting" / "train.py"
        if not gs_train.exists():
            log.error(f"gaussian-splatting/train.py 를 찾을 수 없습니다: {gs_train}")
            log.error("먼저 gaussian-splatting 레포를 클론하세요.")
            return ""

        gs_data = _prepare_gs_dirs()

        cmd = [
            sys.executable, str(gs_train),
            "-s",             str(gs_data),
            "--iterations",   str(iterations),
            "--model_path",   str(SPLAT_DIR),
        ]
        log.info(f"3DGS 학습 시작 (CUDA): {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(ROOT))

        expected = SPLAT_DIR / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if result.returncode != 0 or not expected.exists():
            log.error("3DGS 학습 실패.")
            return ""

        ply_path = expected
        log.info(f"3DGS 학습 완료: {ply_path}")
        return str(ply_path)

    # ------------------------------------------------------------------
    # B. CUDA 없음(Apple Silicon 등): COLMAP 포인트 클라우드로 초기화
    # ------------------------------------------------------------------
    log.warning("CUDA 미감지 → gaussian-splatting 학습 불가.")
    log.info("COLMAP 포인트 클라우드를 3DGS PLY 로 초기화합니다 (학습 없음).")

    points3d_bin = RECON_DIR / "points3D.bin"
    if not points3d_bin.exists():
        log.error(f"points3D.bin 없음: {points3d_bin}")
        log.error("먼저 run_colmap() 을 실행하세요.")
        return ""

    xyz, rgb = _read_points3d_bin(points3d_bin)
    log.info(f"COLMAP 포인트 로드: {len(xyz):,} 개")

    log_scales, quats, raw_opacities, sh_dc = _init_gaussians_from_colmap(xyz, rgb)
    _export_3dgs_ply(ply_path, xyz, log_scales, quats, raw_opacities, sh_dc)

    log.info(f"3DGS 초기화 완료 (학습 없음): {ply_path}")
    return str(ply_path)


# ---------------------------------------------------------------------------
# PLY → .splat 변환
# ---------------------------------------------------------------------------

def convert_to_splat(ply_path: str, output_path: str) -> None:
    """
    3DGS PLY 파일을 .splat 바이너리로 변환한다.

    .splat 포맷 (32 bytes / Gaussian):
      12 bytes: position  (x, y, z)  float32
      12 bytes: scale     (sx,sy,sz) float32  ← exp(log_scale)
       4 bytes: color     (r,g,b,a)  uint8
       4 bytes: rotation  (w,x,y,z)  int8     ← quaternion × 128
    """
    src = Path(ply_path)
    dst = Path(output_path)

    if not src.exists():
        log.error(f"PLY 없음: {src}")
        return

    log.info(f"PLY → .splat 변환 시작: {src.name}")
    data = _read_3dgs_ply(src)
    N    = len(data["x"])

    # opacity 기준 내림차순 정렬 (렌더러 최초 표시 품질 향상)
    opacities = 1.0 / (1.0 + np.exp(-data["opacity"]))
    order     = np.argsort(-opacities)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        for i in order:
            # Position
            f.write(struct.pack("<3f", data["x"][i], data["y"][i], data["z"][i]))

            # Scale (activate: exp of log)
            f.write(struct.pack("<3f",
                                math.exp(float(data["scale_0"][i])),
                                math.exp(float(data["scale_1"][i])),
                                math.exp(float(data["scale_2"][i]))))

            # Color: SH DC → RGB [0,1] → uint8;  Alpha: sigmoid(opacity) → uint8
            r = int(np.clip(_SH_C0 * data["f_dc_0"][i] + 0.5, 0.0, 1.0) * 255)
            g = int(np.clip(_SH_C0 * data["f_dc_1"][i] + 0.5, 0.0, 1.0) * 255)
            b = int(np.clip(_SH_C0 * data["f_dc_2"][i] + 0.5, 0.0, 1.0) * 255)
            a = int(opacities[i] * 255)
            f.write(struct.pack("<4B", r, g, b, a))

            # Rotation: float quaternion → int8 (× 128, clamped)
            q = np.array([data["rot_0"][i], data["rot_1"][i],
                          data["rot_2"][i], data["rot_3"][i]], dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm > 1e-8:
                q /= norm
            qi = np.clip(q * 128.0, -128, 127).astype(np.int8)
            f.write(qi.tobytes())

    size_kb = dst.stat().st_size / 1024
    log.info(f".splat 저장 완료: {dst}  ({N:,} Gaussians, {size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# 단독 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. COLMAP (이미 완료된 경우 건너뜀)
    images_bin = RECON_DIR / "images.bin"
    if images_bin.exists():
        num_images = _count_images(images_bin)
        num_points = _count_points3d(RECON_DIR / "points3D.bin")
        log.info(f"COLMAP 결과 이미 존재 — 이미지: {num_images}, 포인트: {num_points:,}")
    else:
        ok = run_colmap()
        if not ok:
            sys.exit(1)

    # 2. 3DGS 학습 (또는 초기화)
    ply_path = run_3dgs()
    if not ply_path:
        sys.exit(1)

    # 3. .splat 변환
    splat_out = str(SPLAT_DIR / "output.splat")
    convert_to_splat(ply_path, splat_out)

    # 4. 결과 요약
    print()
    print("=" * 50)
    print("완료 요약")
    print("=" * 50)
    print(f"  추출된 프레임 수        : {len(list(FRAMES_DIR.rglob('*.jpg')))}")
    print(f"  COLMAP 등록 이미지 수   : {_count_images(RECON_DIR / 'images.bin')}")
    print(f"  3DGS PLY                : {ply_path}")
    splat_p = Path(splat_out)
    if splat_p.exists():
        print(f"  .splat 파일             : {splat_out}  ({splat_p.stat().st_size / 1024:.0f} KB)")
    print("=" * 50)
