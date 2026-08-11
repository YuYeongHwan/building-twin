from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["models"])

_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_DIR = _ROOT / "data" / "odm" / "project"
_TEXTURING_DIR = _PROJECT_DIR / "odm_texturing"
_ODM_MOUNT_ROOT = _ROOT / "data" / "odm"
# GPS 좌표(EXIF)가 없는 데이터는 ODM이 "_geo" 접미사 파일만 생성한다.
_TEXTURED_MESH_CANDIDATES = [
    _TEXTURING_DIR / "odm_textured_model.obj",
    _TEXTURING_DIR / "odm_textured_model_geo.obj",
]
_RAW_MESH = _PROJECT_DIR / "odm_meshing" / "odm_mesh.ply"


def _find_mesh() -> Optional[Path]:
    for candidate in _TEXTURED_MESH_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _no_mesh_error() -> HTTPException:
    if _RAW_MESH.exists():
        return HTTPException(
            status_code=404,
            detail="텍스처링된 .obj가 없습니다 (odm_meshing/odm_mesh.ply 만 존재). ODM 텍스처링 단계를 확인하세요.",
        )
    return HTTPException(
        status_code=404,
        detail="Mesh 없음. pipeline/odm_runner.py 를 먼저 실행하세요.",
    )


@router.get("/models/mesh/info")
def mesh_info():
    """OBJ/MTL을 fetch()로 텍스트 로딩할 수 있도록 URL 세트 반환.

    - obj_url / mtl_url: 브라우저에서 fetch(...).then(r => r.text())로 받을 URL
      (반드시 text/plain으로 응답 — StaticFiles가 .obj를 application/x-tgif로
      잘못 추론해 OBJLoader가 파싱에 실패하는 문제를 피하기 위함)
    - texture_base: MTLLoader.parse(text, texture_base)에 넘길 텍스처 디렉터리
      (텍스처 PNG는 /odm 정적 마운트에서 그대로 서빙됨)
    """
    mesh_path = _find_mesh()
    if mesh_path is None:
        raise _no_mesh_error()
    rel_dir = mesh_path.parent.relative_to(_ODM_MOUNT_ROOT)
    return {
        "obj_url": "/models/mesh",
        "mtl_url": "/models/mesh/mtl",
        "texture_base": f"/odm/{rel_dir}/",
    }


@router.get("/models/mesh")
def serve_mesh():
    """ODM 결과물(odm_texturing/odm_textured_model[_geo].obj)을 반환."""
    mesh_path = _find_mesh()
    if mesh_path is None:
        raise _no_mesh_error()
    return FileResponse(
        str(mesh_path),
        media_type="text/plain",
        filename="model.obj",
    )


@router.get("/models/mesh/mtl")
def serve_mtl():
    """OBJ와 짝이 되는 .mtl 파일을 text/plain으로 반환."""
    mesh_path = _find_mesh()
    if mesh_path is None:
        raise _no_mesh_error()
    mtl_path = mesh_path.with_suffix(".mtl")
    if not mtl_path.exists():
        raise HTTPException(status_code=404, detail=".mtl 파일이 없습니다.")
    return FileResponse(
        str(mtl_path),
        media_type="text/plain",
        filename="model.mtl",
    )
