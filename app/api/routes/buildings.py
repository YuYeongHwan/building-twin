from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.building import Building
from app.schemas.building import BuildingCreate, BuildingResponse

router = APIRouter(prefix="/api/buildings", tags=["buildings"])


@router.get("/", response_model=list[BuildingResponse])
def list_buildings(db: Session = Depends(get_db)):
    return db.query(Building).order_by(Building.created_at.desc()).all()


@router.post("/", response_model=BuildingResponse, status_code=201)
def create_building(body: BuildingCreate, db: Session = Depends(get_db)):
    building = Building(**body.model_dump())
    db.add(building)
    db.commit()
    db.refresh(building)
    return building


@router.get("/{building_id}", response_model=BuildingResponse)
def get_building(building_id: int, db: Session = Depends(get_db)):
    building = db.query(Building).filter(Building.id == building_id).first()
    if not building:
        raise HTTPException(status_code=404, detail="건물을 찾을 수 없습니다.")
    return building


@router.delete("/{building_id}", status_code=204)
def delete_building(building_id: int, db: Session = Depends(get_db)):
    building = db.query(Building).filter(Building.id == building_id).first()
    if not building:
        raise HTTPException(status_code=404, detail="건물을 찾을 수 없습니다.")
    db.delete(building)
    db.commit()
