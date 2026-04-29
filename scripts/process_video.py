"""
CLI에서 직접 영상을 처리하는 스크립트.
웹 UI 없이 빠르게 테스트할 때 사용합니다.

사용법:
  python scripts/process_video.py --building "테스트빌딩" --video path/to/video.mp4
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from app.core.database import SessionLocal, init_db
from app.models.building import Building
from app.models.inspection import Inspection, InspectionStatus
from ml.pipeline import InspectionPipeline


def main():
    parser = argparse.ArgumentParser(description="창문 오염도 검사 CLI")
    parser.add_argument("--building", required=True, help="건물명")
    parser.add_argument("--video", required=True, help="영상 파일 경로")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    init_db()
    db = SessionLocal()

    try:
        building = db.query(Building).filter(Building.name == args.building).first()
        if not building:
            building = Building(name=args.building)
            db.add(building)
            db.commit()
            db.refresh(building)
            print(f"[OK] 건물 등록: {building.name} (id={building.id})")

        video_filename = os.path.basename(args.video)
        inspection = Inspection(
            building_id=building.id,
            video_filename=video_filename,
            status=InspectionStatus.PENDING,
        )
        db.add(inspection)
        db.commit()
        db.refresh(inspection)
        print(f"[OK] 검사 생성: id={inspection.id}")

        print(f"[...] 처리 중: {args.video}")
        pipeline = InspectionPipeline()
        pipeline.process(inspection, args.video, db)

        db.refresh(inspection)
        print(f"\n[완료] 검사 결과 (id={inspection.id})")
        print(f"  상태: {inspection.status.value}")
        print(f"  탐지된 창문: {inspection.total_windows}개")
        print(f"  처리 프레임: {inspection.processed_frames} / {inspection.total_frames}")
        print(f"\n  웹 대시보드 http://localhost:8000/result/{inspection.id} 에서 확인하세요.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
