from fastapi import APIRouter

from pipeline.lidar_safety import LidarSafetyGuide, run_simulation

router = APIRouter(prefix="/lidar", tags=["lidar"])


@router.get("/status")
def get_lidar_status():
    guide = LidarSafetyGuide(simulation_mode=True)
    lidar_data = guide.read_lidar()
    result = guide.analyze_safety(lidar_data)
    return result


@router.get("/simulate")
def simulate_approach():
    guide = run_simulation(n=10)
    return {
        "total_readings": len(guide.history),
        "alert_count": len(guide.alert_log),
        "final_status": guide.history[-1]["safety"],
        "alerts": guide.alert_log,
    }
