import asyncio
import traceback

from fastapi import APIRouter, HTTPException

from app.models.simulation_models import SimulationRequest, SimulationResponse
from app.paths.deps import executor
from app.runners.simulation_runner import run_full_orbit_simulation

simulation_router = APIRouter(prefix="/simulations", tags=["Simulations"])


@simulation_router.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            executor, run_full_orbit_simulation, request
        )
        return result
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = f"Simulation failed: {str(e)} | {tb}"
        raise HTTPException(status_code=500, detail=error_detail)
