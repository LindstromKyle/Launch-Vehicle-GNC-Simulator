import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException

from app.models.simulation_models import SimulationRequest, SimulationResponse
from app.paths.deps import get_executor
from app.runners.simulation_runner import run_full_orbit_simulation

simulation_router = APIRouter(prefix="/simulations", tags=["Simulations"])


@simulation_router.post("/simulate", response_model=SimulationResponse)
async def simulate(
    request: SimulationRequest,
    executor: ThreadPoolExecutor = Depends(get_executor),
):
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            executor, run_full_orbit_simulation, request
        )
        return result
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = f"Simulation failed: {str(e)} | {tb}"
        raise HTTPException(status_code=500, detail=error_detail)
