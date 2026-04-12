from fastapi import APIRouter, HTTPException
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict

from app.models.simulation_models import (
    SimulationRequest,
    SimulationResponse,
    MonteCarloRequest,
    MonteCarloResult,
    MonteCarloRunStatus,
)
from app.utils.simulation_runner import run_full_orbit_simulation
from app.utils.monte_carlo_runner import MonteCarloRunner
from app.utils.monte_carlo_storage import MonteCarloStorage

# Simulation takes a while to run, so use a thread pool
executor = ThreadPoolExecutor(max_workers=4)

# Initialize Monte Carlo storage and service
mc_storage = MonteCarloStorage()
mc_runner = MonteCarloRunner()

sim_router = APIRouter(prefix="/simulations", tags=["Simulations"])


@sim_router.post("/simulate", response_model=SimulationResponse)
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


@sim_router.post("/monte-carlo", response_model=MonteCarloRunStatus)
async def run_monte_carlo(request: MonteCarloRequest):
    """
    Initiate a Monte Carlo dispersion analysis.
    Returns immediately with a batch_id for polling results.

    Example dispersions:
    {
        "stage1_base_thrust_magnitude": {"mean": 7600000, "std_dev": 100000},
        "stage1_initial_prop_mass": {"mean": 395700, "std_dev": 5000}
    }
    """
    try:
        total_simulations = request.simulation_count()
        base_params = request.base_simulation.model_dump()
        dispersions = {
            param: disp.model_dump() for param, disp in request.dispersions.items()
        }

        # Create batch record with status "in_progress"
        batch_id = mc_storage.create_batch(
            total_simulations=total_simulations,
            base_params=base_params,
            dispersions=dispersions,
        )

        # Run Monte Carlo in background (non-blocking)
        asyncio.ensure_future(
            asyncio.get_running_loop().run_in_executor(
                executor, _run_monte_carlo_background, batch_id, request
            )
        )

        # Return immediately with batch_id and in_progress status
        batch_status = mc_storage.get_batch(batch_id)
        return MonteCarloRunStatus(
            batch_id=batch_status["batch_id"],
            created_at=batch_status["created_at"],
            status=batch_status["status"],
            total_simulations=batch_status["total_simulations"],
            completed_simulations=len(batch_status["simulations"]),
            summary=batch_status["summary"],
        )

    except Exception as e:
        tb = traceback.format_exc()
        error_detail = f"Monte Carlo setup failed: {str(e)} | {tb}"
        raise HTTPException(status_code=500, detail=error_detail)


def _run_monte_carlo_background(batch_id: str, request: MonteCarloRequest):
    """
    Background task to run all Monte Carlo simulations and finalize batch.
    """
    results = []
    try:
        total_simulations = request.simulation_count()

        # Convert dispersions from Pydantic models to dict
        dispersions = {
            param: disp.model_dump() for param, disp in request.dispersions.items()
        }

        # Run all Monte Carlo simulations
        results = mc_runner.run_monte_carlo(
            base_request=request.base_simulation,
            num_simulations=total_simulations,
            dispersions=dispersions,
        )

        # Compute summary statistics
        summary = mc_runner.compute_statistics(results)

        # Finalize batch with results and status "completed"
        mc_storage.finalize_batch(
            batch_id=batch_id,
            simulations=results,
            summary=summary,
        )

    except Exception as e:
        error_msg = f"Background MC batch failed: {str(e)}"
        batch_data = mc_storage.get_batch(batch_id)
        batch_data["simulations"] = results
        batch_data["status"] = "failed"
        batch_data["summary"] = {"error": error_msg}
        mc_storage._save_batch(batch_id, batch_data)


@sim_router.get("/monte-carlo/{batch_id}", response_model=MonteCarloResult)
async def get_monte_carlo_result(batch_id: str):
    """Retrieve results from a Monte Carlo batch."""
    try:
        batch_data = mc_storage.get_batch(batch_id)

        if batch_data["status"] == "in_progress":
            raise HTTPException(
                status_code=202, detail="Monte Carlo batch still in progress"
            )

        if batch_data["status"] == "failed":
            error_detail = "Unknown error"
            if batch_data.get("summary") and batch_data["summary"].get("error"):
                error_detail = batch_data["summary"]["error"]
            raise HTTPException(
                status_code=500,
                detail=f"Monte Carlo batch failed: {error_detail}",
            )

        completed_simulations = len(batch_data["simulations"])
        total_simulations = batch_data["total_simulations"]

        return MonteCarloResult(
            batch_id=batch_data["batch_id"],
            total_simulations=total_simulations,
            completed_simulations=completed_simulations,
            failed_simulations=total_simulations - completed_simulations,
            success_rate=(
                (completed_simulations / total_simulations)
                if total_simulations
                else 0.0
            ),
            statistics=batch_data["summary"],
            created_at=batch_data["created_at"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@sim_router.get("/monte-carlo")
async def list_monte_carlo_runs():
    """
    List all Monte Carlo batches with basic info.
    """
    try:
        batches = mc_storage.list_batches()
        return {"batches": batches, "total": len(batches)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
