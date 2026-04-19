import asyncio
import traceback

from fastapi import APIRouter, HTTPException

from app.models.simulation_models import (
    MonteCarloBatchResponse,
    MonteCarloKickoffResponse,
    MonteCarloRequest,
)
from app.paths.deps import executor, mc_runner, mc_storage

monte_carlo_router = APIRouter(prefix="/simulations", tags=["Simulations"])


@monte_carlo_router.post("/monte-carlo", response_model=MonteCarloKickoffResponse)
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
        return MonteCarloKickoffResponse(
            batch_id=batch_status["batch_id"],
            created_at=batch_status["created_at"],
            status=batch_status["status"],
            total_simulations=batch_status["total_simulations"],
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
        mc_storage.mark_batch_failed(
            batch_id=batch_id,
            simulations=results,
            error_msg=error_msg,
        )


@monte_carlo_router.get(
    "/monte-carlo/{batch_id}", response_model=MonteCarloBatchResponse
)
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

        return MonteCarloBatchResponse(
            batch_id=batch_data["batch_id"],
            created_at=batch_data["created_at"],
            status=batch_data["status"],
            total_simulations=total_simulations,
            completed_simulations=completed_simulations,
            failed_simulations=total_simulations - completed_simulations,
            success_rate=(
                (completed_simulations / total_simulations)
                if total_simulations
                else 0.0
            ),
            statistics=batch_data["summary"],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@monte_carlo_router.get("/monte-carlo")
async def list_monte_carlo_runs():
    """
    List all Monte Carlo batches with basic info.
    """
    try:
        batches = mc_storage.list_batches()
        return {"batches": batches, "total": len(batches)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
