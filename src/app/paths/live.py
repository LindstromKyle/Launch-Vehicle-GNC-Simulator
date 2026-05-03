import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.models.simulation_models import (
    CommandUploadResponse,
    DeorbitCommandRequest,
    LiveSimulationStartResponse,
    SimulationRequest,
)
from app.paths.deps import get_executor, get_live_telemetry_storage
from app.runners.multi_orbital_runner import CONSTELLATION, run_constellation_simulation
from app.runners.simulation_runner import run_full_orbit_simulation
from app.storage.live_telemetry_storage import LiveTelemetryStorage

live_router = APIRouter(prefix="/simulations", tags=["Simulations"])
_LIVE_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "live"
_LIVE_HTML_PATH = _LIVE_FRONTEND_DIR / "index.html"
_LIVE_CSS_PATH = _LIVE_FRONTEND_DIR / "styles.css"
_LIVE_JS_PATH = _LIVE_FRONTEND_DIR / "app.js"
_NO_CACHE_HEADERS = {"Cache-Control": "no-store, max-age=0"}


@live_router.get("/live/view")
async def live_viewer_page():
    """Serve the live telemetry frontend page."""
    return FileResponse(_LIVE_HTML_PATH, headers=_NO_CACHE_HEADERS)


@live_router.get("/live/assets/styles.css")
async def live_viewer_styles():
    return FileResponse(
        _LIVE_CSS_PATH,
        media_type="text/css",
        headers=_NO_CACHE_HEADERS,
    )


@live_router.get("/live/assets/app.js")
async def live_viewer_script():
    return FileResponse(
        _LIVE_JS_PATH,
        media_type="application/javascript",
        headers=_NO_CACHE_HEADERS,
    )


@live_router.post("/live/start", response_model=LiveSimulationStartResponse)
async def start_live_simulation(
    request: SimulationRequest,
    telemetry_interval: float = Query(0.5, gt=0),
    executor: ThreadPoolExecutor = Depends(get_executor),
    live_telemetry_storage: LiveTelemetryStorage = Depends(get_live_telemetry_storage),
):
    """Start a simulation in the background and expose live telemetry by run id."""
    run_id = live_telemetry_storage.create_run(run_kind="single_orbit")
    status_payload = live_telemetry_storage.get_run_status(run_id)

    def _run_background() -> None:
        live_telemetry_storage.mark_running(run_id)
        try:
            result = run_full_orbit_simulation(
                request=request,
                telemetry_callback=lambda frame: _emit_live_frame(
                    live_telemetry_storage,
                    run_id,
                    frame,
                ),
                telemetry_interval=telemetry_interval,
            )
            live_telemetry_storage.mark_completed(
                run_id=run_id,
                summary=result.get("summary"),
            )
        except Exception as exc:
            live_telemetry_storage.mark_failed(run_id=run_id, error=str(exc))

    asyncio.get_running_loop().run_in_executor(executor, _run_background)

    return LiveSimulationStartResponse(
        run_id=run_id,
        status=status_payload["status"],
        created_at=status_payload["created_at"],
    )


def _emit_live_frame(
    live_telemetry_storage: LiveTelemetryStorage,
    run_id: str,
    frame: dict[str, Any],
) -> None:
    """Attach run metadata and persist an emitted telemetry frame."""
    live_telemetry_storage.append_frame(run_id=run_id, frame=frame)


@live_router.post(
    "/live/constellation/start", response_model=LiveSimulationStartResponse
)
async def start_constellation_simulation(
    telemetry_interval: float = Query(10.0, gt=0),
    executor: ThreadPoolExecutor = Depends(get_executor),
    live_telemetry_storage: LiveTelemetryStorage = Depends(get_live_telemetry_storage),
):
    """Start a W-series constellation simulation and stream live telemetry.

    Runs three satellites in distinct LEO orbital planes.
    Each telemetry frame is tagged with ``vehicle_id`` so the frontend can
    render separate traces per satellite.
    """
    allowed_vehicle_ids = [name for name, *_ in CONSTELLATION]
    run_id = live_telemetry_storage.create_run(
        run_kind="constellation",
        allowed_vehicle_ids=allowed_vehicle_ids,
    )
    status_payload = live_telemetry_storage.get_run_status(run_id)

    def _run_background() -> None:
        live_telemetry_storage.mark_running(run_id)
        try:
            run_constellation_simulation(
                telemetry_callback=lambda frame: _emit_live_frame(
                    live_telemetry_storage,
                    run_id,
                    frame,
                ),
                command_provider=lambda vehicle_id, sim_time_s: live_telemetry_storage.pop_due_commands(
                    run_id=run_id,
                    vehicle_id=vehicle_id,
                    sim_time_s=sim_time_s,
                ),
                command_event_callback=lambda event: _emit_live_frame(
                    live_telemetry_storage,
                    run_id,
                    event,
                ),
            )
            live_telemetry_storage.mark_completed(run_id=run_id, summary=None)
        except Exception as exc:
            live_telemetry_storage.mark_failed(run_id=run_id, error=str(exc))

    asyncio.get_running_loop().run_in_executor(executor, _run_background)

    return LiveSimulationStartResponse(
        run_id=run_id,
        status=status_payload["status"],
        created_at=status_payload["created_at"],
    )


@live_router.post("/live/constellation/command", response_model=CommandUploadResponse)
async def upload_constellation_command(
    request: DeorbitCommandRequest,
    live_telemetry_storage: LiveTelemetryStorage = Depends(get_live_telemetry_storage),
):
    """Queue a deorbit command for a constellation vehicle.

    Validates and queues commands, then emits acceptance telemetry.
    """
    upload = live_telemetry_storage.enqueue_deorbit_command(
        run_id=request.run_id,
        vehicle_id=request.vehicle_id,
        execute_at_sim_time_s=request.execute_at_sim_time_s,
        target_perigee_alt_km=request.target_perigee_alt_km,
    )

    if upload["status"] == "accepted":
        _emit_live_frame(
            live_telemetry_storage,
            request.run_id,
            {
                "event_type": "command",
                "event_name": "command_accepted",
                "action": request.action,
                "command_id": upload["command_id"],
                "vehicle_id": request.vehicle_id,
                "execute_at_sim_time_s": request.execute_at_sim_time_s,
                "target_perigee_alt_km": request.target_perigee_alt_km,
                "server_received_at": upload["server_received_at"],
            },
        )

    return CommandUploadResponse(
        command_id=upload["command_id"],
        status=upload["status"],
        server_received_at=upload["server_received_at"],
        reason=upload.get("reason"),
    )


@live_router.websocket("/live/{run_id}/ws")
async def stream_live_frames_ws(
    websocket: WebSocket,
    run_id: str,
    live_telemetry_storage: LiveTelemetryStorage = Depends(get_live_telemetry_storage),
):
    await websocket.accept()
    last_seq = 0
    try:
        while True:
            try:
                status = live_telemetry_storage.get_run_status(run_id)
            except KeyError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Live run {run_id} not found",
                    }
                )
                await websocket.close(code=1008)
                return

            frames_payload = live_telemetry_storage.get_frames_after(
                run_id=run_id,
                after_seq=last_seq,
                limit=500,
            )

            for frame in frames_payload["frames"]:
                message_type = (
                    "command" if frame.get("event_type") == "command" else "telemetry"
                )
                await websocket.send_json({"type": message_type, "data": frame})
                last_seq = max(last_seq, int(frame["seq"]))

            if status["status"] in {"completed", "failed"}:
                await websocket.send_json({"type": "status", "data": status})
                return

            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
