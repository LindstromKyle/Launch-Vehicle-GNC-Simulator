import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.models.simulation_models import LiveSimulationStartResponse, SimulationRequest
from app.paths.deps import executor
from app.runners.multi_orbital_runner import run_constellation_simulation
from app.runners.simulation_runner import run_full_orbit_simulation
from app.storage.live_telemetry_storage import live_telemetry_storage

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
):
    """Start a simulation in the background and expose live telemetry by run id."""
    run_id = live_telemetry_storage.create_run()
    status_payload = live_telemetry_storage.get_run_status(run_id)

    def _run_background() -> None:
        live_telemetry_storage.mark_running(run_id)
        try:
            result = run_full_orbit_simulation(
                request=request,
                telemetry_callback=lambda frame: _emit_live_frame(run_id, frame),
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


def _emit_live_frame(run_id: str, frame: dict[str, Any]) -> None:
    """Attach run metadata and persist an emitted telemetry frame."""
    live_telemetry_storage.append_frame(run_id=run_id, frame=frame)


@live_router.post(
    "/live/constellation/start", response_model=LiveSimulationStartResponse
)
async def start_constellation_simulation(
    telemetry_interval: float = Query(10.0, gt=0),
):
    """Start a W-series constellation simulation and stream live telemetry.

    Runs three satellites (W-1, W-2, W-3) in distinct LEO orbital planes.
    Each telemetry frame is tagged with ``vehicle_id`` so the frontend can
    render separate traces per satellite.
    """
    run_id = live_telemetry_storage.create_run()
    status_payload = live_telemetry_storage.get_run_status(run_id)

    def _run_background() -> None:
        live_telemetry_storage.mark_running(run_id)
        try:
            run_constellation_simulation(
                telemetry_callback=lambda frame: _emit_live_frame(run_id, frame),
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


@live_router.websocket("/live/{run_id}/ws")
async def stream_live_frames_ws(websocket: WebSocket, run_id: str):
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
                await websocket.send_json({"type": "telemetry", "data": frame})
                last_seq = max(last_seq, int(frame["seq"]))

            if status["status"] in {"completed", "failed"}:
                await websocket.send_json({"type": "status", "data": status})
                return

            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
