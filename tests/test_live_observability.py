from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.storage.live_telemetry_storage import LiveTelemetryStorage


def _metric_value(
    metrics_text: str, metric_name: str, labels: str | None = None
) -> float:
    prefix = metric_name if labels is None else f"{metric_name}{{{labels}}}"
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix):
            return float(line.split()[-1])
    return 0.0


@pytest.fixture
def observability_client(monkeypatch):
    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    def _fake_constellation_run(
        telemetry_callback,
        command_provider,
        command_event_callback,
    ):
        _ = (command_provider, command_event_callback)
        telemetry_callback({"time_s": 0.0, "vehicle_id": "W-4"})

    monkeypatch.setattr(
        "app.paths.live.run_constellation_simulation",
        _fake_constellation_run,
    )

    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    app.state.executor = ThreadPoolExecutor(max_workers=2)
    app.state.live_telemetry_storage = LiveTelemetryStorage()

    with TestClient(app) as client:
        yield client, app

    app.state.executor.shutdown(wait=True)


def test_constellation_command_propagates_request_id(observability_client):
    client, _ = observability_client
    response = client.post(
        "/simulations/live/constellation/command",
        headers={"x-request-id": "req-123"},
        json={
            "run_id": "missing-run",
            "vehicle_id": "W-4",
            "action": "deorbit_burn",
            "execute_at_sim_time_s": 1.0,
            "target_perigee_alt_km": 100.0,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert response.json()["status"] == "rejected"


def test_constellation_run_completion_metric_increments(observability_client):
    client, _ = observability_client

    before = client.get("/metrics").text
    before_completed = _metric_value(
        before,
        "constellation_run_outcome_total",
        'status="completed"',
    )

    start_response = client.post("/simulations/live/constellation/start")
    assert start_response.status_code == 200

    after_completed = before_completed
    for _ in range(20):
        time.sleep(0.05)
        after = client.get("/metrics").text
        after_completed = _metric_value(
            after,
            "constellation_run_outcome_total",
            'status="completed"',
        )
        if after_completed >= before_completed + 1.0:
            break

    assert after_completed >= before_completed + 1.0


def test_process_cpu_and_memory_metrics_exposed(observability_client):
    client, _ = observability_client

    metrics_text = client.get("/metrics").text
    cpu_seconds = _metric_value(metrics_text, "app_process_cpu_seconds_total")
    resident_memory_bytes = _metric_value(
        metrics_text,
        "app_process_resident_memory_bytes",
    )

    assert cpu_seconds >= 0.0
    assert resident_memory_bytes > 0.0
