from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response

from app.observability import (
    bootstrap_observability,
    log_event,
    render_metrics_text,
)
from app.paths.deps import get_settings
from app.paths.live import live_router
from app.paths.monte_carlo import monte_carlo_router
from app.paths.simulation import simulation_router
from app.runners.monte_carlo_runner import MonteCarloRunner
from app.settings import Settings
from app.storage.live_telemetry_storage import LiveTelemetryStorage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.storage.monte_carlo_storage import MonteCarloStorage

    settings = Settings()
    app.state.settings = settings
    app.state.executor = ThreadPoolExecutor(
        max_workers=settings.simulator_executor_max_workers
    )
    app.state.mc_storage = MonteCarloStorage()
    app.state.mc_runner = MonteCarloRunner()
    app.state.live_telemetry_storage = LiveTelemetryStorage()

    try:
        yield
    finally:
        app.state.executor.shutdown(wait=True)


def create_app() -> FastAPI:
    settings = Settings()
    bootstrap_observability(otlp_endpoint=settings.otlp_endpoint)
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            if request.url.path.startswith("/simulations/live/constellation"):
                log_event(
                    level=40,
                    event="constellation_http_error",
                    request_id=request_id,
                    route=request.url.path,
                    error=str(exc),
                )
            raise

        response.headers["x-request-id"] = request_id
        return response

    app.include_router(simulation_router)
    app.include_router(live_router)
    app.include_router(monte_carlo_router)

    @app.get("/health")
    async def health(settings: Settings = Depends(get_settings)):
        return {
            "status": "healthy",
            "environment": settings.environment,
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(
            content=render_metrics_text(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


app = create_app()
