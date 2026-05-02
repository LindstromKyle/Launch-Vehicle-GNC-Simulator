from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from app.paths.live import live_router
from app.paths.deps import get_settings
from app.paths.monte_carlo import monte_carlo_router
from app.paths.simulation import simulation_router
from app.runners.monte_carlo_runner import MonteCarloRunner
from app.settings import Settings
from app.storage.live_telemetry_storage import LiveTelemetryStorage
from app.storage.monte_carlo_storage import MonteCarloStorage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.include_router(simulation_router)
    app.include_router(live_router)
    app.include_router(monte_carlo_router)

    @app.get("/health")
    async def health(settings: Settings = Depends(get_settings)):
        return {
            "status": "healthy",
            "environment": settings.environment,
        }

    return app


app = create_app()
