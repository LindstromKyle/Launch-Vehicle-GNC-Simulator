from fastapi import FastAPI

from app.paths.live import live_router
from app.paths.monte_carlo import monte_carlo_router
from app.paths.simulation import simulation_router
from app.settings import Settings

settings = Settings()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    debug=settings.debug,
)

app.include_router(simulation_router)
app.include_router(live_router)
app.include_router(monte_carlo_router)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": settings.environment,
    }
