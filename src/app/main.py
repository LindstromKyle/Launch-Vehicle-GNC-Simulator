from fastapi import FastAPI

from app.paths.simulation_paths import sim_router
from app.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    debug=settings.debug,
)

app.include_router(sim_router)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": settings.environment,
    }
