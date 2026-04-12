from fastapi import FastAPI

from app.paths.simulation_paths import sim_router

app = FastAPI(
    title="6DOF Launch Simulator",
    description="Configurable launch-to-orbit simulation for mission software testing",
)

app.include_router(sim_router)

@app.get("/health")
async def health():
    return {"status": "healthy"}
