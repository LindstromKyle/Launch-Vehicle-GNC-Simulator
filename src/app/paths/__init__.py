from app.paths.live import live_router
from app.paths.monte_carlo import monte_carlo_router
from app.paths.simulation import simulation_router

__all__ = [
    "simulation_router",
    "live_router",
    "monte_carlo_router",
]
