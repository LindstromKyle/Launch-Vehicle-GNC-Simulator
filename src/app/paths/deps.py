from concurrent.futures import ThreadPoolExecutor

from app.runners.monte_carlo_runner import MonteCarloRunner
from app.settings import get_settings
from app.storage.monte_carlo_storage import get_monte_carlo_storage

settings = get_settings()

# Simulation tasks can run for a long time, so use a shared thread pool.
executor = ThreadPoolExecutor(max_workers=settings.simulator_executor_max_workers)

mc_storage = get_monte_carlo_storage()
mc_runner = MonteCarloRunner()
