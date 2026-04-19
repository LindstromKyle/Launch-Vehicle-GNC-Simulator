from concurrent.futures import ThreadPoolExecutor

from app.runners.monte_carlo_runner import MonteCarloRunner
from app.settings import Settings
from app.storage.monte_carlo_storage import MonteCarloStorage

settings = Settings()

# Simulation tasks can run for a long time, so use a shared thread pool.
executor = ThreadPoolExecutor(max_workers=settings.simulator_executor_max_workers)

mc_storage = MonteCarloStorage()
mc_runner = MonteCarloRunner()
