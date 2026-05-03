import contextvars
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .controller import Controller
from .environment import Environment
from .integrator import integrate_verlet
from .mission import MissionPlanner
from .state import State
from .vehicle import Vehicle

_sim_log_name_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sim_log_name", default=None
)
_old_record_factory = logging.getLogRecordFactory()


def _record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    record = _old_record_factory(*args, **kwargs)
    record.sim_log_name = _sim_log_name_ctx.get()
    return record


logging.setLogRecordFactory(_record_factory)


class Simulator:
    """
    Main driver class to run the rocket launch simulation.

    Args:
        vehicle: The rocket Vehicle instance
        environment: Environment instance
        initial_state: Starting state vector
        mission_planner: MissionPlanner instance
        t_0: Start time of simulation (seconds)
        t_final: End time of simulation (seconds)
        delta_t: Fixed time step size (seconds)
        log_interval: How often to write detailed logs (seconds)
        log_name: Name for the log file
    """

    def __init__(
        self,
        vehicle: Vehicle,
        environment: Environment,
        initial_state: State,
        mission_planner: MissionPlanner,
        t_0: float = 0,
        t_final: float = 2000,
        delta_t: float = 0.5,
        log_interval: float = 1,
        log_name: str = "simulation",
    ) -> None:

        self.vehicle = vehicle
        self.environment = environment
        self.initial_state = initial_state
        self.t_0 = t_0
        self.t_final = t_final
        self.delta_t = delta_t
        self.log_interval = log_interval
        self.controller = None
        self.mission_planner = mission_planner
        self.log_name = log_name

    def add_controller(self, controller: Controller) -> None:
        """
        Attach an attitude/thrust controller to the simulator.

        Args:
            controller: Controller instance
        """
        self.controller = controller

    def run(
        self,
        telemetry_callback: Callable[[dict[str, Any]], None] | None = None,
        telemetry_interval: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Execute the full simulation using the selected integrator.

        Args:
            telemetry_callback: Optional callback for receiving live telemetry frames
            telemetry_interval: Minimum seconds between emitted telemetry frames

        Returns:
            Tuple containing:
            - Array of time points (s)
            - Array of state vectors at each time point
            - List of (time, phase_name) tuples marking phase transitions
        """

        # Ensure the logs directory exists (creates it if missing, including any parent folders)
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / f"{self.log_name}.log"

        # Attach a dedicated FileHandler to the root logger so this works even
        # when uvicorn (or another framework) has already called basicConfig.
        token = _sim_log_name_ctx.set(self.log_name)

        class _SimLogFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return getattr(record, "sim_log_name", None) == self.log_name

            def __init__(self, log_name: str):
                super().__init__()
                self.log_name = log_name

        _file_handler = logging.FileHandler(logfile, mode="w")
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        _file_handler.addFilter(_SimLogFilter(self.log_name))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(_file_handler)
        try:
            # Integrate
            t_vals, state_vals, phase_transitions = integrate_verlet(
                vehicle=self.vehicle,
                environment=self.environment,
                initial_state=self.initial_state.as_vector(),
                t_0=self.t_0,
                t_final=self.t_final,
                delta_t=self.delta_t,
                log_interval=self.log_interval,
                controller=self.controller,
                mission_planner=self.mission_planner,
                telemetry_callback=telemetry_callback,
                telemetry_interval=telemetry_interval,
            )
            return t_vals, state_vals, phase_transitions
        finally:
            root_logger.removeHandler(_file_handler)
            _file_handler.close()
            _sim_log_name_ctx.reset(token)
