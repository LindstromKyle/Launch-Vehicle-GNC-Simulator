import logging
from pathlib import Path
import numpy as np

from environment import Environment
from integrator import integrate_rk4, integrate_verlet
from mission import MissionPlanner
from state import State
from vehicle import Vehicle
from controller import Controller


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

    def run(self) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Execute the full simulation using the selected integrator.

        Returns:
            Tuple containing:
            - Array of time points (s)
            - Array of state vectors at each time point
            - List of (time, phase_name) tuples marking phase transitions
        """

        # Ensure the logs directory exists (creates it if missing, including any parent folders)
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / f"{self.log_name}.log"
        
        # Set up logging
        logging.basicConfig( 
            filename=logfile,
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
            filemode="w",
        )

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
        )

        return t_vals, state_vals, phase_transitions
