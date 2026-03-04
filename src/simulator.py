import logging

from environment import Environment
from integrator import integrate_rk4, integrate_verlet
from mission import MissionPlanner
from state import State
from vehicle import Vehicle


class Simulator:
    """
    Main driver class to run the simulation
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
    ):
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

    def add_controller(self, controller):
        """
        Adds the selected attitude controller
        """
        self.controller = controller

    def run(self):
        """
        Runs simulation loop
        """

        # Set up logging
        logging.basicConfig(
            filename=f"../logs/{self.log_name}.log",
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
