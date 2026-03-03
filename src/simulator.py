import numpy as np
import logging

from controller import PIDAttitudeController
from guidance import ModeBasedGuidance
from mission import MissionPlanner, TimeBasedPhase, KickPhase, TargetApoapsisPhase, CoastPhase, CircBurnPhase
from plotting import plot_3D_trajectory, plot_1D_position_velocity_acceleration, plot_3D_integration_segments
from utils import compute_minimal_quaternion_rotation, rotate_vector_by_quaternion
from vehicle import Falcon9FirstStage
from environment import Environment
from state import State
from integrator import integrate_rk4, integrate_verlet


class Simulator:

    def __init__(
        self,
        vehicle,
        environment,
        initial_state,
        mission_planner,
        t_0=0,
        t_final=2000,
        delta_t=0.5,
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
        self.controller = None  # To be set later
        self.mission_planner = mission_planner
        self.log_name = log_name

    def add_controller(self, controller):
        self.controller = controller

    def run(self):
        logging.basicConfig(
            filename=f"../logs/{self.log_name}.log",
            level=logging.INFO,  # Use DEBUG for more detail
            format="[%(levelname)s] %(message)s",
            filemode="w",  # Overwrite log file each run
        )
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

    def plot_1D(self, t_vals, state_vals, axis):
        # Plot 1D params
        plot_1D_position_velocity_acceleration(t_vals, state_vals, axis, self.environment)

    def plot_3D(self, t_vals, state_vals):
        # Plot 3D Trajectory
        plot_3D_trajectory(t_vals, state_vals)
