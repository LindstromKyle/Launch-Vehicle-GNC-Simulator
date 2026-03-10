import logging
import numpy as np

from environment import Environment
from guidance import GuidancePhase
from utils import compute_orbital_elements
from vehicle import Vehicle


class MissionPlanner:
    """
    Manages mission guidance_phases and provides current setpoints to guidance/controller.

    Args:
        guidance_phases: Ordered list of GuidancePhase objects
        environment: Environment instance
        vehicle: Vehicle instance
        start_time: Simulation start time
    """

    def __init__(
        self, guidance_phases: list[GuidancePhase], environment: Environment, vehicle: Vehicle, start_time: float = 0.0
    ):
        self.guidance_phases = guidance_phases
        self.current_phase_idx = 0
        self.current_phase = guidance_phases[0]
        self.environment = environment
        self.vehicle = vehicle
        self.phase_transitions = [(start_time, guidance_phases[0].name)]
        self.mu = environment.gravitational_constant * environment.earth_mass

    def update(self, time: float, state_vector: np.ndarray, log_flag: bool) -> dict:
        """
        Get current phase setpoints and check for phase transitions.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector
            log_flag: Whether to log status this step

        Returns:
            Dictionary of current setpoints (Throttle and attitude mode)
        """
        position = state_vector[0:3]
        velocity = state_vector[3:6]

        elements = compute_orbital_elements(position, velocity, self.mu)
        altitude = (np.linalg.norm(position) - self.environment.earth_radius) / 1000

        self.current_phase = self.guidance_phases[self.current_phase_idx]
        if self.current_phase.is_complete(time, state_vector, elements):
            self.current_phase_idx += 1
            if self.current_phase_idx < len(self.guidance_phases):
                self.phase_transitions.append((time, self.guidance_phases[self.current_phase_idx].name))
            else:
                logging.info(f"Integration segment complete at t={time:.2f}")
                return {"throttle": 0.0, "attitude_mode": "prograde"}
            self.current_phase = self.guidance_phases[self.current_phase_idx]

        if log_flag:
            # Full orbital velocity
            r_unit_vector = position / np.linalg.norm(position)
            orbital_velocity = np.linalg.norm(velocity)
            # Radial velocity
            radial_velocity = np.dot(velocity, r_unit_vector)
            # Tangential velocity
            tangential_velocity = (
                np.sqrt(orbital_velocity**2 - radial_velocity**2)
                if orbital_velocity**2 - radial_velocity**2 > 0.0
                else 0.0
            )
            logging.info(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            logging.info(f"---------------------------------[MISSION PLANNER]----------------------------------------")
            logging.info(f"time (s): {time:.2f} | phase: {self.current_phase.name}")
            logging.info(
                f"current altitude (km): {altitude:.4f} | "
                f"apoapsis altitude (km): {((elements["apoapsis_radius"] - self.environment.earth_radius) / 1000):.4f} | "
                f"periapsis altitude (km): {((elements["periapsis_radius"] - self.environment.earth_radius) / 1000):.4f}"
            )
            logging.info(
                f"orbital vel (km/s): {orbital_velocity/1000:.4f} | tangential vel (km/s): {tangential_velocity/1000:.4f} | radial vel (km/s): {radial_velocity/1000:.4f}"
            )

        # Get base setpoints from phase
        setpoints = self.current_phase.get_setpoints(time, state_vector, elements)
        return setpoints

    def get_phase_transitions(self) -> list[tuple[float, str]]:
        """
        Return list of phase change events for plotting/analysis.

        Returns:
            List of (time, phase_name) tuples
        """
        return self.phase_transitions
