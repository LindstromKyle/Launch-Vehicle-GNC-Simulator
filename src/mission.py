import logging
import numpy as np

from abc import ABC, abstractmethod

from environment import Environment
from utils import compute_orbital_elements, compute_time_to_apoapsis
from vehicle import Vehicle


class Phase(ABC):
    """
    Abstract base class for mission phases. Each phase defines its attitude/throttle setpoints and when it should end.
    """

    @abstractmethod
    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        """
        Check if this phase should end based on current time and state.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector
            elements: Orbital elements dictionary

        Returns:
            True if the phase is finished, False otherwise
        """
        pass

    @abstractmethod
    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        """
        Return the control setpoints for this phase.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector
            elements: Orbital elements dictionary

        Returns:
            Dictionary of setpoints (attitude_mode, throttle, etc.)
        """
        pass


class TimeBasedPhase(Phase):
    """
    Phase that ends after a fixed amount of time has passed.

    Args:
        end_time: Simulation time when this phase should finish (s)
        attitude_mode: Desired attitude mode string
        throttle: Throttle level (0.0 to 1.0)
        name: Name of the phase
    """

    def __init__(self, end_time: float, attitude_mode: str, throttle: float = 1.0, name: str = "Unnamed"):
        self.end_time = end_time
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class CoastPhase(Phase):
    """
    Coast phase that ends when close to apoapsis (or when circularization burn should start).

    Args:
        time_to_apo_threshold: Time (s) remaining to apoapsis to trigger end
        attitude_mode: Desired attitude during coast
        throttle: TODO: should always be zero?
        name: Name of the phase
        buffer: Extra time margin for burn preparation (s)
        use_dynamic_threshold: Whether to estimate burn time dynamically
    """

    def __init__(
        self,
        time_to_apo_threshold: float = 30.0,
        attitude_mode: str = "prograde",
        throttle: float = 0.0,
        name: str = "Coast",
        buffer: float = 5.0,
        use_dynamic_threshold: bool = True,
    ):
        self.time_to_apo_threshold = time_to_apo_threshold
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name
        self.buffer = buffer
        self.use_dynamic_threshold = use_dynamic_threshold

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False

        position = state_vector[:3]
        velocity = state_vector[3:6]

        if (
            not self.use_dynamic_threshold
            or elements.get("apoapsis_radius") == float("inf")
            or elements["semi_major_axis"] <= 0
        ):
            time_to_apo = compute_time_to_apoapsis(position, velocity, elements, self.mu)
            return time_to_apo <= self.time_to_apo_threshold

        # Dynamic computation
        r_apo = elements["apoapsis_radius"]
        a = elements["semi_major_axis"]
        v_apo = np.sqrt(self.mu * (2 / r_apo - 1 / a))
        v_circ = np.sqrt(self.mu / r_apo)
        delta_v = v_circ - v_apo

        if delta_v <= 0:
            return True  # Already circular

        # Estimate burn time
        prop_mass = state_vector[13]
        m0 = self.vehicle.dry_mass + prop_mass
        ve = self.vehicle.average_isp * 9.80665
        T = self.vehicle.base_thrust_magnitude
        if T <= 0 or ve <= 0 or m0 <= 0:
            return False

        burn_time = (m0 * ve / T) * (1 - np.exp(-delta_v / ve))
        half_burn = burn_time / 2 + self.buffer

        time_to_apo = compute_time_to_apoapsis(position, velocity, elements, self.mu)
        return time_to_apo <= half_burn

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class CircBurnPhase(Phase):
    """
    Circularization burn phase — raises periapsis to target altitude.

    Args:
        attitude_mode: Attitude mode to hold while burning
        throttle: Maximum throttle level
        name: Name of the phase
        min_throttle: Minimum throttle during fine control
        throttle_kp: Proportional gain for throttle adjustment
        target_eccentricity: Desired final eccentricity
    """

    def __init__(
        self,
        attitude_mode: str = "prograde",
        throttle: float = 1.0,
        name: str = "Unnamed",
        min_throttle: float = 0.1,
        throttle_kp: float = 20.0,
        target_eccentricity: float = 0.002,
    ):
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name
        self.min_throttle = min_throttle
        self.throttle_kp = throttle_kp
        self.target_eccentricity = target_eccentricity

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False

        # TODO: better way of doing this - currently checking for 1000km apoapsis (missed the cutoff)
        if elements["apoapsis_radius"] > 6371000 + 1000000:
            return True

        # Eccentricity checks
        ecc_ok = elements["eccentricity"] < self.target_eccentricity

        return ecc_ok

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        if elements["apoapsis_radius"] == float("inf") or elements["periapsis_radius"] <= 0:
            # Safety: Fallback for hyperbolic or invalid orbits
            throttle = 0.0
        else:
            ratio = elements["periapsis_radius"] / elements["apoapsis_radius"]
            error = max(0.0, 1.0 - ratio)  # Positive error
            throttle = max(self.min_throttle, min(1.0, self.throttle_kp * error))

        return {
            "throttle": throttle,
            "attitude_mode": self.attitude_mode,
            "target_r": elements.get("apoapsis_radius", float("inf")),
        }


class ProgrammedPitchPhase(Phase):
    """
    Phase that performs a smooth pitch-over maneuver from initial to final pitch angle.

    Args:
        end_time: Simulation time when this phase should end (s)
        initial_pitch_deg: Starting pitch angle from vertical (degrees)
        final_pitch_deg: Ending pitch angle from vertical (degrees)
        kick_direction: Direction of the horizontal kick (default = east [0,1,0])
        throttle: Throttle level (0.0–1.0)
        name: Name of the phase
    """

    def __init__(
        self,
        end_time: float,
        initial_pitch_deg: float,
        final_pitch_deg: float,
        kick_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),  # Default east
        throttle: float = 1.0,
        name: str = "Pitch Program",
    ):
        self.end_time = end_time
        self.initial_pitch_deg = initial_pitch_deg
        self.final_pitch_deg = final_pitch_deg
        self.kick_direction = kick_direction
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "programmed_pitch"

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "initial_pitch_deg": self.initial_pitch_deg,
            "final_pitch_deg": self.final_pitch_deg,
            "kick_direction": self.kick_direction,
        }


class PEGPhase(Phase):
    """
    Powered Explicit Guidance phase. Uses iterative PEG algorithm to steer toward specified apoapsis/periapsis.

    Args:
        target_apoapsis: Desired apoapsis radius (m)
        target_periapsis: Desired periapsis radius (m)
        target_inclination: Target inclination (degrees)
        apo_tolerance: Acceptable error in apoapsis (m)
        peri_tolerance: Acceptable error in periapsis (m)
        min_throttle: Minimum throttle during terminal guidance
        throttle_kp: Proportional gain for throttle adjustment near target
        throttle_threshold_factor: Scaling factor for when to reduce throttle
        throttle: Maximum throttle level
        name: Name of the phase
    """

    def __init__(
        self,
        target_apoapsis: float,
        target_periapsis: float | None = None,
        target_inclination: float | None = None,
        apo_tolerance: float = 5000.0,
        peri_tolerance: float = 5000.0,
        min_throttle: float = 0.1,
        throttle_kp: float = 20.0,
        throttle_threshold_factor: float = 5.0,
        throttle: float = 1.0,
        name: str = "PEG Ascent",
    ):
        self.target_apoapsis = target_apoapsis
        self.target_periapsis = target_periapsis or (
            target_apoapsis - 100000.0
        )  # Default to slight ellipse if not specified
        self.target_a = (self.target_apoapsis + self.target_periapsis) / 2
        self.target_e = (self.target_apoapsis - self.target_periapsis) / (self.target_apoapsis + self.target_periapsis)
        self.target_inclination = target_inclination
        self.apo_tolerance = apo_tolerance
        self.peri_tolerance = peri_tolerance
        self.min_throttle = min_throttle
        self.throttle_kp = throttle_kp
        self.throttle_threshold_factor = throttle_threshold_factor
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "peg"

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False
        apo_ok = abs(elements["apoapsis_radius"] - self.target_apoapsis) <= self.apo_tolerance
        peri_ok = abs(elements["periapsis_radius"] - self.target_periapsis) <= self.peri_tolerance
        inc_ok = True
        if self.target_inclination is not None:
            h_vec = np.cross(state_vector[:3], state_vector[3:6])
            h_norm = np.linalg.norm(h_vec)
            if h_norm > 0:
                inc_current = np.rad2deg(np.arccos(h_vec[2] / h_norm))
                inc_ok = abs(inc_current - self.target_inclination) <= 0.1  # 0.1 deg tolerance
        return apo_ok and peri_ok and inc_ok

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        setpoints = {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "target_apoapsis": self.target_apoapsis,
            "target_periapsis": self.target_periapsis,
            "target_r": self.target_periapsis,
            "target_a": self.target_a,
            "target_inclination": self.target_inclination,
        }
        if elements is not None:
            apo_error = abs(self.target_apoapsis - elements["apoapsis_radius"])
            peri_error = abs(self.target_periapsis - elements["periapsis_radius"])
            max_error = max(apo_error, peri_error)
            if self.target_inclination is not None:
                h_vec = np.cross(state_vector[:3], state_vector[3:6])
                h_norm = np.linalg.norm(h_vec)
                if h_norm > 0:
                    inc_current = np.rad2deg(np.arccos(h_vec[2] / h_norm))
                    inc_error = abs(inc_current - self.target_inclination) * 10000.0
                    max_error = max(max_error, inc_error)
            throttle_threshold = self.throttle_threshold_factor * max(self.apo_tolerance, self.peri_tolerance)
            if max_error < throttle_threshold:
                normalized_error = max_error / self.target_apoapsis
                setpoints["throttle"] = max(self.min_throttle, self.throttle_kp * normalized_error)
        return setpoints


class MissionPlanner:
    """
    Manages mission phases and provides current setpoints to guidance/controller.

    Args:
        phases: Ordered list of Phase objects
        environment: Environment instance
        vehicle: Vehicle instance
        start_time: Simulation start time
    """

    def __init__(self, phases: list[Phase], environment: Environment, vehicle: Vehicle, start_time: float = 0.0):
        self.phases = phases
        self.current_phase_idx = 0
        self.current_phase = phases[0]
        self.mu = environment.gravitational_constant * environment.earth_mass
        self.environment = environment
        self.vehicle = vehicle
        self.phase_transitions = [(start_time, phases[0].name)]
        # Inject mu to phases if needed
        for phase in self.phases:
            # TODO: find better way than his hack
            phase.mu = self.mu
            phase.vehicle = self.vehicle

    def update(self, time: float, state_vector: np.ndarray, log_flag: bool) -> dict:
        """
        Get current phase setpoints and check for phase transitions.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector
            log_flag: Whether to log status this step

        Returns:
            Dictionary of current setpoints for guidance and controller
        """
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        elements = compute_orbital_elements(position, velocity, self.mu)
        altitude = (np.linalg.norm(position) - self.environment.earth_radius) / 1000

        self.current_phase = self.phases[self.current_phase_idx]
        if self.current_phase.is_complete(time, state_vector, elements):
            self.current_phase_idx += 1
            if self.current_phase_idx < len(self.phases):
                self.phase_transitions.append((time, self.phases[self.current_phase_idx].name))
            else:
                logging.info(f"Integration segment complete at t={time:.2f}")
                return {"throttle": 0.0, "attitude_mode": "prograde"}
            self.current_phase = self.phases[self.current_phase_idx]

        # Get base setpoints from phase
        setpoints = self.current_phase.get_setpoints(time, state_vector, elements)

        # Add time info for pitch program
        if setpoints.get("attitude_mode") == "programmed_pitch":
            current_phase_start_time = self.phase_transitions[self.current_phase_idx][0]
            setpoints["start_time"] = current_phase_start_time
            duration = self.current_phase.end_time - current_phase_start_time
            setpoints["duration"] = duration

        # Add additional phase setpoints
        setpoints["mu"] = self.mu
        setpoints["g0"] = 9.80665  # Standard gravity
        setpoints["thrust"] = self.vehicle.base_thrust_magnitude
        setpoints["isp"] = self.vehicle.average_isp
        setpoints["dry_mass"] = self.vehicle.dry_mass

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

        return setpoints

    def get_phase_transitions(self) -> list[tuple[float, str]]:
        """
        Return list of phase change events for plotting/analysis.

        Returns:
            List of (time, phase_name) tuples
        """
        return self.phase_transitions
