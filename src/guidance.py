import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from environment import Environment
from utils import (
    compute_body_z_to_inertial_quat,
    compute_orbital_elements,
    compute_time_to_apoapsis,
    quaternion_from_attitude_mode,
)
from vehicle import Vehicle


class GuidancePhase(ABC):
    """
    Abstract base class for guidance guidance_phases. Each phase defines its quaternion/throttle setpoints and when it should end.
    """

    @abstractmethod
    def is_complete(self, time: float, state_vector: np.ndarray) -> bool:
        """
        Check if this phase should end based on current time and state.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector

        Returns:
            True if the phase is finished, False otherwise
        """
        pass

    @abstractmethod
    def get_setpoints(
        self, time: float, state_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return the guidance setpoints for this phase.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector

        Returns:
            Desired quaternion [w, x, y, z], and throttle modulation (0-1)
        """
        pass


class TimeBasedGuidancePhase(GuidancePhase):
    """
    GuidancePhase that ends after a fixed amount of time has passed.

    Args:
        end_time: Simulation time when this phase should finish (s)
        attitude_mode: Desired attitude mode string
        throttle: Throttle level (0.0 to 1.0)
        name: Name of the phase
    """

    def __init__(
        self,
        end_time: float,
        attitude_mode: str,
        throttle: float = 1.0,
        name: str = "Unnamed",
    ):
        self.end_time = end_time
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray) -> bool:
        return time >= self.end_time

    def get_setpoints(
        self, time: float, state_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        desired_quaternion = quaternion_from_attitude_mode(
            state_vector, self.attitude_mode
        )
        throttle = self.throttle
        return desired_quaternion, throttle


class ProgrammedPitchGuidancePhase(GuidancePhase):
    """
    GuidancePhase that performs a smooth pitch-over maneuver from initial to final pitch angle.

    Args:
        start_time: Simulation time when this phase should start (s)
        end_time: Simulation time when this phase should end (s)
        initial_pitch_deg: Starting pitch angle from vertical (degrees)
        final_pitch_deg: Ending pitch angle from vertical (degrees)
        orbital_normal: The vector describing the direction normal to the orbital plane
        kick_direction: Direction of the horizontal kick (default = east [0,1,0])
        throttle: Throttle level (0.0–1.0)
        name: Name of the phase
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        initial_pitch_deg: float,
        final_pitch_deg: float,
        orbital_normal: np.ndarray,
        kick_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),  # Default east
        throttle: float = 1.0,
        name: str = "Pitch Program",
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.initial_pitch_deg = initial_pitch_deg
        self.final_pitch_deg = final_pitch_deg
        self.orbital_normal = orbital_normal
        self.kick_direction = kick_direction
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "programmed_pitch"

    def is_complete(self, time: float, state_vector: np.ndarray) -> bool:
        return time >= self.end_time

    def get_setpoints(
        self, time: float, state_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        position = state_vector[:3]
        radial_unit_vector = position / np.linalg.norm(position)

        # Normalize time progress (0 to 1)
        progress = max(0.0, min(1.0, (time - self.start_time) / self.duration))

        # Interpolate pitch angle (from horizontal)
        pitch_deg = self.initial_pitch_deg + progress * (
            self.final_pitch_deg - self.initial_pitch_deg
        )
        pitch_rad = np.deg2rad(pitch_deg)

        cross_product = np.cross(self.orbital_normal, radial_unit_vector)
        norm_cross = np.linalg.norm(cross_product)
        if norm_cross > 1e-6:
            horizontal_unit_vector = cross_product / norm_cross
        else:
            # Fallback if zero
            horizontal_projection = (
                self.kick_direction
                - np.dot(self.kick_direction, radial_unit_vector) * radial_unit_vector
            )
            horizontal_unit_vector = horizontal_projection / np.linalg.norm(
                horizontal_projection
            )

        # Desired z
        desired_z_vector = (
            np.cos(pitch_rad) * horizontal_unit_vector
            + np.sin(pitch_rad) * radial_unit_vector
        )
        desired_z_vector /= np.linalg.norm(desired_z_vector)

        desired_quaternion = compute_body_z_to_inertial_quat(desired_z_vector)
        throttle = self.throttle
        return desired_quaternion, throttle


class PEGGuidancePhase(GuidancePhase):
    """
    Powered Explicit Guidance phase. Uses iterative PEG algorithm to steer toward specified apoapsis/periapsis.

    Args:
        target_apoapsis: Desired apoapsis radius (m)
        target_periapsis: Desired periapsis radius (m)
        orbital_normal: The vector describing the direction normal to the orbital plane
        vehicle: Vehicle instance
        environment: Environment instance
        target_inclination: Target inclination (degrees)
        apo_tolerance: Acceptable error in apoapsis (m)
        peri_tolerance: Acceptable error in periapsis (m)
        min_throttle: Minimum throttle during terminal guidance
        throttle_kp: Proportional gain for throttle adjustment near target
        throttle_threshold_factor: Scaling factor (multiply by tolerance for when to start throttling)
        throttle: Maximum throttle level
        name: Name of the phase
    """

    def __init__(
        self,
        target_apoapsis: float,
        target_periapsis: float,
        orbital_normal: np.ndarray,
        vehicle: Vehicle,
        environment: Environment,
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
        self.target_periapsis = target_periapsis
        self.orbital_normal = orbital_normal
        self.vehicle = vehicle
        self.target_sem_maj_axis = (self.target_apoapsis + self.target_periapsis) / 2
        self.target_e = (self.target_apoapsis - self.target_periapsis) / (
            self.target_apoapsis + self.target_periapsis
        )
        self.target_inclination = target_inclination
        self.apo_tolerance = apo_tolerance
        self.peri_tolerance = peri_tolerance
        self.min_throttle = min_throttle
        self.throttle_kp = throttle_kp
        self.throttle_threshold_factor = throttle_threshold_factor
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "peg"
        self.elements = None
        self.mu = environment.gravitational_constant * environment.earth_mass

    def is_complete(self, time: float, state_vector: np.ndarray) -> bool:
        # Compute orbital elements
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        self.elements = compute_orbital_elements(position, velocity, self.mu)
        # Check peri and apo
        apo_ok = (
            abs(self.elements["apoapsis_radius"] - self.target_apoapsis)
            <= self.apo_tolerance
        )
        peri_ok = (
            abs(self.elements["periapsis_radius"] - self.target_periapsis)
            <= self.peri_tolerance
        )
        # Check inclination if specified
        inc_ok = True
        if self.target_inclination is not None:
            h_vec = np.cross(state_vector[:3], state_vector[3:6])
            h_norm = np.linalg.norm(h_vec)
            if h_norm > 0:
                inc_current = np.rad2deg(np.arccos(h_vec[2] / h_norm))
                inc_ok = (
                    abs(inc_current - self.target_inclination) <= 0.1
                )  # 0.1 deg tolerance
        return apo_ok and peri_ok and inc_ok

    def get_setpoints(
        self, time: float, state_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        # Compute orbital elements
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        if self.elements is None:
            self.elements = compute_orbital_elements(position, velocity, self.mu)

        # Compute time to propellant depletion
        g0 = 9.80665
        thrust = self.vehicle.base_thrust_magnitude
        isp = self.vehicle.average_isp
        prop_mass = state_vector[13]
        v_e = isp * g0
        mdot = (thrust * self.throttle) / v_e if v_e > 0 else 1e-6
        time_to_prop_deplete = prop_mass / mdot if mdot > 0 and prop_mass > 0 else 1e6

        # Compute velocity vectors
        position_magnitude = np.linalg.norm(position)
        radial_unit = position / position_magnitude
        horizontal_projection = np.cross(self.orbital_normal, radial_unit)
        horizontal_unit = horizontal_projection / np.linalg.norm(horizontal_projection)
        current_radial_vel = np.dot(velocity, radial_unit)
        current_tangential_vel = np.dot(velocity, horizontal_unit)

        # Compute required delta V (diff between current and desired)
        v_target = np.sqrt(
            self.mu * (2 / self.target_periapsis - 1 / self.target_sem_maj_axis)
        )
        required_delta_v = max(0.0, v_target - current_tangential_vel)

        # Compute gravity
        # TODO: this assumes constant gravity from this point to the end of burn
        g = self.mu / position_magnitude**2
        centrifugal = (
            current_tangential_vel**2 / position_magnitude
            if position_magnitude > 0
            else 0.0
        )
        g_net = centrifugal - g

        # TODO: this PEG implementation seems to be unstable at small delta V. For now use prograde at
        #  the end of the burn
        if required_delta_v <= 500.0:
            logging.warning(
                f"PEG MODE - REQUIRED DELTA V SMALL: {required_delta_v}. DEFAULT TO CURRENT ATTITUDE"
            )
            self.attitude_mode = "prograde"
            return state_vector[6:10], self.throttle

        # Initial guess at time to go from rocket equation
        if required_delta_v > 0:
            exp_term = np.exp(-required_delta_v / v_e)
            time_to_go = time_to_prop_deplete * (1 - exp_term)
        else:
            time_to_go = 0.1
        # Clip to reasonable values
        time_to_go = np.clip(time_to_go, 0.1, time_to_prop_deplete * 0.99)

        # Refinement loop
        # Convergence tolerance in m/s
        tol = 10
        # Damping factor for corrections
        damping = 0.5
        for _ in range(100):
            burn_fraction = time_to_go / time_to_prop_deplete
            log_term = max(1e-10, 1 - burn_fraction)
            # 0th order thrust integral (m/s)
            b0 = -v_e * np.log(log_term)
            # 1st order thurst integral (m)
            b1 = time_to_prop_deplete * b0 - v_e * time_to_go
            # 2nd order thurst integral (m*s)
            b2 = (
                time_to_prop_deplete**2 * b0
                - v_e * time_to_prop_deplete * time_to_go
                - 0.5 * v_e * time_to_go**2
            )
            # 0th order position integral (m)
            c0 = b0 * time_to_go - b1
            # 1st order position integral (m*s)
            c1 = b1 * time_to_go - b2
            # Compute remaining radial position to go
            rhs_dot = -current_radial_vel - g_net * time_to_go
            rhs_r = (
                self.target_periapsis
                - position_magnitude
                - current_radial_vel * time_to_go
                - 0.5 * g_net * time_to_go**2
            )

            # Coefficient matrix
            A_matrix = np.array([[b0, b1], [c0, c1]])
            # Check for singular or ill conditioned
            det = np.linalg.det(A_matrix)
            if abs(det) < 1e-6 or np.isnan(det):
                print(f"PEG Convergence Failed at t={time}. Default Horizontal")
                self.attitude_mode = "horizontal"
                return compute_body_z_to_inertial_quat(horizontal_unit), self.throttle

            # Solve
            solution = np.linalg.solve(A_matrix, [rhs_dot, rhs_r])
            A, B = solution

            # Compute achievable tangential delta v with these steering params and find error with desired delta v
            integral_fr2 = A**2 * b0 + 2 * A * B * b1 + B**2 * b2
            predicted = b0 - 0.5 * integral_fr2
            error = predicted - required_delta_v

            # If within tolerance, done
            if abs(error) < tol:
                break

            # Otherwise update time to go with Newton step (smaller if error is positive, larger if negative)
            der = max(1e-3, b0 / time_to_go)
            time_to_go -= damping * error / der
            time_to_go = np.clip(time_to_go, 0.1, time_to_prop_deplete * 0.99)

        # Radial component of desired thrust vector
        # TODO: currently assuming the constant term alone [A] is a good approximation
        f_r = A
        f_r = np.clip(f_r, -1.0, 1.0)

        # Normal component of the desired thrust vector
        f_n = 0.0
        # No inclination correction unless user specifies
        # TODO: this assumes the out of plane steering is small enough to be negligible when computing required
        #  tangential term
        if self.target_inclination is not None:
            # Angular momentum
            h_vec = np.cross(position, velocity)
            h_norm = np.linalg.norm(h_vec)
            if h_norm > 0:
                # Current inclination
                current_i = np.arccos(h_vec[2] / h_norm)
                # Difference between current and desired
                delta_i = np.deg2rad(self.target_inclination) - current_i
                # Proportional control
                k_out = 0.05
                f_n = k_out * delta_i
                f_n = np.clip(f_n, -0.3, 0.3)

        # Tangential component of desired thurst vector
        f_theta = np.sqrt(max(0.0, 1 - f_r**2 - f_n**2))

        # Construct desired thrust vector from three components
        desired_z_vector = (
            f_r * radial_unit + f_theta * horizontal_unit + f_n * self.orbital_normal
        )
        desired_z_vector /= np.linalg.norm(desired_z_vector)
        desired_quaternion = compute_body_z_to_inertial_quat(desired_z_vector)

        # Terminal stage throttling to prevent overshoot
        apo_error = abs(self.target_apoapsis - self.elements["apoapsis_radius"])
        peri_error = abs(self.target_periapsis - self.elements["periapsis_radius"])
        # Find largest error
        max_error = max(apo_error, peri_error)
        # Compute threshold for when to start throttling
        throttle_threshold = self.throttle_threshold_factor * max(
            self.apo_tolerance, self.peri_tolerance
        )
        if max_error < throttle_threshold:
            # Apply proportional control to normalized error
            normalized_error = max_error / self.target_apoapsis
            self.throttle = max(self.min_throttle, self.throttle_kp * normalized_error)

        return desired_quaternion, self.throttle


class CoastGuidancePhase(GuidancePhase):
    """
    Coast phase that ends when close to apoapsis (or when circularization burn should start).

    Args:
        vehicle: Vehicle instance
        environment: Environment instance
        time_to_apo_threshold: Time (s) remaining to apoapsis to trigger end
        attitude_mode: Desired attitude during coast
        name: Name of the phase
        buffer: Extra time margin for burn preparation (s)
        use_dynamic_threshold: Whether to estimate burn time dynamically
    """

    def __init__(
        self,
        vehicle: Vehicle,
        environment: Environment,
        time_to_apo_threshold: float = 30.0,
        attitude_mode: str = "prograde",
        name: str = "Coast",
        buffer: float = 5.0,
        use_dynamic_threshold: bool = True,
    ):
        self.vehicle = vehicle
        self.time_to_apo_threshold = time_to_apo_threshold
        self.attitude_mode = attitude_mode
        self.name = name
        self.buffer = buffer
        self.use_dynamic_threshold = use_dynamic_threshold
        self.elements = None
        self.mu = environment.gravitational_constant * environment.earth_mass

    def is_complete(self, time: float, state_vector: np.ndarray) -> bool:

        position = state_vector[:3]
        velocity = state_vector[3:6]
        self.elements = compute_orbital_elements(position, velocity, self.mu)

        if (
            not self.use_dynamic_threshold
            or self.elements["apoapsis_radius"] == float("inf")
            or self.elements["semi_major_axis"] <= 0
        ):
            time_to_apo = compute_time_to_apoapsis(
                position, velocity, self.elements, self.mu
            )
            return time_to_apo <= self.time_to_apo_threshold

        # Dynamic computation
        r_apo = self.elements["apoapsis_radius"]
        a = self.elements["semi_major_axis"]
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

        time_to_apo = compute_time_to_apoapsis(
            position, velocity, self.elements, self.mu
        )
        return time_to_apo <= half_burn

    def get_setpoints(
        self, time: float, state_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        desired_quaternion = quaternion_from_attitude_mode(
            state_vector, self.attitude_mode
        )
        throttle = 0.0
        return desired_quaternion, throttle


class CircBurnGuidancePhase(GuidancePhase):
    """
    Circularization burn phase — raises periapsis to target altitude.

    Args:
        environment: Environment instance
        attitude_mode: Attitude mode to hold while burning
        throttle: Maximum throttle level
        name: Name of the phase
        min_throttle: Minimum throttle during fine control
        throttle_kp: Proportional gain for throttle adjustment
        target_eccentricity: Desired final eccentricity
    """

    def __init__(
        self,
        environment: Environment,
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
        self.elements = None
        self.mu = environment.gravitational_constant * environment.earth_mass

    def is_complete(self, time: float, state_vector: np.ndarray) -> bool:
        position = state_vector[:3]
        velocity = state_vector[3:6]
        self.elements = compute_orbital_elements(position, velocity, self.mu)

        # Check for escape (missed cut-off)
        if self.elements["apoapsis_radius"] == float("inf"):
            return True

        # Eccentricity check
        ecc_ok = self.elements["eccentricity"] < self.target_eccentricity

        return ecc_ok

    def get_setpoints(
        self, time: float, state_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        desired_quaternion = quaternion_from_attitude_mode(
            state_vector, self.attitude_mode
        )
        if self.elements is None:
            position = state_vector[:3]
            velocity = state_vector[3:6]
            self.elements = compute_orbital_elements(position, velocity, self.mu)
        if (
            self.elements["apoapsis_radius"] == float("inf")
            or self.elements["periapsis_radius"] <= 0
        ):
            # Safety: Fallback for hyperbolic or invalid orbits
            throttle = 0.0
        else:
            ratio = self.elements["periapsis_radius"] / self.elements["apoapsis_radius"]
            error = max(0.0, 1.0 - ratio)  # Positive error
            throttle = max(self.min_throttle, min(1.0, self.throttle_kp * error))

        return desired_quaternion, throttle
