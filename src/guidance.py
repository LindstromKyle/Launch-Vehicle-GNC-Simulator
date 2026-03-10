import logging
import numpy as np

from abc import ABC, abstractmethod

from utils import (
    compute_body_z_to_inertial_quat,
    quaternion_from_attitude_mode,
    compute_time_to_apoapsis,
    compute_orbital_elements,
)
from vehicle import Vehicle


class GuidancePhase(ABC):
    """
    Abstract base class for guidance guidance_phases. Each phase defines its attitude/throttle setpoints and when it should end.
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
    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        """
        Compute the desired quaternion based on time and current state.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector [position, velocity, quaternion, ang_vel, prop_mass]

        Returns:
            Desired quaternion [w, x, y, z] (normalized)
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
            Dictionary of setpoints (attitude_mode, throttle)
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

    def __init__(self, end_time: float, attitude_mode: str, throttle: float = 1.0, name: str = "Unnamed"):
        self.end_time = end_time
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        return quaternion_from_attitude_mode(state_vector, self.attitude_mode)


class CoastGuidancePhase(GuidancePhase):
    """
    Coast phase that ends when close to apoapsis (or when circularization burn should start).

    Args:
        vehicle: Vehicle instance
        time_to_apo_threshold: Time (s) remaining to apoapsis to trigger end
        attitude_mode: Desired attitude during coast
        name: Name of the phase
        buffer: Extra time margin for burn preparation (s)
        use_dynamic_threshold: Whether to estimate burn time dynamically
    """

    def __init__(
        self,
        vehicle: Vehicle,
        mu: float,
        time_to_apo_threshold: float = 30.0,
        attitude_mode: str = "prograde",
        name: str = "Coast",
        buffer: float = 5.0,
        use_dynamic_threshold: bool = True,
    ):
        self.vehicle = vehicle
        self.mu = mu
        self.time_to_apo_threshold = time_to_apo_threshold
        self.attitude_mode = attitude_mode
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
        return {"throttle": 0.0, "attitude_mode": self.attitude_mode}

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        return quaternion_from_attitude_mode(state_vector, self.attitude_mode)


class CircBurnGuidancePhase(GuidancePhase):
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

        return {"throttle": throttle, "attitude_mode": self.attitude_mode}

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        return quaternion_from_attitude_mode(state_vector, self.attitude_mode)


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

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:

        position = state_vector[:3]
        velocity = state_vector[3:6]
        radial_unit_vector = position / np.linalg.norm(position)
        velocity_magnitude = np.linalg.norm(velocity)

        # Normalize time progress (0 to 1)
        progress = max(0.0, min(1.0, (time - self.start_time) / self.duration))

        # Interpolate pitch angle (from horizontal)
        pitch_deg = self.initial_pitch_deg + progress * (self.final_pitch_deg - self.initial_pitch_deg)
        pitch_rad = np.deg2rad(pitch_deg)

        if velocity_magnitude < 1e-3:
            desired_z_vector = radial_unit_vector  # Fallback to radial if low speed
            self.current_attitude_mode = "radial"

        else:
            cross_product = np.cross(self.orbital_normal, radial_unit_vector)
            norm_cross = np.linalg.norm(cross_product)
            if norm_cross > 1e-6:
                horizontal_unit_vector = cross_product / norm_cross
            else:
                # Fallback if zero
                horizontal_projection = (
                    self.kick_direction - np.dot(self.kick_direction, radial_unit_vector) * radial_unit_vector
                )
                horizontal_unit_vector = horizontal_projection / np.linalg.norm(horizontal_projection)

            # Desired z: sin(pitch) vertical (radial) + cos(pitch) horizontal
            desired_z_vector = np.cos(pitch_rad) * horizontal_unit_vector + np.sin(pitch_rad) * radial_unit_vector
            desired_z_vector /= np.linalg.norm(desired_z_vector)

        return compute_body_z_to_inertial_quat(desired_z_vector)


class PEGGuidancePhase(GuidancePhase):
    """
    Powered Explicit Guidance phase. Uses iterative PEG algorithm to steer toward specified apoapsis/periapsis.

    Args:
        target_apoapsis: Desired apoapsis radius (m)
        target_periapsis: Desired periapsis radius (m)
        orbital_normal: The vector describing the direction normal to the orbital plane
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
        target_periapsis: float,
        orbital_normal: np.ndarray,
        vehicle: Vehicle,
        mu: float,
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
        self.mu = mu
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

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        position = state_vector[:3]
        velocity = state_vector[3:6]
        elements = compute_orbital_elements(position, velocity, self.mu)
        throttle = self.get_setpoints(time, state_vector, elements).get("throttle")
        mu = self.mu
        g0 = 9.80665
        target_r = self.target_periapsis
        target_a = self.target_a
        target_inclination = self.target_inclination
        thrust = self.vehicle.base_thrust_magnitude
        isp = self.vehicle.average_isp
        dry_mass = self.vehicle.dry_mass

        prop_mass = state_vector[13]
        m = dry_mass + prop_mass
        v_e = isp * g0
        mdot = (thrust * throttle) / v_e if v_e > 0 else 1e-6
        tau = m / mdot if mdot > 0 else 1e6
        r = np.linalg.norm(position)
        radial_unit = position / r
        horizontal_projection = np.cross(self.orbital_normal, radial_unit)
        horizontal_unit = horizontal_projection / np.linalg.norm(horizontal_projection)
        dr0 = np.dot(velocity, radial_unit)
        v_theta0 = np.dot(velocity, horizontal_unit)
        v_target = np.sqrt(mu * (2 / target_r - 1 / target_a))
        required_delta_v = max(0.0, v_target - v_theta0)
        g = mu / r**2
        centrifugal = v_theta0**2 / r if r > 0 else 0.0
        g_net = centrifugal - g

        if required_delta_v <= 75.0:
            logging.warning(f"PEG MODE - REQUIRED DELTA V SMALL: {required_delta_v}. DEFAULT TO CURRENT ATTITUDE")
            self.current_attitude_mode = "prograde"
            return state_vector[6:10]

        if required_delta_v > 0:
            exp_term = np.exp(-required_delta_v / v_e)
            T = tau * (1 - exp_term)
        else:
            T = 0.1
        T = np.clip(T, 0.1, tau * 0.99)
        tol = 10
        damping = 0.5
        for _ in range(100):
            u = T / tau
            if u >= 1:
                u = 0.99
                T = tau * u
            log_term = max(1e-10, 1 - u)
            b0 = -v_e * np.log(log_term)
            b1 = tau * b0 - v_e * T
            b2 = tau**2 * b0 - v_e * tau * T - 0.5 * v_e * T**2
            c0 = b0 * T - b1
            c1 = b1 * T - b2
            rhs_dot = -dr0 - g_net * T
            rhs_r = target_r - r - dr0 * T - 0.5 * g_net * T**2
            A_mat = np.array([[b0, b1], [c0, c1]])
            det = np.linalg.det(A_mat)
            if abs(det) < 1e-6 or np.isnan(det):
                print(f"PEG Convergence Failed at t={time}. Default Horizontal")
                self.current_attitude_mode = "horizontal"
                return compute_body_z_to_inertial_quat(horizontal_unit)

            sol = np.linalg.solve(A_mat, [rhs_dot, rhs_r])
            A, B = sol
            integral_fr2 = A**2 * b0 + 2 * A * B * b1 + B**2 * b2
            predicted = b0 - 0.5 * integral_fr2
            error = predicted - required_delta_v
            if abs(error) < tol:
                break
            der = max(1e-3, b0 / T)
            T -= damping * error / der
            T = np.clip(T, 0.1, tau * 0.99)

        f_r = A
        self.prev_f_r = f_r

        t_thresh = 2
        if T <= t_thresh:
            f_r = self.prev_f_r * (T / t_thresh)

        f_r = np.clip(f_r, -1.0, 1.0)
        f_n = 0.0

        if target_inclination is not None:
            h_vec = np.cross(position, velocity)
            h_norm = np.linalg.norm(h_vec)
            if h_norm > 0:
                current_i = np.arccos(h_vec[2] / h_norm)
                delta_i = np.deg2rad(target_inclination) - current_i
                k_out = 0.05
                f_n = k_out * delta_i
                f_n = np.clip(f_n, -0.3, 0.3)

        f_theta = np.sqrt(max(0.0, 1 - f_r**2 - f_n**2))
        desired_z_vector = f_r * radial_unit + f_theta * horizontal_unit + f_n * self.orbital_normal

        # PEG smoothing
        alpha = 0.995
        if not hasattr(self, "last_desired_z"):
            self.last_desired_z = desired_z_vector.copy()
        desired_z_vector = alpha * desired_z_vector + (1 - alpha) * self.last_desired_z
        self.last_desired_z = desired_z_vector.copy()

        desired_z_vector /= np.linalg.norm(desired_z_vector)
        return compute_body_z_to_inertial_quat(desired_z_vector)
