import logging
import numpy as np
import cvxpy as cp

from abc import ABC, abstractmethod

from state import State
from utils import compute_minimal_quaternion_rotation, compute_orbital_elements
from environment import Environment


class Guidance(ABC):
    """
    Base class for guidance systems that compute desired attitude over time. Subclasses must implement
    get_desired_quaternion().
    """

    @abstractmethod
    def get_desired_quaternion(
        self, time: float, state_vector: np.ndarray, mission_phase_parameters: dict
    ) -> np.ndarray:
        """
        Compute the desired quaternion based on time and current state.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector [position, velocity, quaternion, ang_vel, prop_mass]
            mission_phase_parameters: Dictionary of current mission phase setpoints

        Returns:
            Desired quaternion [w, x, y, z] (normalized)
        """
        pass


class ModeBasedGuidance(Guidance):
    """
    Guidance system that selects attitude mode based on mission phase setpoints.

    Args:
        orbital_normal: Unit vector normal to the orbital plane (used for horizontal reference)
        environment: Environment instance
    """

    def __init__(self, orbital_normal: np.ndarray, environment: Environment):
        self.orbital_normal = orbital_normal
        self.environment = environment
        self.current_attitude_mode = None
        self.prev_f_r = 0.0

    def get_desired_quaternion(
        self, time: float, state_vector: np.ndarray, mission_planner_setpoints: dict
    ) -> np.ndarray:
        """
        Compute desired quaternion according to current attitude mode.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector [pos, vel, quat, ang_vel, prop_mass]
            mission_planner_setpoints: Current phase setpoints from mission planner

        Returns:
            Desired quaternion [w, x, y, z] (normalized)
        """
        position = state_vector[:3]
        velocity = state_vector[3:6]
        radial_unit_vector = position / np.linalg.norm(position)
        velocity_magnitude = np.linalg.norm(velocity)

        mode = mission_planner_setpoints.get("attitude_mode")
        self.current_attitude_mode = mode

        if mode == "radial":
            desired_z_vector = radial_unit_vector
        elif mode == "kick":
            kick_direction = mission_planner_setpoints.get("kick_direction")
            kick_angle_rad = mission_planner_setpoints.get("kick_angle_rad")
            horizontal_projection = kick_direction - np.dot(kick_direction, radial_unit_vector) * radial_unit_vector
            horizontal_unit_vector = horizontal_projection / np.linalg.norm(horizontal_projection)
            desired_z_vector = (
                np.cos(kick_angle_rad) * radial_unit_vector + np.sin(kick_angle_rad) * horizontal_unit_vector
            )
            desired_z_vector /= np.linalg.norm(desired_z_vector)
        elif mode == "prograde":
            if velocity_magnitude < 1e-3:
                desired_z_vector = radial_unit_vector
                self.current_attitude_mode = "radial"
            else:
                desired_z_vector = velocity / velocity_magnitude
        elif mode == "retrograde":
            if velocity_magnitude < 1e-3:
                desired_z_vector = -radial_unit_vector
                self.current_attitude_mode = "radial_down"
            else:
                desired_z_vector = -velocity / velocity_magnitude
        elif mode == "radial_down":
            desired_z_vector = -radial_unit_vector
        elif mode == "programmed_pitch":
            # Extract parameters from mission_planner_setpoints
            start_time = mission_planner_setpoints.get("start_time", 0.0)  # Phase start time
            duration = mission_planner_setpoints.get("duration", 100.0)  # Phase duration for interpolation
            initial_pitch_deg = mission_planner_setpoints.get("initial_pitch_deg", 80.0)
            final_pitch_deg = mission_planner_setpoints.get("final_pitch_deg", 45.0)
            kick_direction = mission_planner_setpoints.get("kick_direction", np.array([0.0, 1.0, 0.0]))  # Default east

            # Normalize time progress (0 to 1)
            progress = max(0.0, min(1.0, (time - start_time) / duration))

            # Interpolate pitch angle (from horizontal)
            pitch_deg = initial_pitch_deg + progress * (final_pitch_deg - initial_pitch_deg)
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
                        kick_direction - np.dot(kick_direction, radial_unit_vector) * radial_unit_vector
                    )
                    horizontal_unit_vector = horizontal_projection / np.linalg.norm(horizontal_projection)

                # Desired z: sin(pitch) vertical (radial) + cos(pitch) horizontal
                desired_z_vector = np.cos(pitch_rad) * horizontal_unit_vector + np.sin(pitch_rad) * radial_unit_vector
                desired_z_vector /= np.linalg.norm(desired_z_vector)

        elif mode == "pitch_to_apoapsis":
            # Extract parameters from mission_planner_setpoints
            start_time = mission_planner_setpoints.get("start_time", 0.0)
            initial_pitch_deg = mission_planner_setpoints.get("initial_pitch_deg", 90.0)
            base_pitch_rate_deg_per_sec = mission_planner_setpoints.get("base_pitch_rate_deg_per_sec", 0.4)
            min_pitch_deg = mission_planner_setpoints.get("min_pitch_deg", 20.0)
            kp_apo = mission_planner_setpoints.get("kp_apo", 1.0)
            kp_vel = mission_planner_setpoints.get("kp_vel", 0.8)
            vel_weight = mission_planner_setpoints.get("vel_weight", 0.5)
            vel_threshold_factor = mission_planner_setpoints.get("vel_threshold_factor", 0.95)
            target_apoapsis = mission_planner_setpoints.get("target_apoapsis")

            # Compute elements
            position = state_vector[:3]
            velocity = state_vector[3:6]
            mu = self.environment.gravitational_constant * self.environment.earth_mass
            elements = compute_orbital_elements(position, velocity, mu)  # Assuming mu available

            current_apo = elements["apoapsis_radius"]
            apo_error = (target_apoapsis - current_apo) / target_apoapsis  # Positive if lagging apo

            # Velocity error
            a = elements["semi_major_axis"]
            r_apo = elements["apoapsis_radius"]
            if a == float("inf") or r_apo == float("inf"):
                vel_error = 0.0  # Fallback - early burn
            else:
                v_apo_projected = np.sqrt(mu * (2 / r_apo - 1 / a)) if a > 0 else 0.0
                v_circ = np.sqrt(mu / r_apo)
                vel_error = (v_circ * vel_threshold_factor - v_apo_projected) / (
                    v_circ * vel_threshold_factor
                )  # Positive if lagging velocity

            # Dynamic pitch rate
            dynamic_pitch_rate = base_pitch_rate_deg_per_sec * (1 + kp_apo * apo_error + kp_vel * vel_error)

            # Clamp rate to prevent extremes
            dynamic_pitch_rate = np.clip(dynamic_pitch_rate, 0.1, 1.0)  # e.g., min 0.1 deg/s, max 1.0

            # Elapsed and pitch calc
            elapsed = max(0.0, time - start_time)
            pitch_deg = max(min_pitch_deg, initial_pitch_deg - dynamic_pitch_rate * elapsed)
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
                    horizontal_unit_vector = np.array([0.0, 1.0, 0.0])  # Fallback eastward

                # Desired z: cos(pitch) horizontal + sin(pitch) vertical
                desired_z_vector = np.cos(pitch_rad) * horizontal_unit_vector + np.sin(pitch_rad) * radial_unit_vector
                desired_z_vector /= np.linalg.norm(desired_z_vector)

        elif mode == "peg":

            mu = mission_planner_setpoints["mu"]
            g0 = mission_planner_setpoints["g0"]
            target_r = mission_planner_setpoints["target_r"]
            target_a = mission_planner_setpoints["target_a"]
            target_inclination = mission_planner_setpoints.get("target_inclination")
            thrust = mission_planner_setpoints["thrust"]
            isp = mission_planner_setpoints["isp"]
            dry_mass = mission_planner_setpoints["dry_mass"]
            throttle = mission_planner_setpoints.get("throttle", 1.0)

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
                    return compute_minimal_quaternion_rotation(horizontal_unit)

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

        elif mode == "passive":
            # Set desired to current quaternion (no control needed)
            desired_quaternion = state_vector[6:10].copy()
            desired_quaternion /= np.linalg.norm(desired_quaternion)  # Normalize for safety
            return desired_quaternion

        else:
            raise ValueError(f"Unknown attitude mode: {mode}")

        return compute_minimal_quaternion_rotation(desired_z_vector)
