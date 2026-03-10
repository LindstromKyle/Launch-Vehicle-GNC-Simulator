import logging
import numpy as np

from scipy.optimize import nnls
from abc import ABC, abstractmethod

from utils import quaternion_multiply, quaternion_inverse, quat_to_angle_axis, rotate_body_to_inertial_by_quat
from vehicle import Vehicle


class Controller(ABC):
    """
    Abstract base class for all attitude and thrust controllers. Subclasses must implement the update() method to
    compute control inputs.
    """

    @abstractmethod
    def update(
        self,
        time: float,
        state_vector: np.ndarray,
        desired_quaternion: np.ndarray,
        throttle: float,
        log_flag: bool,
        attitude_mode: str,
    ) -> dict:
        """
        Compute control inputs for the current time step.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector [pos, vel, quat, ang_vel, prop_mass]
            desired_quaternion: Current desired quaternion from guidance
            throttle: Throttle modulation value
            log_flag: Whether to log information this step
            attitude_mode: Attitude mode from guidance (for logging only)

        Returns:
            Dictionary containing control commands (gimbal angles, RCS levels)
        """
        pass


class PIDAttitudeController(Controller):
    """
    PID-based attitude controller that computes gimbal angles and RCS commands. Uses quaternion error and PID control
    to generate desired body torque, then allocates it to engine gimbals and RCS thrusters.

    Args:
        kp: Proportional gain vector [x, y, z] (N·m/rad)
        ki: Integral gain vector [x, y, z] (N·m/rad·s)
        kd: Derivative gain vector [x, y, z] (N·m/(rad/s))
        vehicle: Vehicle instance
    """

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        vehicle: Vehicle,
    ):
        # Copy PID gains to allow modification
        self.kp = kp.copy()
        self.ki = ki.copy()
        self.kd = kd.copy()
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.previous_d_term = np.zeros(3)
        self.last_update_time = None
        self.prev_error_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle = vehicle

    def update(
        self,
        time: float,
        state_vector: np.ndarray,
        desired_quaternion: np.ndarray,
        throttle: float,
        log_flag: bool,
        attitude_mode: str,
    ) -> dict:
        """
        Compute control inputs for the current time step.

        Args:
            time: Current simulation time (seconds)
            state_vector: Full state vector [pos, vel, quat, ang_vel, prop_mass]
            desired_quaternion: Current desired quaternion from guidance
            throttle: Throttle modulation value
            log_flag: Whether to log information this step
            attitude_mode: Attitude mode from guidance (for logging only)

        Returns:
            Dictionary containing control commands (gimbal angles, RCS levels)
        """
        current_propellant_mass = state_vector[13]
        current_quaternion = state_vector[6:10]
        current_quaternion /= np.linalg.norm(current_quaternion)

        # Check if the guidance is in passive mode
        if not np.all(desired_quaternion == current_quaternion):
            # Compute quaternion error
            error_quaternion = quaternion_multiply(desired_quaternion, quaternion_inverse(current_quaternion))
            error_quaternion /= np.linalg.norm(error_quaternion)

            # Enforce the shortest rotation
            if np.dot(error_quaternion, self.prev_error_quat) < 0:
                error_quaternion = -error_quaternion
            previous_angle_axis = quat_to_angle_axis(self.prev_error_quat)
            self.prev_error_quat = error_quaternion

            # Convert to angle-axis for PID
            angle_axis = quat_to_angle_axis(error_quaternion)
            current_error = angle_axis[0] * angle_axis[1:]

            # Basic gain scheduling
            current_mass = self.vehicle.dry_mass + current_propellant_mass
            mass_ratio = current_mass / (self.vehicle.dry_mass + self.vehicle.initial_propellant_mass)
            kp_scheduled = self.kp.copy() * mass_ratio
            kd_scheduled = self.kd.copy() * mass_ratio

            # Deadband RCS if coasting to hold prograde without continuously firing
            if throttle <= 0.01:
                prev_error_angle = np.rad2deg(previous_angle_axis[0])
                error_angle = np.rad2deg(angle_axis[0])
                hysteresis_inner_flag = (prev_error_angle > error_angle) and (error_angle < 0.1)
                hysteresis_outer_flag = (prev_error_angle < error_angle) and (error_angle < 3.5)
                if hysteresis_inner_flag or hysteresis_outer_flag:
                    kp_scheduled = 0
                    kd_scheduled = 0

            # Compute PID terms
            p_term = kp_scheduled * current_error
            d_term = np.zeros(3)
            if self.last_update_time:
                dt = time - self.last_update_time
                self.integral_error += current_error * dt
                d_term = kd_scheduled * (current_error - self.previous_error) / dt
            i_term = self.ki * self.integral_error

            # Low pass filter on d term
            alpha = 0.3
            d_term = alpha * d_term + (1 - alpha) * self.previous_d_term
            self.previous_d_term = d_term

            # Update stored variables
            self.last_update_time = time
            self.previous_error = current_error

            # Compute unsaturated torque
            unsaturated_torque = p_term + i_term + d_term

            # Map torque to actuators
            effective_thrust_magnitude = self.vehicle.get_thrust_magnitude(throttle)
            gimbal_arm = self.vehicle.get_gimbal_arm(current_propellant_mass)
            thrust_per_engine = effective_thrust_magnitude / self.vehicle.num_engines
            gimbal_angles_list, rcs_levels, achieved_torque = self.get_actuator_commands(
                unsaturated_torque, thrust_per_engine, gimbal_arm
            )

            # Anti-windup - limit integral error based on saturation (if achieved_torque != unsaturated_torque)
            mask = self.ki != 0
            self.integral_error[mask] = (achieved_torque[mask] - p_term[mask] - d_term[mask]) / self.ki[mask]

        else:
            # Everything is passive
            gimbal_angles_list = []
            for engine in range(len(self.vehicle.engines)):
                gimbal_angles_list.append([0, 0])
            unsaturated_torque = np.zeros(3)
            throttle = 0.0
            rcs_levels = np.zeros(len(self.vehicle.rcs_thrusters))
            desired_quaternion = current_quaternion
            error_quaternion = np.array([1, 0, 0, 0])
            angle_axis = np.zeros(4)
            current_error = np.zeros(3)
            p_term = np.zeros(3)
            i_term = np.zeros(3)
            d_term = np.zeros(3)

        if log_flag:
            current_z_unit_vector = rotate_body_to_inertial_by_quat(np.array([0, 0, 1]), current_quaternion)
            desired_z_unit_vector = rotate_body_to_inertial_by_quat(np.array([0, 0, 1]), desired_quaternion)
            attitude_error = desired_z_unit_vector - current_z_unit_vector
            position = state_vector[:3]
            radial_unit_vector = position / np.linalg.norm(position)
            current_dot = np.dot(current_z_unit_vector, radial_unit_vector)
            desired_dot = np.dot(desired_z_unit_vector, radial_unit_vector)
            current_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(current_dot, -1.0, 1.0)))
            desired_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(desired_dot, -1.0, 1.0)))
            logging.info(f"------------------------------------[GUIDANCE]--------------------------------------------")
            logging.info(f"attitude mode: {attitude_mode}")
            logging.info(
                f"current quat: {np.round(current_quaternion, 4)} | current attitude (z_hat): {np.round(current_z_unit_vector, 4)}"
            )
            logging.info(
                f"desired quat: {np.round(desired_quaternion, 4)} | desired attitude (z_hat): {np.round(desired_z_unit_vector, 4)}"
            )
            logging.info(
                f"error quat: {np.round(error_quaternion, 4)} | error attitude (z_hat): {np.round(attitude_error, 4)}"
            )
            logging.info(f"current pitch (deg): {current_pitch:.2f}")
            logging.info(f"desired pitch (deg): {desired_pitch:.2f}")
            logging.info(
                f"pitch error (deg): {(desired_pitch - current_pitch):.2f} | quat error angle (deg): {np.round(np.rad2deg(angle_axis[0]), 4)}"
            )
            logging.info(f"-----------------------------------[CONTROLLER]-------------------------------------------")
            logging.info(f"body frame error (deg): {np.round(np.rad2deg(current_error), 4)}")
            logging.info(
                f"PID p term: {np.round(p_term, 4)} | PID i term: {np.round(i_term, 4)} | PID d term: {np.round(d_term, 4)}"
            )
            logging.info(f"desired torque (N*m): {np.round(unsaturated_torque, 4)} | throttle: {throttle:.4f}")
        return {
            "engine_gimbal_angles": gimbal_angles_list,
            "rcs_levels": rcs_levels,
        }

    def get_actuator_commands(
        self, desired_torque: np.ndarray, effective_thrust_e: float, gimbal_arm: float
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Allocate desired torque to engine gimbals and RCS thrusters.

        Args:
            desired_torque: Desired control torque in body frame (N·m)
            effective_thrust_e: Effective thrust level for torque calculation (N)
            gimbal_arm: Current lever arm from CoM to engine plane (m)

        Returns:
            Tuple containing:
                - list of [pitch, yaw] gimbal angles (rad) for each engine
                - array of RCS throttle levels (0 to 1) for each RCS thruster
                - the actual achieved torque from engines + RCS
        """
        sin_lim = np.sin(self.vehicle.engine_gimbal_limit_rad)
        num_engines = self.vehicle.num_engines
        num_gimbals = 2 * num_engines
        A = np.zeros((3, num_gimbals))

        for i in range(num_engines):
            pos = self.vehicle.engines[i]["position"].copy()
            pos[2] = -gimbal_arm

            # Pitch
            delta_f_pitch = np.array([0.0, -effective_thrust_e, 0.0])
            torque_pitch = np.cross(pos, delta_f_pitch)
            A[:, 2 * i] = torque_pitch

            # Yaw
            delta_f_yaw = np.array([effective_thrust_e, 0.0, 0.0])
            torque_yaw = np.cross(pos, delta_f_yaw)
            A[:, 2 * i + 1] = torque_yaw

        # Solve least-squares for gimbal commands
        u, _, _, _ = np.linalg.lstsq(A, desired_torque, rcond=None)
        u_clipped = np.clip(u, -sin_lim, sin_lim)

        # Convert to angles
        gimbal_angles_list = []
        for i in range(num_engines):
            sin_pitch = u_clipped[2 * i]
            sin_yaw = u_clipped[2 * i + 1]
            pitch = np.arcsin(sin_pitch)
            yaw = np.arcsin(sin_yaw)
            gimbal_angles_list.append([pitch, yaw])

        achieved_torque = np.dot(A, u_clipped)

        # RCS allocation for residual torque
        if self.vehicle.rcs_thrusters:
            num_rcs = len(self.vehicle.rcs_thrusters)
            A_rcs = np.zeros((3, num_rcs))
            for j, rcs in enumerate(self.vehicle.rcs_thrusters):
                delta_f = rcs["max_thrust"] * rcs["direction"]
                torque = np.cross(rcs["position"], delta_f)
                A_rcs[:, j] = torque

            residual = desired_torque - achieved_torque
            residual = np.where(np.abs(residual) < 1e-12, 0.0, residual)

            rcs_levels, _ = nnls(A_rcs, residual)
            achieved_rcs_torque = A_rcs @ rcs_levels
            achieved_torque += achieved_rcs_torque

            if np.any(rcs_levels > 1):
                max_level = np.max(rcs_levels)
                rcs_levels *= 1.0 / max_level
        else:
            rcs_levels = np.array([])

        return gimbal_angles_list, rcs_levels, achieved_torque
