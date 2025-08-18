import logging
from scipy.optimize import nnls  # Add this import
import numpy as np
from abc import ABC, abstractmethod

from guidance import Guidance
from state import State
from utils import quaternion_multiply, quaternion_inverse, quat_to_angle_axis, rotate_vector_by_quaternion
from vehicle import Vehicle, Falcon9SecondStage


class Controller(ABC):
    """
    Base class for controllers. Subclasses implement control logic.
    """

    @abstractmethod
    def update(self, time: float, state: State, mission_planner_setpoints: dict, log_flag: bool) -> dict:
        """
        Compute control inputs based on time and current state.

        Args:
            log_flag ():
            mission_planner_setpoints ():
            time: Current simulation time (s)
            state: Current State object

        Returns:
            Dict with control outputs, e.g., {'gimbal_angles': np.array([pitch, yaw]), 'fin_deflections': dict(...)}
        """
        pass


class PIDAttitudeController(Controller):
    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        guidance: Guidance,
        vehicle: Vehicle,
    ):
        """

        Args:
            kp ():
            ki ():
            kd ():
            guidance ():
            vehicle ():
        """
        self.kp = kp
        self.ki = ki.copy()  # Allow modification
        self.kd = kd
        self.guidance = guidance
        self.integral_error = np.zeros(3)  # Accumulator for I term
        self.previous_error = np.zeros(3)
        self.previous_d_term = np.zeros(3)
        self.vehicle = vehicle
        self.last_update_time = None
        self.prev_error_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, time: float, state_vector: np.ndarray, mission_planner_setpoints: dict, log_flag: bool) -> dict:
        current_propellant_mass = state_vector[13]
        current_quaternion = state_vector[6:10]
        attitude_mode = mission_planner_setpoints.get("attitude_mode", "prograde")
        if attitude_mode != "passive":

            # Throttle
            throttle = mission_planner_setpoints.get("throttle", 1.0)

            # Get desired quaternion from guidance
            desired_quaternion = self.guidance.get_desired_quaternion(time, state_vector, mission_planner_setpoints)
            desired_quaternion /= np.linalg.norm(desired_quaternion)

            # Compute quaternion error (expressed in Body basis vectors)
            error_quaternion = quaternion_multiply(desired_quaternion, quaternion_inverse(current_quaternion))
            error_quaternion /= np.linalg.norm(error_quaternion)

            if np.dot(error_quaternion, self.prev_error_quat) < 0:
                error_quaternion = -error_quaternion
            self.prev_error_quat = error_quaternion

            # Convert to angle-axis for PID
            angle_axis = quat_to_angle_axis(error_quaternion)
            current_error = angle_axis[0] * angle_axis[1:]  # angle (rad) * Axis

            # Basic gain scheduling
            # Example: Scale kp/kd inversely with mass (higher control authority as mass drops)
            current_mass = self.vehicle.dry_mass + current_propellant_mass
            # TODO: think about this
            # mass_ratio = current_mass / self.vehicle.dry_mass  # >1 early, ~1 late
            # kp_scheduled = self.kp / mass_ratio  # Lower early, higher late
            # kd_scheduled = self.kd / mass_ratio**0.5  # Mild scaling
            kp_scheduled = self.kp
            kd_scheduled = self.kd

            if throttle <= 0.1 or attitude_mode == "prograde":  # Coast/low-thrust mode
                scale = 0.1  # Reduce gains; tune 0.05-0.2
                kp_scheduled *= scale
                kd_scheduled *= scale

            # Compute PID terms
            d_term = np.zeros(3)
            if self.last_update_time:
                dt = time - self.last_update_time
                if dt > 0:
                    self.integral_error += current_error * dt
                    d_term = kd_scheduled * (current_error - self.previous_error) / dt

            # Low pass filter on d term
            # TODO: put alpha in __init__?
            # alpha = 0.1
            # d_term = alpha * d_term + (1 - alpha) * self.previous_d_term
            # self.previous_d_term = d_term

            i_term = self.ki * self.integral_error
            p_term = kp_scheduled * current_error

            self.last_update_time = time
            self.previous_error = current_error

            # Compute unsaturated torque
            unsaturated_torque = p_term + i_term + d_term

            # Clip torque for saturation (anti-windup prep)
            control_torque = unsaturated_torque.copy()

            # Map torque to actuators
            thrust_magnitude = self.vehicle.get_thrust_magnitude(time)
            effective_thrust_magnitude = thrust_magnitude * throttle
            gimbal_arm = self.vehicle.get_gimbal_arm(current_propellant_mass)
            max_torque = effective_thrust_magnitude * np.sin(self.vehicle.engine_gimbal_limit_rad) * gimbal_arm
            # Clamp control torque to max for pitch and yaw
            control_torque[0] = np.clip(control_torque[0], -max_torque, max_torque)
            control_torque[1] = np.clip(control_torque[1], -max_torque, max_torque)

            # Anti-windup: Recalculate integral error based on saturated torque
            # Skip axes where ki=0 to avoid division by zero
            mask = self.ki != 0
            self.integral_error[mask] = (control_torque[mask] - p_term[mask] - d_term[mask]) / self.ki[mask]

            effective_thrust_e = effective_thrust_magnitude / self.vehicle.num_engines
            gimbal_angles_list, rcs_levels = self.get_actuator_commands(control_torque, effective_thrust_e, gimbal_arm)
        else:
            gimbal_angles_list = []
            for engine in range(len(self.vehicle.engines)):
                gimbal_angles_list.append([0, 0])
            control_torque = np.zeros(3)
            throttle = 0.0
            rcs_levels = np.zeros(len(self.vehicle.rcs_thrusters))
            desired_quaternion = current_quaternion
            error_quaternion = np.array([1, 0, 0, 0])
            angle_axis = np.zeros(4)
            current_error = np.zeros(3)
            p_term = np.zeros(3)
            i_term = np.zeros(3)
            d_term = np.zeros(3)
            self.guidance.current_attitude_mode = "passive"

        if log_flag:
            current_z_unit_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), current_quaternion)
            desired_z_unit_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), desired_quaternion)
            attitude_error = desired_z_unit_vector - current_z_unit_vector
            position = state_vector[:3]
            radial_unit_vector = position / np.linalg.norm(position)
            current_dot = np.dot(current_z_unit_vector, radial_unit_vector)
            desired_dot = np.dot(desired_z_unit_vector, radial_unit_vector)
            current_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(current_dot, -1.0, 1.0)))
            desired_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(desired_dot, -1.0, 1.0)))
            logging.info(f"------------------------------------[GUIDANCE]--------------------------------------------")
            logging.info(f"attitude mode: {self.guidance.current_attitude_mode}")
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
        return {
            "desired_torque": control_torque,
            "engine_gimbal_angles": gimbal_angles_list,
            "throttle": throttle,
            "propellant_mass": current_propellant_mass,  # Pass to dynamics
            "rcs_levels": rcs_levels,
        }

    def get_actuator_commands(
        self, desired_torque: np.ndarray, effective_thrust_e: float, gimbal_arm: float
    ) -> tuple[list, np.ndarray]:
        sin_lim = np.sin(self.vehicle.engine_gimbal_limit_rad)
        num_engines = self.vehicle.num_engines
        num_gimbal_dofs = 2 * num_engines
        A = np.zeros((3, num_gimbal_dofs))

        for i in range(num_engines):
            pos = self.vehicle.engines[i]["position"].copy()
            pos[2] = -gimbal_arm

            # Pitch dof (affects fy = -effective_thrust_e * sin_pitch)
            delta_f_pitch = np.array([0.0, -effective_thrust_e, 0.0])
            torque_pitch = np.cross(pos, delta_f_pitch)
            A[:, 2 * i] = torque_pitch

            # Yaw dof (affects fx = effective_thrust_e * sin_yaw)
            delta_f_yaw = np.array([effective_thrust_e, 0.0, 0.0])
            torque_yaw = np.cross(pos, delta_f_yaw)
            A[:, 2 * i + 1] = torque_yaw

        # Solve for u = [sin_pitch1, sin_yaw1, ..., sin_pitchN, sin_yawN]
        u, _, _, _ = np.linalg.lstsq(A, desired_torque, rcond=None)
        u_clipped = np.clip(u, -sin_lim, sin_lim)

        # Compute gimbal angles
        gimbal_angles_list = []
        for i in range(num_engines):
            sin_pitch = u_clipped[2 * i]
            sin_yaw = u_clipped[2 * i + 1]
            pitch = np.arcsin(sin_pitch)
            yaw = np.arcsin(sin_yaw)
            gimbal_angles_list.append([pitch, yaw])

        # Compute achieved torque from gimbals
        achieved_torque = np.dot(A, u_clipped)

        # RCS allocation if available
        if self.vehicle.rcs_thrusters:
            num_rcs = len(self.vehicle.rcs_thrusters)
            A_rcs = np.zeros((3, num_rcs))
            for j, rcs in enumerate(self.vehicle.rcs_thrusters):
                delta_f = rcs["max_thrust"] * rcs["direction"]
                torque = np.cross(rcs["position"], delta_f)
                A_rcs[:, j] = torque

            residual = desired_torque - achieved_torque
            # Clean up tiny floating-point noise in residual (set near-zero to exact zero)
            residual = np.where(np.abs(residual) < 1e-12, 0.0, residual)

            rcs_levels, _ = nnls(A_rcs, residual)

            # Handle cases where levels > 1 by scaling
            if np.any(rcs_levels > 1):
                max_level = np.max(rcs_levels)
                rcs_levels *= 1.0 / max_level
        else:
            rcs_levels = np.array([])

        return gimbal_angles_list, rcs_levels
