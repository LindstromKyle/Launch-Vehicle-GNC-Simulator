import logging
from abc import ABC, abstractmethod
import numpy as np

from utils import compute_orbital_elements, compute_time_to_apoapsis


class Phase(ABC):
    @abstractmethod
    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        pass

    @abstractmethod
    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        pass


class TimeBasedPhase(Phase):
    def __init__(self, end_time: float, attitude_mode: str, throttle: float = 1.0, name: str = "Unnamed"):
        self.end_time = end_time
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class KickPhase(Phase):
    def __init__(
        self,
        end_time: float,
        kick_direction: np.ndarray,
        kick_angle_deg: float,
        throttle: float = 1.0,
        name: str = "Unnamed",
    ):
        self.end_time = end_time
        self.attitude_mode = "kick"
        self.throttle = throttle
        self.kick_direction = kick_direction
        self.kick_angle_deg = kick_angle_deg
        self.kick_angle_rad = np.deg2rad(self.kick_angle_deg)
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "kick_direction": self.kick_direction,
            "kick_angle_rad": self.kick_angle_rad,
        }


class TargetApoapsisPhase(Phase):
    def __init__(
        self, target_apoapsis: float, attitude_mode: str = "prograde", throttle: float = 1.0, name: str = "Unnamed"
    ):
        self.target_apoapsis = target_apoapsis
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return (elements is not None) and (elements["apoapsis_radius"] >= self.target_apoapsis)

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class PitchToApoapsisPhase(Phase):

    def __init__(
        self,
        initial_pitch_deg: float,
        base_pitch_rate_deg_per_sec: float,
        target_apoapsis: float,
        min_pitch_deg: float = 20.0,
        throttle: float = 1.0,
        name: str = "Pitch to Apoapsis",
        kp_apo: float = 1.0,  # Renamed for clarity
        kp_vel: float = 0.8,  # New: Gain for velocity error (tune lower than apo if velocity is secondary)
        vel_weight: float = 0.5,  # New: How much to weight velocity vs. apo in combined error (0-1)
        vel_threshold_factor: float = 0.95,  # New: Mirror is_complete threshold
    ):
        self.initial_pitch_deg = initial_pitch_deg
        self.base_pitch_rate_deg_per_sec = base_pitch_rate_deg_per_sec
        self.target_apoapsis = target_apoapsis
        self.min_pitch_deg = min_pitch_deg
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "pitch_to_apoapsis"
        self.kp_apo = kp_apo
        self.kp_vel = kp_vel
        self.vel_weight = vel_weight
        self.vel_threshold_factor = vel_threshold_factor

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False

        # Existing apoapsis check
        apo_ok = elements["apoapsis_radius"] >= self.target_apoapsis

        # New: Velocity targeting
        mu = self.mu  # Now available
        a = elements["semi_major_axis"]
        r_apo = elements["apoapsis_radius"]

        if a == float("inf") or r_apo == float("inf"):  # Parabolic/hyperbolic edge case
            return apo_ok  # Fallback to just apoapsis

        # Projected velocity at apoapsis
        v_apo_projected = np.sqrt(mu * (2 / r_apo - 1 / a)) if a > 0 else 0.0  # Handle elliptic

        # Circular velocity at apoapsis altitude
        v_circ = np.sqrt(mu / r_apo)

        # Threshold: e.g., 95% of circular speed (tune this)
        vel_threshold = 0.95
        vel_ok = v_apo_projected >= vel_threshold * v_circ

        return apo_ok and vel_ok

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "initial_pitch_deg": self.initial_pitch_deg,
            "base_pitch_rate_deg_per_sec": self.base_pitch_rate_deg_per_sec,
            "min_pitch_deg": self.min_pitch_deg,
            "kp_apo": self.kp_apo,
            "kp_vel": self.kp_vel,  # New
            "vel_weight": self.vel_weight,  # New
            "target_apoapsis": self.target_apoapsis,
            "vel_threshold_factor": self.vel_threshold_factor,  # New
        }


class CoastPhase(Phase):
    def __init__(
        self,
        time_to_apo_threshold: float = 30.0,  # Fallback fixed threshold (s)
        attitude_mode: str = "prograde",
        throttle: float = 0.0,
        name: str = "Coast",
        buffer: float = 5.0,  # Safety buffer (s) for starting early
        use_dynamic_threshold: bool = True,  # Toggle for dynamic vs. fixed
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
            # Fallback to fixed threshold for hyperbolic/elliptic edge cases
            time_to_apo = compute_time_to_apoapsis(position, velocity, elements, self.mu)
            return time_to_apo <= self.time_to_apo_threshold

        # Dynamic computation
        r_apo = elements["apoapsis_radius"]
        a = elements["semi_major_axis"]
        v_apo = np.sqrt(self.mu * (2 / r_apo - 1 / a))
        v_circ = np.sqrt(self.mu / r_apo)
        delta_v = v_circ - v_apo

        if delta_v <= 0:
            return True  # Already circular or super-circular

        # Estimate burn time (assume full throttle for conservatism)
        prop_mass = state_vector[13]
        m0 = self.vehicle.dry_mass + prop_mass
        ve = self.vehicle.average_isp * 9.80665  # g0 standard value
        T = self.vehicle.base_thrust_magnitude  # Can multiply by e.g. 0.7 if average throttle <1
        if T <= 0 or ve <= 0 or m0 <= 0:
            return False  # Invalid; fallback implicitly

        burn_time = (m0 * ve / T) * (1 - np.exp(-delta_v / ve))
        half_burn = burn_time / 2 + self.buffer

        time_to_apo = compute_time_to_apoapsis(position, velocity, elements, self.mu)
        return time_to_apo <= half_burn

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class CircBurnPhase(Phase):
    def __init__(
        self,
        peri_tolerance_factor: float = 0.95,
        attitude_mode: str = "prograde",
        throttle: float = 1.0,  # Max/default throttle
        name: str = "Unnamed",
        min_throttle: float = 0.1,  # Engine minimum (prevent flameout/stability issues)
        throttle_kp: float = 20.0,  # Proportional gain; tune so error=0.05 -> throttle=1.0
    ):
        self.peri_tolerance_factor = peri_tolerance_factor
        self.attitude_mode = attitude_mode
        self.throttle = throttle  # Unused now; kept for consistency
        self.name = name
        self.min_throttle = min_throttle
        self.throttle_kp = throttle_kp

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False

        # Existing check
        peri_ok = elements["periapsis_radius"] >= self.peri_tolerance_factor * elements["apoapsis_radius"]

        # Enhanced checks
        ecc_ok = elements["eccentricity"] < 0.01  # Near-circular

        # Energy check similar to above
        mu = self.mu
        a = elements["semi_major_axis"]
        specific_energy = -mu / (2 * a) if a > 0 else float("inf")
        r_apo = elements["apoapsis_radius"]
        target_circ_energy = -mu / (2 * r_apo)  # Circular at current apo
        margin = 1e3
        energy_ok = abs(specific_energy - target_circ_energy) <= margin

        # Complete if peri ok AND ecc/energy stable
        return peri_ok and ecc_ok and energy_ok

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        if elements["apoapsis_radius"] == float("inf") or elements["periapsis_radius"] <= 0:
            # Safety: Fallback for hyperbolic or invalid orbits (e.g., early in sim)
            dynamic_throttle = 1.0
        else:
            ratio = elements["periapsis_radius"] / elements["apoapsis_radius"]
            error = max(0.0, 1.0 - ratio)  # Positive error; 0 when circular/peri > apo (rare)
            dynamic_throttle = max(self.min_throttle, min(1.0, self.throttle_kp * error))

        return {
            "throttle": dynamic_throttle,
            "attitude_mode": self.attitude_mode,
            "target_r": elements.get("apoapsis_radius", float("inf")),  # For consistency with PEG
        }


class ProgrammedPitchPhase(Phase):
    def __init__(
        self,
        end_time: float,
        initial_pitch_deg: float,
        final_pitch_deg: float,
        kick_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),  # Default eastward
        throttle: float = 1.0,
        name: str = "Pitch Program",
    ):
        self.end_time = end_time
        self.initial_pitch_deg = initial_pitch_deg
        self.final_pitch_deg = final_pitch_deg
        self.kick_direction = kick_direction
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "programmed_pitch"  # Custom mode for guidance

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "initial_pitch_deg": self.initial_pitch_deg,
            "final_pitch_deg": self.final_pitch_deg,
            "kick_direction": self.kick_direction,
            # Note: start_time and duration are added dynamically by MissionPlanner
        }


class PEGPhase(Phase):
    def __init__(
        self,
        target_apoapsis: float,
        apo_tolerance: float = 1000.0,  # m tolerance for apoapsis
        vel_threshold_factor: float = 0.92,  # Lower for elliptical (leave more for circ)
        throttle: float = 1.0,
        name: str = "PEG Guidance",
    ):
        self.attitude_mode = "peg"
        self.target_apoapsis = target_apoapsis
        self.apo_tolerance = apo_tolerance
        self.vel_threshold_factor = vel_threshold_factor
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False

        # Existing checks
        apo_ok = abs(elements["apoapsis_radius"] - self.target_apoapsis) <= self.apo_tolerance
        mu = self.mu  # Injected via MissionPlanner
        a = elements["semi_major_axis"]
        r_apo = elements["apoapsis_radius"]
        if a == float("inf") or r_apo == float("inf"):
            return apo_ok
        v_apo_projected = np.sqrt(mu * (2 / r_apo - 1 / a)) if a > 0 else 0.0
        v_circ = np.sqrt(mu / r_apo)
        vel_ok = v_apo_projected >= self.vel_threshold_factor * v_circ

        # Enhanced checks for stability
        ecc_ok = elements["eccentricity"] < 0.8  # Sub-orbital but not too hyperbolic

        # Periapsis safety: complete early if approaching orbital (peri > -100 km altitude)
        # Assuming earth_radius is accessible via self.environment.earth_radius (inject if needed)
        # TODO: better way to handle this than hard-code earth radius
        peri_alt = elements["periapsis_radius"] - 6371000
        peri_approaching = peri_alt > -100000  # -100 km

        # Energy check: specific orbital energy
        specific_energy = -mu / (2 * a) if a > 0 else float("inf")
        target_circ_energy = -mu / (2 * self.target_apoapsis)  # For circular at apo
        margin = 1e3  # m^2/s^2, tune as needed
        energy_high = specific_energy > target_circ_energy - margin

        # Complete if main criteria met AND ecc ok, OR safety triggers
        return (apo_ok and vel_ok and ecc_ok) or peri_approaching or energy_high

    def get_setpoints(self, time: float, state_vector: np.ndarray, elements: dict) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "target_r": self.target_apoapsis,
            "vel_threshold_factor": self.vel_threshold_factor,
        }


class MissionPlanner:
    def __init__(self, phases: list[Phase], environment, vehicle, start_time: float = 0.0):
        self.phases = phases
        self.current_phase_idx = 0
        self.current_phase = phases[0]
        self.mu = environment.gravitational_constant * environment.earth_mass
        self.environment = environment
        self.vehicle = vehicle
        self.phase_transitions = [(start_time, phases[0].name)]
        self.phase_start_times = [start_time] * len(phases)  # Initialize list for each phase's start time
        self.phase_start_times[0] = start_time  # Set initial
        # Inject mu to phases if needed (for CoastPhase)
        for phase in self.phases:
            # TODO: find better way than his hack
            phase.mu = self.mu
            phase.vehicle = self.vehicle

    def update(self, time: float, state_vector: np.ndarray, log_flag: bool) -> dict:
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        elements = compute_orbital_elements(position, velocity, self.mu)
        altitude = (np.linalg.norm(position) - self.environment.earth_radius) / 1000

        if log_flag:
            # Full orbital velocity (magnitude)
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

        self.current_phase = self.phases[self.current_phase_idx]
        if self.current_phase.is_complete(time, state_vector, elements):
            self.current_phase_idx += 1
            if self.current_phase_idx < len(self.phases):
                self.phase_transitions.append((time, self.phases[self.current_phase_idx].name))
                # Update start time for the new phase
                self.phase_start_times[self.current_phase_idx] = time
            else:
                logging.info(f"Integration segment complete at t={time:.2f}")
                return {"throttle": 0.0, "attitude_mode": "prograde"}
            self.current_phase = self.phases[self.current_phase_idx]

        # Get base setpoints from phase
        setpoints = self.current_phase.get_setpoints(time, state_vector, elements)

        # Dynamically add start_time and duration if the phase supports it (e.g., for ProgrammedPitchPhase)
        if setpoints.get("attitude_mode") in ["programmed_pitch", "pitch_to_apoapsis"]:
            start_time = self.phase_start_times[self.current_phase_idx]
            setpoints["start_time"] = start_time
            if setpoints.get("attitude_mode") == "programmed_pitch":
                duration = self.current_phase.end_time - start_time
                setpoints["duration"] = duration

        setpoints["mu"] = self.mu
        setpoints["g0"] = 9.80665  # Standard gravity
        setpoints["target_r"] = setpoints.get(
            "target_r", elements["apoapsis_radius"]
        )  # For circ phase, use current apo
        # Pass vehicle params (thrust, isp from vehicle; assume constant for now)
        setpoints["thrust"] = (
            self.vehicle.base_thrust_magnitude
        )  # Wait, vehicle is not in environment; fix: add self.vehicle = vehicle in MissionPlanner __init__
        # In MissionPlanner __init__, add self.vehicle = vehicle (pass vehicle in orbit.py when creating planner)
        setpoints["thrust"] = self.vehicle.base_thrust_magnitude
        setpoints["isp"] = self.vehicle.average_isp
        setpoints["dry_mass"] = self.vehicle.dry_mass

        return setpoints

    def get_phase_transitions(self) -> list[tuple[float, str]]:
        return self.phase_transitions
