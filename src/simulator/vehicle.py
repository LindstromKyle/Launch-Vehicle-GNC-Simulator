from abc import ABC, abstractmethod

import numpy as np

from .utils import rotate_body_to_inertial_by_quat


class Vehicle(ABC):
    """
    Abstract base class for rocket stages/vehicles.

    Args:
        dry_mass: Mass of the vehicle without propellant (kg)
        initial_prop_mass: Initial propellant mass (kg)
        base_thrust_magnitude: Thrust at full throttle (N)
        average_isp: Specific impulse (seconds)
        moment_of_inertia: 3×3 inertia tensor in body frame
        base_drag_coefficient: Reference drag coefficient
        drag_scaling_coefficient: Coefficient for angle-of-attack drag increase
        cross_sectional_area: Reference area for drag calculation (m²)
        engine_gimbal_limit_deg: Maximum engine gimbal angle (degrees)
        engine_gimbal_arm_len: Distance from fully loaded CoM to engines (m)
        dry_com_z: Z-coordinate of center of mass with empty tanks (m)
        prop_com_z: Z-coordinate of center of mass with propellant (m)
    """

    def __init__(
        self,
        dry_mass: float,
        initial_prop_mass: float,
        base_thrust_magnitude: float,
        average_isp: float,
        moment_of_inertia: np.ndarray,
        base_drag_coefficient: float,
        drag_scaling_coefficient: float,
        cross_sectional_area: float,
        engine_gimbal_limit_deg: float,
        engine_gimbal_arm_len: float,
        dry_com_z: float,
        prop_com_z: float,
    ):
        self.dry_mass = dry_mass
        self.initial_propellant_mass = initial_prop_mass
        self.base_thrust_magnitude = base_thrust_magnitude
        self.average_isp = average_isp
        self.g0 = 9.80665
        self.mdot_max = (
            base_thrust_magnitude / (average_isp * self.g0) if average_isp > 0 else 0.0
        )
        self.moment_of_inertia = moment_of_inertia
        self.base_drag_coefficient = base_drag_coefficient
        self.drag_scaling_coefficient = drag_scaling_coefficient
        self.cross_sectional_area = cross_sectional_area
        self.engine_gimbal_limit_deg = engine_gimbal_limit_deg
        self.engine_gimbal_limit_rad = np.deg2rad(self.engine_gimbal_limit_deg)
        self.engine_lever_arm = engine_gimbal_arm_len  # Nominal arm length
        self.dry_com_z = dry_com_z
        self.prop_com_z = prop_com_z

        # Stage-specific engine config
        self.num_engines = None
        self.thrust_per_engine = None
        self.engines = []
        self.rcs_thrusters = []
        # Initialize real values
        self._setup_propulsion_system()

    # TODO: This needs to live somewhere else
    def get_grid_fin_deflections(
        self, time: float | None, state: np.ndarray | None
    ) -> dict[str, float]:
        """
        Placeholder for future grid fin / control surface deflection logic.

        Args:
            time: Current simulation time (unused in placeholder)
            state: Current state vector (unused in placeholder)

        Returns:
            Dictionary of fin deflection angles (radians)
        """
        return {
            "Fin 1": 0.0,
            "Fin 2": 0.0,
            "Fin 3": 0.0,
            "Fin 4": 0.0,
        }

    @abstractmethod
    def _setup_propulsion_system(self) -> None:
        """Stage-specific engine and RCS configuration (positions, count, etc.)."""
        pass

    def get_gimbal_arm(self, propellant_mass: float) -> float:
        """
        Compute current lever arm from CoM to engines based on propellant remaining.

        Args:
            propellant_mass: Current propellant mass (kg)

        Returns:
            Z-coordinate of engines relative to current CoM (m)
        """
        if propellant_mass <= 0:
            return self.dry_com_z
        total_mass = self.dry_mass + propellant_mass
        com_z = (
            self.dry_mass * self.dry_com_z + propellant_mass * self.prop_com_z
        ) / total_mass
        return com_z

    def thrust_vector(
        self,
        quaternion: np.ndarray,
        gimbal_angles_list: list[list[float]],
        throttle: float = 1.0,
        propellant_mass: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute total thrust force (inertial) and torque (body) from all engines.

        Args:
            quaternion: Current attitude quaternion [w, x, y, z]
            gimbal_angles_list: List of [pitch, yaw] angles for each engine (rad)
            throttle: Throttle level (0.0 to 1.0)
            propellant_mass: Current propellant mass (kg)

        Returns:
            Tuple of (total thrust force in inertial frame, total torque in body frame)
        """
        if len(gimbal_angles_list) != self.num_engines:
            raise Exception(
                f"Expected {self.num_engines} gimbal angles but received {len(gimbal_angles_list)}"
            )

        # Get dynamic gimbal arm based on propellant mass
        gimbal_arm = self.get_gimbal_arm(propellant_mass)

        thrust_force = np.zeros(3)
        thrust_vector_torque = np.zeros(3)
        for i in range(self.num_engines):
            # Clamp engine angles
            gimbal_pitch, gimbal_yaw = np.clip(
                gimbal_angles_list[i],
                -self.engine_gimbal_limit_rad,
                self.engine_gimbal_limit_rad,
            )
            # Thrust direction in body frame
            body_thrust_direction = np.array(
                [
                    np.sin(gimbal_yaw),
                    -np.sin(gimbal_pitch),
                    np.cos(gimbal_pitch) * np.cos(gimbal_yaw),
                ]
            )
            body_thrust_direction /= np.linalg.norm(body_thrust_direction)
            # Effective thrust from throttle
            engine_thrust = self.thrust_per_engine * throttle
            body_force = engine_thrust * body_thrust_direction
            # Rotate to inertial and add to total thrust force
            inertial_force = rotate_body_to_inertial_by_quat(body_force, quaternion)
            thrust_force += inertial_force
            # Torque: r cross F. Update z position with dynamic CoM
            engine_pos = self.engines[i]["position"].copy()
            engine_pos[2] = -gimbal_arm
            body_torque = np.cross(engine_pos, body_force)
            thrust_vector_torque += body_torque

        return thrust_force, thrust_vector_torque

    def get_thrust_magnitude(self, throttle: float = 1.0) -> float:
        """
        Compute total thrust magnitude at given throttle level.

        Args:
            throttle: Throttle level (0.0 to 1.0)

        Returns:
            Total thrust force magnitude (N)
        """
        return self.base_thrust_magnitude * max(min(throttle, 1.0), 0.0)

    def rcs_vector(
        self, quaternion: np.ndarray, rcs_levels: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute total RCS force (inertial) and torque (body) from all thrusters.

        Args:
            quaternion: Current attitude quaternion [w, x, y, z]
            rcs_levels: List of throttle levels (0.0–1.0) for each RCS thruster

        Returns:
            Tuple of (total RCS force in inertial frame, total RCS torque in body frame)
        """
        if not self.rcs_thrusters or len(rcs_levels) == 0:
            return np.zeros(3), np.zeros(3)

        thrust_force = np.zeros(3)  # Inertial
        thrust_torque = np.zeros(3)  # Body
        for i, level in enumerate(rcs_levels):
            if level <= 0:
                continue
            rcs = self.rcs_thrusters[i]
            body_force = level * rcs["max_thrust"] * rcs["direction"]
            inertial_force = rotate_body_to_inertial_by_quat(body_force, quaternion)
            # Torque: r cross F
            torque = np.cross(rcs["position"], body_force)
            thrust_force += inertial_force
            thrust_torque += torque
        return thrust_force, thrust_torque


class Falcon9FirstStage(Vehicle):
    """
    Falcon 9 first stage. Implements 9 merlin engine layout.
    """

    def _setup_propulsion_system(self) -> None:
        self.num_engines = 9
        self.thrust_per_engine = self.base_thrust_magnitude / self.num_engines
        self.mdot_per_engine = self.mdot_max / self.num_engines
        self.engine_radius = 1.5  # Outer engine radius (m)
        pz = -self.engine_lever_arm
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        outer_positions = [
            np.array(
                [
                    self.engine_radius * np.cos(theta),
                    self.engine_radius * np.sin(theta),
                    pz,
                ]
            )
            for theta in angles
        ]
        center_position = np.array([0.0, 0.0, pz])
        self.engines = [
            {"position": pos} for pos in outer_positions + [center_position]
        ]


class Falcon9SecondStage(Vehicle):
    """
    Falcon 9 second stage. Implements single merlin vacuum engine and RCS thrusters.
    """

    def _setup_propulsion_system(self) -> None:
        self.num_engines = 1
        self.thrust_per_engine = self.base_thrust_magnitude
        self.mdot_per_engine = self.mdot_max
        self.engine_radius = 0.0  # Centered
        pz = -self.engine_lever_arm
        center_position = np.array([0.0, 0.0, pz])
        self.engines = [{"position": center_position}]

        self._setup_rcs()

    def _setup_rcs(self) -> None:
        rcs_max_thrust = 2000.0
        rcs_radius = 1.5
        rcs_arm = 5.0

        # 4 pod positions at 90-degree intervals (body frame)
        positions = [
            np.array([rcs_radius, 0.0, rcs_arm]),
            np.array([0.0, rcs_radius, rcs_arm]),
            np.array([-rcs_radius, 0.0, rcs_arm]),
            np.array([0.0, -rcs_radius, rcs_arm]),
        ]

        # Tangential directions
        tan_dirs = [
            np.array([0.0, 1.0, 0.0]),  # +y at +x pos
            np.array([-1.0, 0.0, 0.0]),  # -x at +y pos
            np.array([0.0, -1.0, 0.0]),  # -y at -x pos
            np.array([1.0, 0.0, 0.0]),  # +x at -y pos
        ]

        # Radial outward directions
        rad_dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]

        self.rcs_thrusters = []
        for i in range(4):
            # Add tangential thrusters in both directions
            for sign in [1.0, -1.0]:
                self.rcs_thrusters.append(
                    {
                        "position": positions[i],
                        "direction": sign * tan_dirs[i] / np.linalg.norm(tan_dirs[i]),
                        "max_thrust": rcs_max_thrust,
                    }
                )
            # Add radial thrusters in both directions
            for sign in [1.0, -1.0]:
                self.rcs_thrusters.append(
                    {
                        "position": positions[i],
                        "direction": sign * rad_dirs[i] / np.linalg.norm(rad_dirs[i]),
                        "max_thrust": rcs_max_thrust,
                    }
                )


class WSeriesCapsule(Vehicle):
    """
    Varda W-series reentry capsule in passive LEO orbit.

    A lightweight capsule with no main propulsion. Only gravitational and
    aerodynamic forces act on it; attitude is passively stable.

    Args:
        dry_mass: Dry mass of the capsule (kg)
        cross_sectional_area: Reference drag area (m²)
    """

    def __init__(self, dry_mass: float = 350.0, cross_sectional_area: float = 1.5):
        inertia = np.diag([120.0, 120.0, 80.0])  # kg·m², small cylindrical capsule
        super().__init__(
            dry_mass=dry_mass,
            initial_prop_mass=0.0,
            base_thrust_magnitude=0.0,
            average_isp=1.0,  # placeholder; zero thrust → zero mass flow
            moment_of_inertia=inertia,
            base_drag_coefficient=1.1,
            drag_scaling_coefficient=0.0,
            cross_sectional_area=cross_sectional_area,
            engine_gimbal_limit_deg=0.0,
            engine_gimbal_arm_len=0.5,
            dry_com_z=0.3,
            prop_com_z=0.3,
        )

    def _setup_propulsion_system(self) -> None:
        self.num_engines = 0
        self.thrust_per_engine = 0.0
        self.mdot_per_engine = 0.0
        self.engines = []
        self.rcs_thrusters = []
