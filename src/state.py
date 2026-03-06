import numpy as np
from scipy.spatial.transform import Rotation


class State:
    """
    Represents the complete state vector of the rocket.

    Args:
        position: Position vector in inertial frame (m) [x, y, z]
        velocity: Velocity vector in inertial frame (m/s) [vx, vy, vz]
        quaternion: Orientation quaternion [w, x, y, z]
        angular_velocity: Angular velocity vector in body frame (rad/s)
        propellant_mass: Current propellant mass remaining (kg)
    """

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        propellant_mass: float,
    ):

        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.quaternion = Rotation.from_quat(quaternion).as_quat()
        self.angular_velocity = np.array(angular_velocity)
        self.propellant_mass = propellant_mass

    def as_vector(self) -> np.ndarray:
        """
        Convert the full state into a single 1D numpy array for integration.

        Returns:
            1D state vector: [position, velocity, quaternion, ang_vel, prop_mass]
        """
        return np.concatenate(
            [self.position, self.velocity, self.quaternion, self.angular_velocity, [self.propellant_mass]]
        )
