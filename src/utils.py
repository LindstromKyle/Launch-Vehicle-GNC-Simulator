import numpy as np
from scipy.spatial.transform import Rotation
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)


def compute_acceleration(t_vals, velocity_vals):
    acceleration_vals = np.gradient(velocity_vals, t_vals)
    return acceleration_vals


def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rotate a vector from body frame to ECI frame using a quaternion.
    Args:
        vector: 3D numpy array in body frame
        quaternion: Quaternion [w, x, y, z] representing ECI←body rotation

    returns: Rotated vector in ECI frame
    """
    # Reorder quaternion to match scipy's format: [q1, q2, q3, q0]
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(q)
    return np.round(rotation.apply(vector), 4)


def get_rotated_basis_from_quat(quaternion, basis=None):
    """
    Compute the rotated body-frame basis vectors in ECI frame.
    Args:
        quaternion: Quaternion [w, x, y, z] representing ECI←body rotation
        basis: Optional param - basis to start rotation from

    Returns: Dict with rotated unit vectors for body X, Y, Z in ECI
    """
    # Reorder quaternion to match scipy's format: [q1, q2, q3, q0]
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(q)

    if basis:
        rotated_basis = {
            "X": np.round(rotation.apply(basis["X"]), 4),
            "Y": np.round(rotation.apply(basis["Y"]), 4),
            "Z": np.round(rotation.apply(basis["Z"]), 4),
        }

    else:
        rotated_basis = {
            "X": np.round(rotation.apply([1, 0, 0]), 4),
            "Y": np.round(rotation.apply([0, 1, 0]), 4),
            "Z": np.round(rotation.apply([0, 0, 1]), 4),
        }

    return rotated_basis


def compute_quaternion_derivative(quaternion, angular_velocity):
    """
    Computes dq/dt given quaternion and body-frame angular velocity.
    Quaternion is in [q0, q1, q2, q3] format.
    """
    w_x, w_y, w_z = angular_velocity
    Omega = np.array(
        [
            [0, -w_x, -w_y, -w_z],
            [w_x, 0, w_z, -w_y],
            [w_y, -w_z, 0, w_x],
            [w_z, w_y, -w_x, 0],
        ]
    )
    derivative = 0.5 * Omega @ quaternion
    return derivative


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.linalg.norm(q) ** 2


def quat_to_angle_axis(q: np.ndarray) -> np.ndarray:
    q /= np.linalg.norm(q)
    # Check for rotations greater than 180 deg - flip quaternion for minimal angle
    if q[0] < 0:
        q = -q
    angle = 2 * np.arccos(np.clip(q[0], -1.0, 1.0))
    if np.sin(angle / 2) < 1e-10:
        return np.array([0, 0, 0, 0])  # No rotation
    axis = q[1:] / np.sin(angle / 2)

    # Enforce the shortest path for large angles
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis

    return np.append(angle, axis)


def angle_axis_to_quat(angle_axis: np.ndarray) -> np.ndarray:
    axis_norm = np.linalg.norm(angle_axis[1:])
    angle = angle_axis[0]
    if (axis_norm == 0) or (angle == 0):
        return np.array([1, 0, 0, 0])
    unit_axis = angle_axis[1:] / axis_norm
    quaternion = np.append(np.array([np.cos(angle / 2)]), np.sin(angle / 2) * unit_axis)
    return quaternion / np.linalg.norm(quaternion)


def compute_minimal_quaternion_rotation(desired_z_vector):
    # Compute minimal quaternion to rotate body Z [0,0,1] to desired_z
    body_z_vector = np.array([0, 0, 1])
    dot = np.dot(body_z_vector, desired_z_vector)
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.99999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.99999:
        # 180 flip
        return np.array([0.0, 1.0, 0.0, 0.0])

    cross_product = np.cross(body_z_vector, desired_z_vector)
    cross_norm = np.linalg.norm(cross_product)
    if cross_norm < 1e-6:
        axis = np.array([0.0, 0.0, 0.0])
    else:
        axis = cross_product / cross_norm
    angle = np.arccos(dot)
    sin_half = np.sin(angle / 2.0)
    cos_half = np.cos(angle / 2.0)
    quat = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])
    return quat / np.linalg.norm(quat)


def compute_orbital_elements(position: np.ndarray, velocity: np.ndarray, gravitational_parameter: float) -> dict:
    """
    Compute Keplerian elements from position and velocity (inertial frame).
    Returns dict with 'semi_major_axis', 'eccentricity', 'apoapsis_radius', 'periapsis_radius', 'eccentricity_vector'.
    Handles hyperbolic and parabolic cases without returning None.
    """
    position_magnitude = np.linalg.norm(position)
    velocity_squared = np.dot(velocity, velocity)
    kinetic_energy = velocity_squared / 2
    gravitational_potential_energy = gravitational_parameter / position_magnitude
    specific_orbital_energy = kinetic_energy - gravitational_potential_energy

    angular_momentum_vector = np.cross(position, velocity)
    eccentricity_vector = (1 / gravitational_parameter) * np.cross(
        velocity, angular_momentum_vector
    ) - position / position_magnitude
    eccentricity = np.linalg.norm(eccentricity_vector)

    ENERGY_TOLERANCE = 1e-10

    if specific_orbital_energy < -ENERGY_TOLERANCE:
        # Elliptical orbit
        semi_major_axis = -gravitational_parameter / (2 * specific_orbital_energy)
        apoapsis_radius = semi_major_axis * (1 + eccentricity)
        periapsis_radius = semi_major_axis * (1 - eccentricity)
    elif specific_orbital_energy > ENERGY_TOLERANCE:
        # Hyperbolic orbit
        semi_major_axis = -gravitational_parameter / (2 * specific_orbital_energy)
        apoapsis_radius = float("inf")
        periapsis_radius = semi_major_axis * (1 - eccentricity)
    else:
        # Parabolic orbit
        semi_major_axis = float("inf")
        apoapsis_radius = float("inf")
        h = np.linalg.norm(angular_momentum_vector)
        semi_latus_rectum = h**2 / gravitational_parameter
        periapsis_radius = semi_latus_rectum / 2
        eccentricity = 1.0

    h_norm = np.linalg.norm(angular_momentum_vector)
    inclination = np.arccos(angular_momentum_vector[2] / h_norm) if h_norm > 0 else 0.0

    return {
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "apoapsis_radius": apoapsis_radius,
        "periapsis_radius": periapsis_radius,
        "eccentricity_vector": eccentricity_vector,
        "inclination": inclination,
    }


def compute_time_to_apoapsis(
    position: np.ndarray, velocity: np.ndarray, orbital_elements: dict, gravitational_parameter: float
) -> float:
    """
    Compute time (s) to next apoapsis from current state.
    Assumes elliptic orbit (orbital_elements not None).
    """
    position_magnitude = np.linalg.norm(position)
    eccentricity_vector = orbital_elements["eccentricity_vector"]
    eccentricity = orbital_elements["eccentricity"]
    semi_major_axis = orbital_elements["semi_major_axis"]
    # True anomaly
    cosine_true_anomaly = np.dot(eccentricity_vector, position) / (eccentricity * position_magnitude)
    cosine_true_anomaly = np.clip(cosine_true_anomaly, -1.0, 1.0)
    true_anomaly_initial = np.arccos(cosine_true_anomaly)
    radial_velocity = np.dot(velocity, position) / position_magnitude  # Radial velocity
    if radial_velocity < 0:
        true_anomaly = 2 * np.pi - true_anomaly_initial
    else:
        true_anomaly = true_anomaly_initial
    # Eccentric anomaly
    sqrt_term = np.sqrt((1 - eccentricity) / (1 + eccentricity))
    tangent_half_true_anomaly = np.tan(true_anomaly / 2)
    eccentric_anomaly = 2 * np.arctan(sqrt_term * tangent_half_true_anomaly)
    if eccentric_anomaly < 0:
        eccentric_anomaly += 2 * np.pi
    # Mean anomaly
    mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
    if mean_anomaly < np.pi:
        delta_mean_anomaly = np.pi - mean_anomaly
    else:
        delta_mean_anomaly = np.pi - mean_anomaly + 2 * np.pi
    # Mean motion
    mean_motion = np.sqrt(gravitational_parameter / semi_major_axis**3)
    time_to_apoapsis = delta_mean_anomaly / mean_motion
    return time_to_apoapsis
