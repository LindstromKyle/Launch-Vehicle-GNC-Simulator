import warnings

import numpy as np
from scipy.spatial.transform import Rotation

warnings.filterwarnings("error", category=RuntimeWarning)


def rotate_body_to_inertial_by_quat(
    vector: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    """
    Rotate a vector from body frame to inertial frame using a quaternion.

    Args:
        vector: 3D vector in body frame
        quaternion: Quaternion [w, x, y, z]

    Returns:
        Rotated vector in inertial (ECI) frame
    """
    # Reorder quaternion to match scipy's format: [x, y, z, w]
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(q)
    return np.round(rotation.apply(vector), 4)


def compute_quaternion_derivative(
    quaternion: np.ndarray, angular_velocity: np.ndarray
) -> np.ndarray:
    """
    Compute time derivative of quaternion from angular velocity.

    Args:
        quaternion: Current quaternion [w, x, y, z]
        angular_velocity: Angular velocity vector in body frame (rad/s)

    Returns:
        Quaternion derivative dq/dt
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
    """
    Multiply two quaternions

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion q1 * q2
    """
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
    """
    Compute the inverse of a quaternion.

    Args:
        q: Input quaternion [w, x, y, z]

    Returns:
        Inverse quaternion
    """
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.linalg.norm(q) ** 2


def quat_to_angle_axis(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle representation.

    Args:
        q: Input quaternion [w, x, y, z]

    Returns:
        Array [angle, axis_x, axis_y, axis_z]
    """
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
    """
    Convert axis-angle representation to quaternion.

    Args:
        angle_axis: Array [angle (rad), axis_x, axis_y, axis_z]

    Returns:
        Quaternion [w, x, y, z]
    """
    axis_norm = np.linalg.norm(angle_axis[1:])
    angle = angle_axis[0]
    if (axis_norm == 0) or (angle == 0):
        return np.array([1, 0, 0, 0])
    unit_axis = angle_axis[1:] / axis_norm
    quaternion = np.append(np.array([np.cos(angle / 2)]), np.sin(angle / 2) * unit_axis)
    return quaternion / np.linalg.norm(quaternion)


def compute_body_z_to_inertial_quat(desired_z_vector: np.ndarray) -> np.ndarray:
    """
    Compute the minimal rotation quaternion that aligns body Z-axis with a target direction.

    Args:
        desired_z_vector: Desired direction for body Z-axis (unit vector)

    Returns:
        Quaternion that rotates [0,0,1] to desired_z_vector
    """
    body_z_vector = np.array([0, 0, 1])
    dot = np.dot(body_z_vector, desired_z_vector)
    if abs(dot - 1.0) < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Already aligned
    if abs(dot + 1.0) < 1e-6:
        # 180 degree rotation around arbitrary perpendicular axis
        perp = (
            np.array([1, 0, 0]) if abs(body_z_vector[0]) < 0.9 else np.array([0, 1, 0])
        )
        cross = np.cross(body_z_vector, perp)
        cross /= np.linalg.norm(cross)
        return np.array([0.0, cross[0], cross[1], cross[2]])

    cross = np.cross(body_z_vector, desired_z_vector)
    w = 1.0 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]])
    quat /= np.linalg.norm(quat)
    return quat


def compute_orbital_elements(
    position: np.ndarray, velocity: np.ndarray, gravitational_parameter: float
) -> dict[str, float | np.ndarray]:
    """
    Compute Keplerian orbital elements from position and velocity.

    Args:
        position: Position vector (m)
        velocity: Velocity vector (m/s)
        gravitational_parameter: μ = GM (m³/s²)

    Returns:
        Dictionary with semi-major axis, eccentricity, apoapsis, periapses, etc.
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

    energy_tolerance = 1e-10

    if specific_orbital_energy < -energy_tolerance:
        # Elliptical orbit
        semi_major_axis = -gravitational_parameter / (2 * specific_orbital_energy)
        apoapsis_radius = semi_major_axis * (1 + eccentricity)
        periapsis_radius = semi_major_axis * (1 - eccentricity)
    elif specific_orbital_energy > energy_tolerance:
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
    position: np.ndarray,
    velocity: np.ndarray,
    orbital_elements: dict,
    gravitational_parameter: float,
) -> float:
    """
    Compute time to next apoapsis for an elliptic orbit.

    Args:
        position: Current position vector (m)
        velocity: Current velocity vector (m/s)
        orbital_elements: Dictionary of orbital elements
        gravitational_parameter: μ = G·M (m³/s²)

    Returns:
        Time to apoapsis (seconds)
    """
    position_magnitude = np.linalg.norm(position)
    eccentricity_vector = orbital_elements["eccentricity_vector"]
    eccentricity = orbital_elements["eccentricity"]
    semi_major_axis = orbital_elements["semi_major_axis"]

    # True anomaly
    cosine_true_anomaly = np.dot(eccentricity_vector, position) / (
        eccentricity * position_magnitude
    )
    cosine_true_anomaly = np.clip(cosine_true_anomaly, -1.0, 1.0)
    true_anomaly_initial = np.arccos(cosine_true_anomaly)
    radial_velocity = np.dot(velocity, position) / position_magnitude
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


def compute_acceleration(t_vals: np.ndarray, velocity_vals: np.ndarray) -> np.ndarray:
    """
    Compute acceleration from velocity.

    Args:
        t_vals: Array of time points (s)
        velocity_vals: Array of velocity values (m/s)

    Returns:
        Array of acceleration values (m/s²)
    """
    acceleration_vals = np.gradient(velocity_vals, t_vals)
    return acceleration_vals


def quaternion_from_attitude_mode(
    state_vector: np.ndarray, attitude_mode: str
) -> np.ndarray:

    position = state_vector[:3]
    velocity = state_vector[3:6]
    radial_unit_vector = position / np.linalg.norm(position)
    velocity_magnitude = np.linalg.norm(velocity)

    if attitude_mode == "radial":
        desired_z_vector = radial_unit_vector
    elif attitude_mode == "prograde":
        desired_z_vector = velocity / velocity_magnitude
    elif attitude_mode == "retrograde":
        desired_z_vector = -velocity / velocity_magnitude
    elif attitude_mode == "radial_down":
        desired_z_vector = -radial_unit_vector
    elif attitude_mode == "passive":
        # Set desired to current quaternion (no control needed)
        desired_quaternion = state_vector[6:10].copy()
        desired_quaternion /= np.linalg.norm(desired_quaternion)
        return desired_quaternion
    return compute_body_z_to_inertial_quat(desired_z_vector)
