import numpy as np
from utils import (
    rotate_body_to_inertial_by_quat,
    quaternion_multiply,
    quaternion_inverse,
    quat_to_angle_axis,
    compute_orbital_elements,
    compute_time_to_apoapsis,
)


def test_rotate_vector_by_quaternion():
    # Identity quaternion: no rotation
    vector = np.array([1, 0, 0])
    quat = np.array([1, 0, 0, 0])
    rotated = rotate_body_to_inertial_by_quat(vector, quat)
    np.testing.assert_allclose(rotated, vector, atol=1e-4)

    # 90 deg rotation around Z
    quat = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    rotated = rotate_body_to_inertial_by_quat(vector, quat)
    np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-4)


def test_quaternion_multiply():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 1.0, 0.0, 0.0])
    result = quaternion_multiply(q1, q2)
    np.testing.assert_allclose(result, q2)


def test_quaternion_inverse():
    q = np.array([1.0, 1.0, 1.0, 1.0])
    q /= np.linalg.norm(q)  # Normalize
    inv = quaternion_inverse(q)
    product = quaternion_multiply(q, inv)
    np.testing.assert_allclose(product, [1, 0, 0, 0], atol=1e-8)


def test_quat_to_angle_axis():
    # 180 deg around X
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    angle_axis = quat_to_angle_axis(quat)
    np.testing.assert_allclose(angle_axis, [np.pi, 1, 0, 0], atol=1e-8)


def test_compute_orbital_elements_circular():
    # Circular orbit at 400km altitude, equatorial
    mu = 3.986004418e14  # Earth's mu
    r = 6371000 + 400000
    pos = np.array([r, 0, 0])
    v = np.sqrt(mu / r)
    vel = np.array([0, v, 0])
    elements = compute_orbital_elements(pos, vel, mu)
    assert np.isclose(elements["semi_major_axis"], r, atol=1e-3)
    assert np.isclose(elements["eccentricity"], 0, atol=1e-6)
    assert np.isclose(elements["inclination"], 0, atol=1e-6)


def test_compute_time_to_apoapsis():
    # Simple elliptic orbit
    mu = 3.986004418e14
    pos = np.array([6371000 + 100000, 0, 0])  # Near periapsis
    vel = np.array([0, 8000, 0])  # Adjust for elliptic
    elements = compute_orbital_elements(pos, vel, mu)
    time_to_apo = compute_time_to_apoapsis(pos, vel, elements, mu)
    assert time_to_apo > 0  # Basic sanity
