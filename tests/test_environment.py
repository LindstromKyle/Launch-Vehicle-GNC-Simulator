import numpy as np
from environment import Environment
from vehicle import Falcon9FirstStage


# Constants
G = 6.67430e-11
M_earth = 5.972e24
R_earth = 6371000


def test_gravity_at_surface():
    env = Environment()
    position = np.array([R_earth, 0, 0])  # 1 Earth radius away along x-axis
    mass = 1.0  # 1 kg test mass
    g_expected = -G * M_earth * mass / R_earth**2
    g_computed = env.gravitational_force(position, mass)
    assert np.isclose(np.linalg.norm(g_computed), abs(g_expected), rtol=1e-5)


def test_gravity_direction():
    env = Environment()
    position = np.array([0, R_earth, 0])
    force = env.gravitational_force(position, 1.0)
    direction = force / np.linalg.norm(force)
    assert np.allclose(direction, -position / np.linalg.norm(position), rtol=1e-5)


def test_drag_force_zero_velocity():
    env = Environment()
    vehicle = Vehicle(
        dry_mass=25600,
        initial_prop_mass=395700,
        thrust_magnitude=7200000,
        burn_duration=162,
        moment_of_inertia=np.diag([470297, 470297, 705445]),
        base_drag_coefficient=0.3,
        drag_scaling_coefficient=2.0,
        cross_sectional_area=10.5,
        engine_gimbal_limit_deg=10.0,
        engine_gimbal_arm_len=18.0,
    )
    position = np.array([0, 0, R_earth + 1000])
    velocity = np.zeros(3)
    quaternion = np.array([1, 0, 0, 0])
    drag = env.drag_force(position, velocity, vehicle, quaternion)
    assert np.allclose(drag, np.zeros(3))


def test_drag_force_magnitude():
    env = Environment()
    vehicle = Vehicle(
        dry_mass=25600,
        initial_prop_mass=395700,
        thrust_magnitude=7200000,
        burn_duration=162,
        moment_of_inertia=np.diag([470297, 470297, 705445]),
        base_drag_coefficient=0.5,
        drag_scaling_coefficient=2.0,
        cross_sectional_area=0.1,
        engine_gimbal_limit_deg=10.0,
        engine_gimbal_arm_len=18.0,
    )
    velocity = np.array([0, 0, -100])
    position = np.array([0, 0, R_earth + 1000])
    quaternion = np.array([1, 0, 0, 0])
    drag = env.drag_force(position, velocity, vehicle, quaternion)
    rho = env.atmospheric_density(1000)
    expected_magnitude = 0.5 * rho * 100**2 * 0.5 * 0.1
    assert np.isclose(np.linalg.norm(drag), expected_magnitude, rtol=1e-3)
