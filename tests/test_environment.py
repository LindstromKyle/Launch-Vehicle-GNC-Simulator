import numpy as np

from simulator.environment import Environment


def test_gravitational_force_surface(basic_environment):
    # At Earth's surface, along -Z
    pos = np.array([0, 0, basic_environment.earth_radius])
    mass = 1.0
    force = basic_environment.gravitational_force(pos, mass)
    g = -9.80665  # Expected
    np.testing.assert_allclose(force / mass, [0, 0, g], atol=0.1)  # Accel


def test_gravitational_force_zero(basic_environment):
    pos = np.zeros(3)
    force = basic_environment.gravitational_force(pos, 1.0)
    np.testing.assert_allclose(force, [0, 0, 0])


def test_atmospheric_density():
    env = Environment()
    density_sea = env.atmospheric_density(0)
    assert np.isclose(density_sea, 1.225)
    density_high = env.atmospheric_density(100000)
    assert density_high < 1e-5  # Very low at 100km


def test_drag_force(basic_environment, basic_vehicle):
    pos = np.array([0, 0, basic_environment.earth_radius])  # Surface
    vel = np.array([0, 0, 1000])  # Upward
    quat = np.array([1, 0, 0, 0])  # Aligned
    drag = basic_environment.drag_force(pos, vel, basic_vehicle, quat)
    assert drag[2] < 0  # Opposes velocity (negative Z)
    assert np.linalg.norm(drag) > 0


def test_drag_force_zero_vel(basic_environment, basic_vehicle):
    drag = basic_environment.drag_force(
        np.array([0, 0, 1e6]), np.zeros(3), basic_vehicle, np.array([1, 0, 0, 0])
    )
    np.testing.assert_allclose(drag, [0, 0, 0])
