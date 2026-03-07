import numpy as np


def test_get_gimbal_arm(basic_vehicle):
    arm_full = basic_vehicle.get_gimbal_arm(basic_vehicle.initial_propellant_mass)
    assert arm_full < 0  # Below dry CoM
    arm_empty = basic_vehicle.get_gimbal_arm(0)
    assert np.isclose(arm_empty, basic_vehicle.dry_com_z)


def test_thrust_vector(basic_vehicle):
    quat = np.array([1, 0, 0, 0])
    gimbal_angles = [[0, 0]]  # No gimbal
    force, torque = basic_vehicle.thrust_vector(quat, gimbal_angles, throttle=1.0, propellant_mass=100)
    np.testing.assert_allclose(force, [0, 0, basic_vehicle.base_thrust_magnitude], atol=1e-3)  # Along +Z inertial
    np.testing.assert_allclose(torque, [0, 0, 0])  # No torque


def test_thrust_vector_gimbaled(basic_vehicle):
    gimbal_angles = [[np.deg2rad(5), 0]]  # Small pitch
    force, torque = basic_vehicle.thrust_vector(np.array([1, 0, 0, 0]), gimbal_angles, 1.0, 100)
    assert np.linalg.norm(torque) > 0  # Torque generated
