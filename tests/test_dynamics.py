import numpy as np
from dynamics import calculate_dynamics


def test_calculate_dynamics_gravity_only(basic_environment, basic_vehicle):
    state = np.zeros(14)
    state[2] = basic_environment.earth_radius  # Z pos
    state[6] = 1.0  # Quaternion w
    state[13] = 100.0  # Prop mass
    controls = {"throttle": 0.0, "engine_gimbal_angles": [[0, 0]], "rcs_levels": []}
    deriv = calculate_dynamics(state, basic_vehicle, basic_environment, False, controls)
    accel = deriv[3:6]
    np.testing.assert_allclose(accel, [0, 0, -9.81], atol=0.1)  # Gravity accel


def test_calculate_dynamics_thrust(basic_environment, basic_vehicle):
    state = np.zeros(14)
    state[6] = 1.0  # Quaternion
    state[13] = 100.0
    controls = {"throttle": 1.0, "engine_gimbal_angles": [[0, 0]], "rcs_levels": [], "desired_torque": np.zeros(3)}
    deriv = calculate_dynamics(state, basic_vehicle, basic_environment, False, controls)
    accel = deriv[3:6]
    mass = basic_vehicle.dry_mass + 100
    expected_accel = basic_vehicle.base_thrust_magnitude / mass
    np.testing.assert_allclose(accel[2], expected_accel, atol=1e-3)  # Thrust along +Z
