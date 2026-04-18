import numpy as np

from simulator.guidance import TimeBasedGuidancePhase
from simulator.utils import compute_body_z_to_inertial_quat


def test_get_desired_quaternion_prograde():
    guidance = TimeBasedGuidancePhase(end_time=100.0, attitude_mode="prograde")
    state = np.zeros(14)
    state[0:3] = [1e6, 0, 0]  # Pos
    state[3:6] = [0, 1000, 0]  # Vel along Y
    desired_quaternion, throttle = guidance.get_setpoints(time=1.0, state_vector=state)
    expected = compute_body_z_to_inertial_quat(np.array([0, 1, 0]))  # Along vel
    np.testing.assert_allclose(desired_quaternion, expected, atol=1e-4)


def test_get_desired_quaternion_radial():
    guidance = TimeBasedGuidancePhase(end_time=100.0, attitude_mode="radial")
    state = np.zeros(14)
    state[0:3] = [1e6, 0, 0]  # Along X
    desired_quaternion, throttle = guidance.get_setpoints(time=1.0, state_vector=state)
    expected = compute_body_z_to_inertial_quat(np.array([1, 0, 0]))
    np.testing.assert_allclose(desired_quaternion, expected, atol=1e-4)
