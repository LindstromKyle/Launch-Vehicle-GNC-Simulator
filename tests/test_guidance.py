import numpy as np
from guidance import ModeBasedGuidance
from utils import compute_minimal_quaternion_rotation


def test_get_desired_quaternion_prograde(basic_environment):
    orbital_normal = np.array([0, 0, 1])
    guidance = ModeBasedGuidance(orbital_normal, basic_environment)
    state = np.zeros(14)
    state[0:3] = [1e6, 0, 0]  # Pos
    state[3:6] = [0, 1000, 0]  # Vel along Y
    setpoints = {"attitude_mode": "prograde"}
    quat = guidance.get_desired_quaternion(0, state, setpoints)
    expected = compute_minimal_quaternion_rotation(np.array([0, 1, 0]))  # Along vel
    np.testing.assert_allclose(quat, expected, atol=1e-4)


def test_get_desired_quaternion_radial(basic_environment):
    orbital_normal = np.array([0, 0, 1])
    guidance = ModeBasedGuidance(orbital_normal, basic_environment)
    state = np.zeros(14)
    state[0:3] = [1e6, 0, 0]  # Along X
    setpoints = {"attitude_mode": "radial"}
    quat = guidance.get_desired_quaternion(0, state, setpoints)
    expected = compute_minimal_quaternion_rotation(np.array([1, 0, 0]))
    np.testing.assert_allclose(quat, expected, atol=1e-4)
