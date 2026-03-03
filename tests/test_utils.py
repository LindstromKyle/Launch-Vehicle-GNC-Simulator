import numpy as np

from utils import (
    rotate_vector_by_quaternion,
    angle_axis_to_quat,
    quaternion_multiply,
    quaternion_inverse,
    quat_to_angle_axis,
    get_rotated_basis_from_quat,
)


def test_rotate_vector_by_quaternion_identity():
    v = np.array([1, 0, 0])
    q_identity = np.array([1, 0, 0, 0])  # No rotation
    rotated = rotate_vector_by_quaternion(v, q_identity)
    assert np.allclose(rotated, v)


def test_angle_axis_to_quat_rotation():
    angle_axis = np.array([np.pi / 4, -1, 1, 0])
    quaternion = angle_axis_to_quat(angle_axis)
    rotated_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), quaternion)

    assert np.allclose(rotated_vector, np.array([0.5, 0.5, 0.70710678]))


def test_error_quat_1():
    current_angle_axis = np.array([np.pi / 4, -1, 1, 0])
    current_quat = angle_axis_to_quat(current_angle_axis)
    current_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), current_quat)

    desired_angle_axis = np.array([np.pi / 2, -1, 1, 0])
    desired_quat = angle_axis_to_quat(desired_angle_axis)
    desired_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), desired_quat)

    error_quaternion = quaternion_multiply(desired_quat, quaternion_inverse(current_quat))
    error_angle_axis = quat_to_angle_axis(error_quaternion)
    final_vector = rotate_vector_by_quaternion(current_vector, error_quaternion)
    assert np.allclose(desired_vector, final_vector)


def test_error_quat_3():

    current_angle_axis = np.array([-np.pi / 2, 1, 0, 0])
    current_quat = angle_axis_to_quat(current_angle_axis)
    current_basis = get_rotated_basis_from_quat(current_quat)

    desired_angle_axis = np.array([-np.pi / 2, 0, 0, 1])
    desired_quat = angle_axis_to_quat(desired_angle_axis)
    desired_basis = get_rotated_basis_from_quat(desired_quat)

    error_quaternion = quaternion_multiply(desired_quat, quaternion_inverse(current_quat))
    error_angle_axis = quat_to_angle_axis(error_quaternion)
    final_basis = get_rotated_basis_from_quat(error_quaternion, basis=current_basis)

    assert np.allclose(desired_basis["X"], final_basis["X"])
    assert np.allclose(desired_basis["Y"], final_basis["Y"])
    assert np.allclose(desired_basis["Z"], final_basis["Z"])
