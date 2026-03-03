import numpy as np

from controller import PIDAttitudeController
from environment import Environment
from simulator import Simulator
from state import State
from utils import quat_to_angle_axis
from vehicle import Vehicle


def test_pid_attitude_hold():
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
    environment = Environment()
    initial_state = State(
        position=[0, 0, environment.earth_radius],
        velocity=[0, 0, 0],
        quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0],
    )
    sim = Simulator(
        vehicle=vehicle,
        environment=environment,
        initial_state=initial_state,
        t_0=0,
        t_final=161,
        delta_t=0.1,
        log_interval=0.5,
    )
    controller = PIDAttitudeController(
        kp=np.array([1e4, 1e4, 5e4]),
        ki=np.array([1e-3, 1e-3, 1e-3]),
        kd=np.array([1e6, 1e6, 5e5]),
        desired_quaternion=np.array([0.9990, 0.0, 0.0436, 0.0]),  # 5 deg pitch
        vehicle=sim.vehicle,
    )
    sim.add_controller(controller)
    t_vals, state_vals = sim.run()

    quaternions = state_vals[:, 6:10]
    yaws = []
    for quat in quaternions:
        angle_axis = quat_to_angle_axis(quat)
        angle = np.degrees(angle_axis[0])
        yaws.append(angle)

    # Assert after burnout pitch is ~5 degrees
    assert round(yaws[-1] - 5, 2) == 0
