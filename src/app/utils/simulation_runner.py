import numpy as np

from app.models.simulation_models import SimResults, SimulationRequest
from state import State
from simulator import Simulator
from mission import MissionPlanner
from controller import PIDAttitudeController
from vehicle import Falcon9FirstStage, Falcon9SecondStage
from environment import Environment
from guidance import (
    TimeBasedGuidancePhase,
    ProgrammedPitchGuidancePhase,
    PEGGuidancePhase,
    CoastGuidancePhase,
    CircBurnGuidancePhase,
)
from utils import compute_body_z_to_inertial_quat, compute_orbital_elements, rotate_body_to_inertial_by_quat


def run_full_orbit_simulation(request: SimulationRequest) -> dict:
    """
    Runs the two-stage orbital simulation, configurable via request.

    Args:
        request: The SimulationRequest model
    
    Returns:
        Dictionary with final sim results
    """

    env = Environment()
    mu = env.gravitational_constant * env.earth_mass

    # ==================== STAGE 1 ====================
    stage1_inertia = np.diag(request.stage1_moment_of_inertia)

    stage1 = Falcon9FirstStage(
        dry_mass=request.stage1_dry_mass,          # now 167100
        initial_prop_mass=request.stage1_initial_prop_mass,
        base_thrust_magnitude=request.stage1_base_thrust_magnitude,
        average_isp=request.stage1_average_isp,
        moment_of_inertia=stage1_inertia,
        base_drag_coefficient=request.stage1_base_drag_coefficient,
        drag_scaling_coefficient=request.stage1_drag_scaling_coefficient,
        cross_sectional_area=request.stage1_cross_sectional_area,
        engine_gimbal_limit_deg=request.stage1_engine_gimbal_limit_deg,
        engine_gimbal_arm_len=request.stage1_engine_gimbal_arm_len,
        dry_com_z=request.stage1_dry_com_z,
        prop_com_z=request.stage1_prop_com_z,
    )

    # Launch site & initial state
    lat_rad = np.deg2rad(request.launch_latitude_deg)
    initial_position = env.earth_radius * np.array([np.cos(lat_rad), 0.0, np.sin(lat_rad)])
    omega_cross_r = np.cross(env.earth_angular_velocity_vector, initial_position)
    radial_unit = initial_position / np.linalg.norm(initial_position)
    initial_quat = compute_body_z_to_inertial_quat(radial_unit)

    kick_dir_body = np.array([0, 1, 0])
    kick_dir_inertial = rotate_body_to_inertial_by_quat(kick_dir_body, initial_quat)
    horizontal = kick_dir_inertial - np.dot(kick_dir_inertial, radial_unit) * radial_unit
    horizontal /= np.linalg.norm(horizontal)
    orbital_normal = np.cross(radial_unit, horizontal)
    orbital_normal /= np.linalg.norm(orbital_normal)

    initial_state = State(
        position=initial_position,
        velocity=omega_cross_r,
        quaternion=initial_quat,
        angular_velocity=np.zeros(3),
        propellant_mass=request.stage1_initial_prop_mass,
    )

    # Stage 1 phases
    stage1_phases = [
        TimeBasedGuidancePhase(end_time=request.stage1_pitch_start_time, attitude_mode="radial", throttle=1.0, name="Initial Ascent"),
        ProgrammedPitchGuidancePhase(
            start_time=request.stage1_pitch_start_time,
            end_time=request.stage1_burnout_time,
            initial_pitch_deg=request.stage1_initial_pitch_deg,
            final_pitch_deg=request.stage1_final_pitch_deg,
            orbital_normal=orbital_normal,
            kick_direction=kick_dir_inertial,
            throttle=1.0,
            name="Stage 1 Pitch Program",
        ),
    ]

    stage1_planner = MissionPlanner(stage1_phases, env, stage1, start_time=0.0)

    controller1 = PIDAttitudeController(
        kp=np.array(request.stage1_kp),
        ki=np.array(request.stage1_ki),
        kd=np.array(request.stage1_kd),
        vehicle=stage1,
    )

    sim1 = Simulator(
        vehicle=stage1, environment=env, initial_state=initial_state,
        mission_planner=stage1_planner, t_0=0, t_final=request.stage1_burnout_time,
        delta_t=request.delta_t_stage1, log_interval=request.log_interval, log_name="api_stage1",
    )
    sim1.add_controller(controller1)
    stage1_t, stage1_state, stage1_trans = sim1.run()

    # ==================== STAGE 2 ====================
    burnout_vec = stage1_state[-1]
    stage2_state = State(
        position=burnout_vec[:3], velocity=burnout_vec[3:6],
        quaternion=burnout_vec[6:10], angular_velocity=burnout_vec[10:13],
        propellant_mass=request.stage2_initial_prop_mass,
    )

    stage2_inertia = np.diag(request.stage2_moment_of_inertia)

    stage2 = Falcon9SecondStage(
        dry_mass=request.stage2_dry_mass,
        initial_prop_mass=request.stage2_initial_prop_mass,
        base_thrust_magnitude=request.stage2_base_thrust_magnitude,
        average_isp=request.stage2_average_isp,
        moment_of_inertia=stage2_inertia,
        base_drag_coefficient=request.stage2_base_drag_coefficient,
        drag_scaling_coefficient=request.stage2_drag_scaling_coefficient,
        cross_sectional_area=request.stage2_cross_sectional_area,
        engine_gimbal_limit_deg=request.stage2_engine_gimbal_limit_deg,
        engine_gimbal_arm_len=request.stage2_engine_gimbal_arm_len,
        dry_com_z=request.stage2_dry_com_z,
        prop_com_z=request.stage2_prop_com_z,
    )

    target_apo = request.target_apo_alt_km * 1000 + env.earth_radius
    target_peri = request.target_peri_alt_km * 1000 + env.earth_radius
    sim_end_time = request.stage1_burnout_time + 5500

    # Stage 2 phases
    stage2_phases = [
        PEGGuidancePhase(
            target_apoapsis=target_apo,
            target_periapsis=target_peri,
            orbital_normal=orbital_normal,
            vehicle=stage2,
            environment=env,
            apo_tolerance=request.peg_apo_tolerance,
            peri_tolerance=request.peg_peri_tolerance,
            target_inclination=None,
            min_throttle=0.1,
            throttle_kp=20.0,
            throttle_threshold_factor=5.0,
            throttle=1.0,
            name="Stage 2 Ascent Burn",
        ),
        CoastGuidancePhase(
            vehicle=stage2,
            environment=env,
            time_to_apo_threshold=30.0,
            attitude_mode="prograde",
            name="Coast",
            buffer=5.0,
            use_dynamic_threshold=False,
        ),
        CircBurnGuidancePhase(
            environment=env,
            attitude_mode="prograde",
            name="Circularization",
            target_eccentricity=request.circ_target_eccentricity,
            min_throttle=0.1,
            throttle_kp=20.0,
        ),
        TimeBasedGuidancePhase(
            end_time=sim_end_time,
            attitude_mode="passive",
            throttle=0.0,
            name="Orbit",
        ),
    ]

    stage2_planner = MissionPlanner(stage2_phases, env, stage2, start_time=request.stage1_burnout_time)

    controller2 = PIDAttitudeController(
        kp=np.array(request.stage2_kp),
        ki=np.array(request.stage2_ki),
        kd=np.array(request.stage2_kd),
        vehicle=stage2,
    )

    sim2 = Simulator(
        vehicle=stage2, environment=env, initial_state=stage2_state,
        mission_planner=stage2_planner, t_0=request.stage1_burnout_time,
        t_final=sim_end_time, delta_t=request.delta_t_stage2,
        log_interval=request.log_interval, log_name="api_stage2",
    )
    sim2.add_controller(controller2)
    stage2_t, stage2_state, stage2_trans = sim2.run()

    # Combine results (your existing logic)
    all_t = np.append(stage1_t, stage2_t)
    all_state = np.vstack((stage1_state, stage2_state))
    all_trans = [(float(t), name) for t, name in stage1_trans] + [(float(t), name) for t, name in stage2_trans]

    ds = request.downsample
    full_data = {}
    if request.sim_results == SimResults.full:
        full_data = {
            "time_s": all_t[::ds].tolist(),
            "position_km": (all_state[::ds, :3] / 1000).tolist(),
            "velocity_kms": (all_state[::ds, 3:6] / 1000).tolist(),
            "quaternion": all_state[::ds, 6:10].tolist(),
            "phase_transitions": [{"time": t, "name": name} for t, name in all_trans],
        }

    orbital_elements = compute_orbital_elements(
        position=stage2_state[-1, :3],
        velocity=stage2_state[-1, 3:6],
        gravitational_parameter=mu
    )

    return {
        "message": "Simulation completed successfully",
        "summary": {
            "total_duration_s": float(stage2_t[-1]),
            "final_altitude_km": float(np.linalg.norm(stage2_state[-1, :3]) - env.earth_radius) / 1000,
            "apoapsis_altitude_km": float(orbital_elements["apoapsis_radius"] - env.earth_radius) / 1000,
            "periapsis_altitude_km": float(orbital_elements["periapsis_radius"] - env.earth_radius) / 1000,
            "semi_major_axis_km": float(orbital_elements["semi_major_axis"]) / 1000,
            "eccentricity": float(orbital_elements["eccentricity"]),
            "inclination": float(orbital_elements["inclination"]),
        },
        "full_data": full_data
    }
