import numpy as np

from controller import PIDAttitudeController
from environment import Environment
from guidance import ModeBasedGuidance
from mission import CoastPhase, CircBurnPhase, TimeBasedPhase, MissionPlanner, TargetApoapsisPhase, TargetApoPitchPhase
from simulator import Simulator
from state import State
from vehicle import Falcon9SecondStage

# Stage 2 params
stage2_dry_mass = 4000
stage2_prop = 111500
stage2_thrust = 934000
stage2_avg_isp = 348
stage2_moi = np.diag([10000, 10000, 20000])  # Approximate scaled
stage2_cd_base = 0.3
stage2_cd_scale = 2.0
stage2_area = 7.0  # Smaller
stage2_gimbal_limit = 5.0  # Vacuum engine
stage2_gimbal_arm = 2.0
stage2_dry_com_z = 3.0
stage2_prop_com_z = 6.0
separation_time = 162  # For now; later based on velocity/alt

# Environment
environment = Environment()


# State vals from Stage 1 Sim
burnout_state_vector = np.array(
    [
        5672360.837266085,
        168103.82985744637,
        3079231.728317932,
        882.6120740259768,
        2458.1781805775236,
        467.79933736021735,
        0.7564277164859167,
        -0.6269429798924853,
        0.18643147721624603,
        0.0017070437607218827,
        -0.0063562536208505,
        -0.005572160288829116,
        -0.00624481871270561,
        0.0,
    ]
)

current_state = State(
    position=burnout_state_vector[:3],
    velocity=burnout_state_vector[3:6],
    quaternion=burnout_state_vector[6:10],
    angular_velocity=burnout_state_vector[10:13],
    propellant_mass=stage2_prop,
)

# Stage 2 Vehicle
stage_2 = Falcon9SecondStage(
    dry_mass=stage2_dry_mass,
    initial_prop_mass=stage2_prop,
    base_thrust_magnitude=stage2_thrust,
    average_isp=stage2_avg_isp,
    moment_of_inertia=stage2_moi,
    base_drag_coefficient=stage2_cd_base,
    drag_scaling_coefficient=stage2_cd_scale,
    cross_sectional_area=stage2_area,
    engine_gimbal_limit_deg=stage2_gimbal_limit,
    engine_gimbal_arm_len=stage2_gimbal_arm,
    dry_com_z=stage2_dry_com_z,
    prop_com_z=stage2_prop_com_z,
)

target_alt = 200000.0
target_apoapsis = target_alt + environment.earth_radius
# simulation_end_time = separation_time + 1200
simulation_end_time = 477
stage2_phases = [
    TargetApoPitchPhase(
        target_apoapsis=target_apoapsis,
        initial_pitch_deg=20,
        final_pitch_deg=0,
        kick_direction=np.array([0.0, 1.0, 0.0]),  # Default eastward
        throttle=1.0,
        name="Stage 2 Ascent Burn",
    ),
    CoastPhase(time_to_apo_threshold=110.0, attitude_mode="prograde", throttle=0.0, name="Coast"),
    CircBurnPhase(peri_tolerance_factor=0.99, attitude_mode="prograde", throttle=1.0, name="Circularization"),
    TimeBasedPhase(end_time=simulation_end_time, attitude_mode="prograde", throttle=0.0, name="Orbit"),
]
stage2_planner = MissionPlanner(phases=stage2_phases, environment=environment, start_time=separation_time)

# orbital_normal = np.cross(burnout_state_vector[:3], burnout_state_vector[3:6])
# orbital_normal /= np.linalg.norm(orbital_normal)
orbital_normal = np.array([-0.4771586578473682, 5.421010862427522e-20, 0.8788171682672672])
stage2_guidance = ModeBasedGuidance(orbital_normal=orbital_normal)

stage2_p = 3e2
stage2_i = 0
stage2_d = 1e4

controller_stage2 = PIDAttitudeController(
    kp=np.array([stage2_p, stage2_p, 2 * stage2_p]),
    ki=np.array([stage2_i, stage2_i, 2 * stage2_i]),
    kd=np.array([stage2_d, stage2_d, 2 * stage2_d]),
    guidance=stage2_guidance,
    vehicle=stage_2,
)

sim_stage2 = Simulator(
    vehicle=stage_2,
    environment=environment,
    initial_state=current_state,
    mission_planner=stage2_planner,
    t_0=separation_time,
    t_final=simulation_end_time,  # Enough for orbit
    delta_t=0.1,  # Larger step ok for vacuum
    log_interval=1,
    log_name="stage_2",
)
sim_stage2.add_controller(controller_stage2)

print("\nSimulating Stage 2 to Orbit...")
stage2_t_vals, stage2_state_vals, stage2_phase_transitions = sim_stage2.run()
