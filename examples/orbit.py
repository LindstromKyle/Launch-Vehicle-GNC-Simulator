import numpy as np

from controller import PIDAttitudeController
from environment import Environment
from guidance import (
    TimeBasedGuidancePhase,
    ProgrammedPitchGuidancePhase,
    PEGGuidancePhase,
    CoastGuidancePhase,
    CircBurnGuidancePhase,
)
from mission import MissionPlanner
from plotting import plot_3D_integration_segments
from simulator import Simulator
from state import State
from utils import compute_body_z_to_inertial_quat, rotate_body_to_inertial_by_quat
from vehicle import Falcon9FirstStage, Falcon9SecondStage

# Stage 1
stage1_dry_mass = 25600
stage1_ascent_prop = 395700
stage1_reserve_prop = 30000  # Return burn
stage1_thrust = 7600000
stage1_avg_isp = 300
stage1_moi = np.diag([470297, 470297, 705445])
stage1_cd_base = 0.3
stage1_cd_scale = 0.2
stage1_area = 10.5
stage1_gimbal_limit = 10.0
stage1_gimbal_arm = 20.0
burnout_time = 155.0

# Stage 2 params
stage2_dry_mass = 4000
stage2_prop = 111500
stage2_thrust = 934000
stage2_avg_isp = 348
stage2_moi = np.diag([10000, 10000, 20000])
stage2_cd_base = 0.3
stage2_cd_scale = 2.0
stage2_area = 7.0
stage2_gimbal_limit = 5.0
stage2_gimbal_arm = 2.0
stage2_dry_com_z = 3.0
stage2_prop_com_z = 6.0
separation_time = burnout_time

# Combined vehicle for ascent
combined_dry_mass = stage1_dry_mass + stage1_reserve_prop + stage2_dry_mass + stage2_prop

# Vehicle
stage_1 = Falcon9FirstStage(
    dry_mass=combined_dry_mass,
    initial_prop_mass=stage1_ascent_prop,
    base_thrust_magnitude=stage1_thrust,
    average_isp=stage1_avg_isp,
    moment_of_inertia=stage1_moi + stage2_moi,
    base_drag_coefficient=stage1_cd_base,
    drag_scaling_coefficient=stage1_cd_scale,
    cross_sectional_area=stage1_area,
    engine_gimbal_limit_deg=stage1_gimbal_limit,
    engine_gimbal_arm_len=stage1_gimbal_arm,
    dry_com_z=15,
    prop_com_z=20,
)

# Environment
environment = Environment()
mu = environment.gravitational_constant * environment.earth_mass

# Launch site parameters
launch_latitude_deg = 28.5  # Cape Canaveral
launch_latitude_rad = np.deg2rad(launch_latitude_deg)

# Initial position
cos_lat = np.cos(launch_latitude_rad)
sin_lat = np.sin(launch_latitude_rad)
initial_position = environment.earth_radius * np.array([cos_lat, 0.0, sin_lat])

# Initial velocity
omega_cross_r = np.cross(environment.earth_angular_velocity_vector, initial_position)

# Initial quaternion: align body Z with local vertical (radial unit vector)
radial_unit_vector = initial_position / np.linalg.norm(initial_position)
initial_quaternion = compute_body_z_to_inertial_quat(radial_unit_vector)
# Initial kick east during pitch program
kick_direction = rotate_body_to_inertial_by_quat(np.array([0, 1, 0]), initial_quaternion)

# Find orbital normal
horizontal_projection = kick_direction - np.dot(kick_direction, radial_unit_vector) * radial_unit_vector
horizontal_unit = horizontal_projection / np.linalg.norm(horizontal_projection)
orbital_normal = np.cross(radial_unit_vector, horizontal_unit)
orbital_normal /= np.linalg.norm(orbital_normal)

# State
initial_state = State(
    position=initial_position,
    velocity=omega_cross_r,
    quaternion=initial_quaternion,
    angular_velocity=np.array([0, 0, 0]),
    propellant_mass=stage1_ascent_prop,
)

# Set up phase timing
pitch_start_time = 10.0

# Phases
stage1_phases = [
    TimeBasedGuidancePhase(end_time=pitch_start_time, attitude_mode="radial", throttle=1.0, name="Initial Ascent"),
    ProgrammedPitchGuidancePhase(
        start_time=pitch_start_time,
        end_time=burnout_time,
        initial_pitch_deg=90,
        final_pitch_deg=23.1,
        orbital_normal=orbital_normal,
        kick_direction=kick_direction,
        throttle=1.0,
        name="Stage 1 Pitch Program",
    ),
]

# Mission Planner
stage1_planner = MissionPlanner(
    guidance_phases=stage1_phases, environment=environment, vehicle=stage_1, start_time=0.0
)

# Controller gains
stage1_p = 6e3
stage1_i = 0.1
stage1_d = 2e5

# Controller
ascent_controller = PIDAttitudeController(
    kp=np.array([stage1_p, stage1_p, 1.5 * stage1_p]),
    ki=np.array([stage1_i, stage1_i, 1.5 * stage1_i]),
    kd=np.array([stage1_d, stage1_d, 1.5 * stage1_d]),
    vehicle=stage_1,
)

# Simulator
stage1_sim = Simulator(
    vehicle=stage_1,
    environment=environment,
    initial_state=initial_state,
    mission_planner=stage1_planner,
    t_0=0,
    t_final=burnout_time,
    delta_t=0.1,
    log_interval=5,
    log_name="orbit",
)
stage1_sim.add_controller(ascent_controller)

print(f"Simulating Ascent...")
stage1_t_vals, stage1_state_vals, stage1_phase_transitions = stage1_sim.run()

"""
STAGE 2
"""

# Separation
burnout_state_vector = stage1_state_vals[-1]
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

target_apo_alt = 300000.0
target_peri_alt = 200000.0
target_apoapsis = target_apo_alt + environment.earth_radius
target_periapsis = target_peri_alt + environment.earth_radius
simulation_end_time = separation_time + 5500
burnout_quaternion = burnout_state_vector[6:10]
burnout_position = burnout_state_vector[:3]
burnout_z_unit_vector = rotate_body_to_inertial_by_quat(np.array([0, 0, 1]), burnout_quaternion)
burnout_radial_unit_vector = burnout_position / np.linalg.norm(burnout_position)
burnout_dot = np.dot(burnout_z_unit_vector, burnout_radial_unit_vector)
burnout_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(burnout_dot, -1.0, 1.0)))

stage2_phases = [
    PEGGuidancePhase(
        target_apoapsis=target_apoapsis,
        target_periapsis=target_periapsis,
        orbital_normal=orbital_normal,
        vehicle=stage_2,
        mu=mu,
        target_inclination=None,
        apo_tolerance=10000.0,
        peri_tolerance=20000.0,
        min_throttle=0.1,
        throttle_kp=20.0,
        throttle_threshold_factor=5.0,
        throttle=1.0,
        name="Stage 2 Ascent Burn",
    ),
    CoastGuidancePhase(
        vehicle=stage_2,
        mu=mu,
        time_to_apo_threshold=30.0,
        attitude_mode="prograde",
        name="Coast",
        buffer=5.0,
        use_dynamic_threshold=False,
    ),
    CircBurnGuidancePhase(
        attitude_mode="prograde",
        throttle=1.0,
        name="Circularization",
        min_throttle=0.1,
        throttle_kp=20.0,
        target_eccentricity=0.0011,
    ),
    TimeBasedGuidancePhase(end_time=simulation_end_time, attitude_mode="passive", throttle=0.0, name="Orbit"),
]

stage2_planner = MissionPlanner(
    guidance_phases=stage2_phases, environment=environment, vehicle=stage_2, start_time=separation_time
)

stage2_p = 3e3
stage2_i = 0
stage2_d = 2e4

controller_stage2 = PIDAttitudeController(
    kp=np.array([stage2_p, stage2_p, 2 * stage2_p]),
    ki=np.array([stage2_i, stage2_i, 2 * stage2_i]),
    kd=np.array([stage2_d, stage2_d, 2 * stage2_d]),
    vehicle=stage_2,
)

sim_stage2 = Simulator(
    vehicle=stage_2,
    environment=environment,
    initial_state=current_state,
    mission_planner=stage2_planner,
    t_0=separation_time,
    t_final=simulation_end_time,  # Enough for orbit
    delta_t=0.2,
    log_interval=0.5,
    log_name="orbit",
)
sim_stage2.add_controller(controller_stage2)

print("\nSimulating Stage 2 to Orbit...")
stage2_t_vals, stage2_state_vals, stage2_phase_transitions = sim_stage2.run()

# Combine phase transitions for plotting
all_phase_transitions = [(t, f"{name}") for t, name in stage1_phase_transitions] + [
    (t, f"{name}") for t, name in stage2_phase_transitions
]

# Combine t and state vals for plotting
all_t_vals = np.append(stage1_t_vals, stage2_t_vals)
all_state_vals = np.vstack((stage1_state_vals, stage2_state_vals))

plot_3D_integration_segments(
    t_vals=all_t_vals,
    state_vals=all_state_vals,
    phase_transitions=all_phase_transitions,
    show_earth=True,
)
