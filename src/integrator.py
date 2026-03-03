import logging

from dynamics import calculate_dynamics
import numpy as np
from tqdm import tqdm


def integrate_rk4(
    vehicle, environment, initial_state, t_0, t_final, delta_t, log_interval, controller, mission_planner
):

    # Initialize starting time and position
    t_values = [t_0]
    state_values = [initial_state]
    current_time = t_0
    current_state = initial_state.copy()
    last_logged_time = None
    phase_transitions = []  # Collect transitions if planner provided

    # Progress bar
    max_iterations = np.floor((t_final - t_0) / delta_t)
    p_bar = tqdm(total=max_iterations, desc="Processing", leave=True)

    while current_time < t_final:

        # Log dynamic variables every [log_interval] seconds of simulation time
        log_flag = False
        if last_logged_time is None or round(current_time - last_logged_time, 12) >= log_interval:
            log_flag = True
            last_logged_time = current_time

        # Calculate step size, ensure we don’t overshoot
        h = min(delta_t, t_final - current_time)

        # Mission planner setpoints
        setpoints = mission_planner.update(current_time, current_state, log_flag)
        # Controller update
        controls = controller.update(current_time, current_state, setpoints, log_flag)

        # Calculate intermediate slopes, logging k_1 every [log_interval] seconds
        k_1 = calculate_dynamics(
            time=current_time,
            state=current_state,
            vehicle=vehicle,
            environment=environment,
            log_flag=log_flag,
            controls=controls,
        )
        k_2 = calculate_dynamics(
            time=current_time + h / 2,
            state=current_state + k_1 / 2,
            vehicle=vehicle,
            environment=environment,
            log_flag=False,
            controls=controls,
        )
        k_3 = calculate_dynamics(
            time=current_time + h / 2,
            state=current_state + k_2 / 2,
            vehicle=vehicle,
            environment=environment,
            log_flag=False,
            controls=controls,
        )
        k_4 = calculate_dynamics(
            time=current_time + h,
            state=current_state + k_3,
            vehicle=vehicle,
            environment=environment,
            log_flag=False,
            controls=controls,
        )

        # Update state and time
        weighted_average = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        # If we hit the ground, don't change state
        if np.linalg.norm(current_state[:3]) >= environment.earth_radius:
            current_state += h * weighted_average

        current_time += h

        # Clamp propellant
        current_state[13] = max(current_state[13], 0)

        # Normalize quaternion to prevent drift
        current_state[6:10] /= np.linalg.norm(current_state[6:10])

        # Append state and time to results
        t_values.append(current_time)
        state_values.append(current_state.copy())

        phase_transitions = mission_planner.get_phase_transitions()

        # Progress bar
        p_bar.update(1)

    p_bar.close()
    return np.array(t_values), np.array(state_values), phase_transitions


def integrate_verlet(
    vehicle,
    environment,
    initial_state,
    t_0,
    t_final,
    delta_t,
    log_interval,
    controller,
    mission_planner,
):
    """
    Pure Velocity Verlet (translation) + exact quaternion propagation (rotation).
    - Symplectic energy conservation on orbit (no 100 km decay).
    - Quaternion norm preserved exactly every step.
    - Only 2 dynamics calls per step → fast and simple.
    - Drop-in replacement for your old hybrid.
    """
    import numpy as np
    from tqdm import tqdm
    from dynamics import calculate_dynamics
    from utils import quat_to_angle_axis, angle_axis_to_quat, quaternion_multiply

    t_vals = [t_0]
    state_vals = [initial_state.copy()]
    current_time = t_0
    current_state = initial_state.copy()
    last_logged_time = None
    phase_transitions = []

    max_iterations = int(np.floor((t_final - t_0) / delta_t))
    p_bar = tqdm(total=max_iterations, desc="Simple Verlet (energy-preserving)", leave=True)

    while current_time < t_final:
        log_flag = last_logged_time is None or round(current_time - last_logged_time, 12) >= log_interval
        if log_flag:
            last_logged_time = current_time

        h = min(delta_t, t_final - current_time)

        # === 1. Get controls and current acceleration/torque ===
        setpoints = mission_planner.update(current_time, current_state, log_flag)
        controls = controller.update(current_time, current_state, setpoints, log_flag)

        deriv_current = calculate_dynamics(
            time=current_time,
            state=current_state,
            vehicle=vehicle,
            environment=environment,
            log_flag=log_flag,
            controls=controls,
        )
        acc_current = deriv_current[3:6]
        ang_acc_current = deriv_current[10:13]

        # === 2. Half-step velocity and angular velocity (Velocity Verlet style) ===
        current_state[3:6] += 0.5 * acc_current * h
        current_state[10:13] += 0.5 * ang_acc_current * h

        # === 3. Full position update ===
        current_state[:3] += current_state[3:6] * h

        # === 4. Exact quaternion update (constant ω over the step) ===
        omega = current_state[10:13]
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-12:
            angle_axis = np.zeros(4)
            angle_axis[0] = omega_norm * h
            angle_axis[1:] = omega / omega_norm
            delta_q = angle_axis_to_quat(angle_axis)
            current_state[6:10] = quaternion_multiply(delta_q, current_state[6:10])
        # else: ω ≈ 0 → quaternion unchanged (perfect for coast)

        # Normalize (safety + exact preservation)
        qn = np.linalg.norm(current_state[6:10])
        if qn > 1e-12:
            current_state[6:10] /= qn

        # Propellant (linear, exact)
        current_state[13] += deriv_current[13] * h
        current_state[13] = max(current_state[13], 0.0)

        # === 5. Recompute forces/torques with NEW position + NEW attitude ===
        deriv_new = calculate_dynamics(
            time=current_time + h,
            state=current_state,
            vehicle=vehicle,
            environment=environment,
            log_flag=False,
            controls=controls,
        )
        acc_new = deriv_new[3:6]
        ang_acc_new = deriv_new[10:13]

        # === 6. Finish velocity and angular velocity ===
        current_state[3:6] += 0.5 * acc_new * h
        current_state[10:13] += 0.5 * ang_acc_new * h

        current_time += h

        t_vals.append(current_time)
        state_vals.append(current_state.copy())
        phase_transitions = mission_planner.get_phase_transitions()

        p_bar.update(1)

    p_bar.close()
    return np.array(t_vals), np.array(state_vals), phase_transitions
