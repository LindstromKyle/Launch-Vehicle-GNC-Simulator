from typing import Union

import numpy as np
import re
import matplotlib.pyplot as plt

from plotting import plot_3D_integration_segments


def parse_log_to_structured_array(filename):
    data = []
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("[INFO] time (s):"):
            record = {}

            # Parse MISSION PLANNER
            mission_line = lines[i][6:].strip()  # Skip [INFO]
            parts = [p.strip() for p in mission_line.split("|")]
            record["time"] = float(re.search(r"time \(s\): ([-.\deE]+)", parts[0]).group(1))
            record["phase"] = re.search(r"phase: (.*)", parts[1]).group(1).strip()

            i += 1
            alt_line = lines[i][6:].strip()
            parts = [p.strip() for p in alt_line.split("|")]
            record["current_altitude"] = float(re.search(r"current altitude \(km\): ([-.\deE]+)", parts[0]).group(1))
            record["apoapsis_altitude"] = float(re.search(r"apoapsis altitude \(km\): ([-.\deE]+)", parts[1]).group(1))
            record["periapsis_altitude"] = float(
                re.search(r"periapsis altitude \(km\): ([-.\deE]+)", parts[2]).group(1)
            )

            i += 1
            vel_line = lines[i][6:].strip()
            parts = [p.strip() for p in vel_line.split("|")]
            record["orbital_vel"] = float(re.search(r"orbital vel \(km/s\): ([-.\deE]+)", parts[0]).group(1))
            record["tangential_vel"] = float(re.search(r"tangential vel \(km/s\): ([-.\deE]+)", parts[1]).group(1))
            record["radial_vel"] = float(re.search(r"radial vel \(km/s\): ([-.\deE]+)", parts[2]).group(1))

            # Skip to GUIDANCE
            while i < len(lines) and "GUIDANCE" not in lines[i]:
                i += 1
            i += 1  # Now at attitude mode line

            if i >= len(lines):
                break
            att_mode_line = lines[i][6:].strip()
            record["attitude_mode"] = re.search(r"attitude mode: (.*)", att_mode_line).group(1).strip()

            i += 1  # Now at current quat line
            quat_line = lines[i][6:].strip()
            parts = [p.strip() for p in quat_line.split("|")]
            quat_str = re.search(r"current quat: \[(.*?)\]", parts[0]).group(1)
            record["current_quat"] = np.fromstring(quat_str, sep=" ")
            att_str = re.search(r"current attitude \(z_hat\): \[(.*?)\]", parts[1]).group(1)
            record["attitude"] = np.fromstring(att_str, sep=" ")

            i += 1
            des_quat_line = lines[i][6:].strip()
            parts = [p.strip() for p in des_quat_line.split("|")]
            des_quat_str = re.search(r"desired quat: \[(.*?)\]", parts[0]).group(1)
            record["desired_quat"] = np.fromstring(des_quat_str, sep=" ")
            des_att_str = re.search(r"desired attitude \(z_hat\): \[(.*?)\]", parts[1]).group(1)
            record["desired_attitude"] = np.fromstring(des_att_str, sep=" ")

            i += 1
            err_line = lines[i][6:].strip()
            parts = [p.strip() for p in err_line.split("|")]
            err_quat_str = re.search(r"error quat: \[(.*?)\]", parts[0]).group(1)
            record["error_quat"] = np.fromstring(err_quat_str, sep=" ")
            err_att_str = re.search(r"error attitude \(z_hat\): \[(.*?)\]", parts[1]).group(1)
            record["error_attitude"] = np.fromstring(err_att_str, sep=" ")

            i += 1
            current_pitch_line = lines[i][6:].strip()
            record["current_pitch"] = float(
                re.search(r"current pitch \(deg\): ([-.\deE]+)", current_pitch_line).group(1)
            )

            i += 1
            desired_pitch_line = lines[i][6:].strip()
            record["desired_pitch"] = float(
                re.search(r"desired pitch \(deg\): ([-.\deE]+)", desired_pitch_line).group(1)
            )

            i += 1
            pitch_error_line = lines[i][6:].strip()
            parts = [p.strip() for p in pitch_error_line.split("|")]
            record["pitch_error"] = float(re.search(r"pitch error \(deg\): ([-.\deE]+)", parts[0]).group(1))
            record["error_angle"] = float(re.search(r"quat error angle \(deg\): ([-.\deE]+)", parts[1]).group(1))

            # Skip to CONTROLLER
            while i < len(lines) and "CONTROLLER" not in lines[i]:
                i += 1
            i += 1  # Now at body frame error line

            if i >= len(lines):
                break
            body_error_line = lines[i][6:].strip()
            body_err_str = re.search(r"body frame error \(deg\): \[(.*?)\]", body_error_line).group(1)
            record["body_frame_error"] = np.fromstring(body_err_str, sep=" ")

            i += 1
            pid_line = lines[i][6:].strip()
            parts = [p.strip() for p in pid_line.split("|")]
            p_str = re.search(r"PID p term: \[(.*?)\]", parts[0]).group(1)
            record["pid_p"] = np.fromstring(p_str, sep=" ")
            i_str = re.search(r"PID i term: \[(.*?)\]", parts[1]).group(1)
            record["pid_i"] = np.fromstring(i_str, sep=" ")
            d_str = re.search(r"PID d term: \[(.*?)\]", parts[2]).group(1)
            record["pid_d"] = np.fromstring(d_str, sep=" ")

            i += 1
            torque_line = lines[i][6:].strip()
            parts = [p.strip() for p in torque_line.split("|")]
            des_torque_str = re.search(r"desired torque \(N\*m\): \[(.*?)\]", parts[0]).group(1)
            record["desired_torque"] = np.fromstring(des_torque_str, sep=" ")
            record["throttle"] = float(re.search(r"throttle: ([-.\deE]+)", torque_line).group(1))

            i += 1
            app_line = lines[i][6:].strip()
            parts = [p.strip() for p in app_line.split("|")]
            app_torque_str = re.search(r"applied torque \(N\*m\): \[(.*?)\]", parts[0]).group(1)
            record["applied_torque"] = np.fromstring(app_torque_str, sep=" ")
            ang_vel_str = re.search(r"ang vel \(rad/s\): \[(.*?)\]", parts[1]).group(1)
            record["ang_vel"] = np.fromstring(ang_vel_str, sep=" ")
            ang_acc_str = re.search(r"ang acc \(rad/s/s\): \[(.*?)\]", parts[2]).group(1)
            record["ang_acc"] = np.fromstring(ang_acc_str, sep=" ")

            i += 1
            new_torque_line = lines[i][6:].strip()
            parts = [p.strip() for p in new_torque_line.split("|")]
            tvt_str = re.search(r"thrust vector torque: \[(.*?)\]", parts[0]).group(1)
            record["thrust_vector_torque"] = np.fromstring(tvt_str, sep=" ")
            rcs_str = re.search(r"rcs torque: \[(.*?)\]", parts[1]).group(1)
            record["rcs_torque"] = np.fromstring(rcs_str, sep=" ")

            # Parse gimbal angles (3 lines)
            gimbal_angles = np.zeros((9, 2))
            indices_list = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
            i += 1  # First gimbal line
            for j in range(3):
                gimbal_line = lines[i][6:].strip()
                angle_strs = re.findall(r"\[(.*?)\]", gimbal_line)
                for k, angle_str in enumerate(angle_strs):
                    gimbal_angles[indices_list[j][k]] = np.fromstring(angle_str, sep=" ")
                i += 1

            record["engine_gimbal_angles"] = gimbal_angles

            # Now at [DYNAMICS]
            while i < len(lines) and "DYNAMICS" not in lines[i]:
                i += 1
            i += 1  # Now at pos line

            if i >= len(lines):
                break
            pos_line = lines[i][6:].strip()
            parts = [p.strip() for p in pos_line.split("|")]
            pos_str = re.search(r"pos \(m\): \[(.*?)\]", parts[0]).group(1)
            record["pos"] = np.fromstring(pos_str, sep=" ")
            vel_str = re.search(r"vel \(m/s\): \[(.*?)\]", parts[1]).group(1)
            record["vel"] = np.fromstring(vel_str, sep=" ")
            acc_str = re.search(r"acc \(m/s/s\): \[(.*?)\]", parts[2]).group(1)
            record["acc"] = np.fromstring(acc_str, sep=" ")

            i += 1
            force_line = lines[i][6:].strip()
            parts = [p.strip() for p in force_line.split("|")]
            thrust_str = re.search(r"thrust \(N\): \[(.*?)\]", parts[0]).group(1)
            record["thrust"] = np.fromstring(thrust_str, sep=" ")
            drag_str = re.search(r"drag \(N\): \[(.*?)\]", parts[1]).group(1)
            record["drag"] = np.fromstring(drag_str, sep=" ")
            gravity_str = re.search(r"gravity \(N\): \[(.*?)\]", parts[2]).group(1)
            record["gravity"] = np.fromstring(gravity_str, sep=" ")
            net_str = re.search(r"net force \(N\): \[(.*?)\]", parts[3]).group(1)
            record["net_force"] = np.fromstring(net_str, sep=" ")

            i += 1
            mass_line = lines[i][6:].strip()
            parts = [p.strip() for p in mass_line.split("|")]
            record["total_mass"] = float(re.search(r"total mass \(kg\): ([-.\deE]+)", parts[0]).group(1))
            record["propellant_mass"] = float(re.search(r"propellant mass \(kg\): ([-.\deE]+)", parts[1]).group(1))
            record["mass_flow"] = float(re.search(r"mass flow \(kg/s\): ([-.\deE]+)", parts[2]).group(1))

            data.append(record)

        i += 1

    # Define dtype
    dtype = [
        ("time", np.float64),
        ("phase", "U50"),
        ("attitude_mode", "U50"),
        ("current_altitude", np.float64),
        ("apoapsis_altitude", np.float64),
        ("periapsis_altitude", np.float64),
        ("orbital_vel", np.float64),
        ("tangential_vel", np.float64),
        ("radial_vel", np.float64),
        ("current_quat", np.float64, (4,)),
        ("attitude", np.float64, (3,)),
        ("desired_quat", np.float64, (4,)),
        ("desired_attitude", np.float64, (3,)),
        ("error_quat", np.float64, (4,)),
        ("error_attitude", np.float64, (3,)),
        ("current_pitch", np.float64),
        ("desired_pitch", np.float64),
        ("pitch_error", np.float64),
        ("error_angle", np.float64),
        ("body_frame_error", np.float64, (3,)),
        ("pid_p", np.float64, (3,)),
        ("pid_i", np.float64, (3,)),
        ("pid_d", np.float64, (3,)),
        ("desired_torque", np.float64, (3,)),
        ("throttle", np.float64),
        ("applied_torque", np.float64, (3,)),
        ("thrust_vector_torque", np.float64, (3,)),
        ("rcs_torque", np.float64, (3,)),
        ("ang_vel", np.float64, (3,)),
        ("ang_acc", np.float64, (3,)),
        ("engine_gimbal_angles", np.float64, (9, 2)),
        ("pos", np.float64, (3,)),
        ("vel", np.float64, (3,)),
        ("acc", np.float64, (3,)),
        ("thrust", np.float64, (3,)),
        ("drag", np.float64, (3,)),
        ("gravity", np.float64, (3,)),
        ("net_force", np.float64, (3,)),
        ("total_mass", np.float64),
        ("propellant_mass", np.float64),
        ("mass_flow", np.float64),
    ]

    # Create structured array
    structured_data = np.zeros(len(data), dtype=dtype)
    for idx, rec in enumerate(data):
        for field in dtype:
            name = field[0]
            structured_data[name][idx] = rec[name]

    return structured_data


def plot_six(time, y_list, labels=None):
    """
    Plots six arrays in a 3x2 grid, with time on x-axis and y values on y-axis.
    All subplots share the x-axis.

    Parameters:
    - time: numpy array, the common x-axis values
    - y_list: list of 6 numpy arrays, the y-values for each plot
    - labels: optional list of 6 strings, labels for each subplot
    """
    if len(y_list) != 6:
        raise ValueError("y_list must contain exactly 6 arrays")
    if labels is None:
        labels = [f"Plot {i+1}" for i in range(6)]
    elif len(labels) != 6:
        raise ValueError("labels must contain exactly 6 strings")

    fig, axs = plt.subplots(3, 2, figsize=(15, 7.5), sharex=True)

    # Left column: first 3
    for i in range(3):
        if len(y_list[i].shape) > 1:
            axs[i, 0].plot(time, y_list[i], label=["X", "Y", "Z"])
            axs[i, 0].legend(fontsize=6)
        else:
            axs[i, 0].plot(time, y_list[i])
        axs[i, 0].set_title(labels[i])
        axs[i, 0].set_xlabel("Time")
        axs[i, 0].tick_params(labelbottom=True)

    # Right column: next 3
    for i in range(3):
        if len(y_list[i + 3].shape) > 1:
            axs[i, 1].plot(time, y_list[i + 3], label=["X", "Y", "Z"])
            axs[i, 1].legend(fontsize=6)
        else:
            axs[i, 1].plot(time, y_list[i + 3])
        axs[i, 1].set_title(labels[i + 3])
        axs[i, 1].set_xlabel("Time")
        axs[i, 1].tick_params(labelbottom=True)

    plt.tight_layout()
    plt.show()


def plot_gimbal_angles(time, engine_gimbal_angles):
    """
    Plots gimbal angles for 9 engines in a 3x3 grid mimicking engine positions viewed from CoM looking down (-Z).

    Parameters:
    - time: numpy array, the time values for the x-axis
    - engine_gimbal_angles: numpy array of shape (N, 9, 2), where N is number of time steps,
      9 is engines (0-7 outer counterclockwise from +X, 8 center), and 2 is [x, y] angles (e.g., pitch, yaw) in degrees.
    """
    if engine_gimbal_angles.shape[1] != 9 or engine_gimbal_angles.shape[2] != 2:
        raise ValueError("engine_gimbal_angles must be of shape (N, 9, 2)")

    fig, axs = plt.subplots(3, 3, figsize=(15, 7.5), sharex=True, sharey=True)

    # Mapping of engine index to (row, col) in 3x3 grid
    # Viewed from CoM down (-Z): X right (+col), Y up (-row)
    # Engine 0: +X (mid-right: row1,col2)
    # Counterclockwise: 1: top-right (0,2), 2: top-mid (0,1), 3: top-left (0,0),
    # 4: mid-left (1,0), 5: bottom-left (2,0), 6: bottom-mid (2,1), 7: bottom-right (2,2)
    # 8: center (1,1)
    engine_positions = [
        (1, 2),  # Engine 0
        (0, 2),  # 1
        (0, 1),  # 2
        (0, 0),  # 3
        (1, 0),  # 4
        (2, 0),  # 5
        (2, 1),  # 6
        (2, 2),  # 7
        (1, 1),  # 8 (center)
    ]

    for engine_idx, (row, col) in enumerate(engine_positions):
        ax = axs[row, col]
        angles = engine_gimbal_angles[:, engine_idx, :]  # (N, 2)

        ax.plot(time, angles[:, 0], label="X", color="blue")
        ax.plot(time, angles[:, 1], label="Y", color="orange")

        ax.set_title(f"Engine {engine_idx + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)
        ax.legend(fontsize=6)
        ax.grid(True)

    # Hide unused axes if any (but here all 9 are used)
    plt.tight_layout()
    plt.show()


def plot_exhaust_flow_directions(time_value, structured_data, exaggerate_factor):
    """
    Generates a 3D matplotlib plot showing the exhaust directions for 9 engines at the specified time step.

    Args:
        exaggerate_factor (float): Amount to exaggerate gimbal angles
        time_value (float): The simulation time (s) to plot.
        structured_data (np.ndarray): The structured array from parse_log_to_structured_array().
    """
    # Find the index of the closest time step
    times = structured_data["time"]
    idx = np.argmin(np.abs(times - time_value))
    print(f"Plotting for closest time: {times[idx]:.2f} s")

    # Get gimbal angles in degrees (9 engines x [pitch, yaw])
    gimbal_angles_deg = structured_data["engine_gimbal_angles"][idx]
    applied_torque = structured_data["applied_torque"][idx]

    # Convert to radians and exaggerate
    gimbal_angles_rad = np.deg2rad(gimbal_angles_deg * exaggerate_factor)

    # Define engine positions in body frame (xy plane at z=0, Falcon 9-like arrangement)
    engine_radius = 1.5  # From vehicle.py
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    positions = [np.array([engine_radius * np.cos(t), engine_radius * np.sin(t), 0.0]) for t in theta]
    positions.append(np.array([0.0, 0.0, 0.0]))  # Center engine
    positions = np.array(positions)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    arrow_length = 1.0  # Arbitrary length for visualization (m)

    for i in range(9):
        # Gimbal angles: [pitch, yaw] in rad
        pitch_rad = gimbal_angles_rad[i, 0]
        yaw_rad = gimbal_angles_rad[i, 1]

        # Thrust direction in body frame (from vehicle.py)
        thrust_dir_body = np.array([np.sin(yaw_rad), -np.sin(pitch_rad), np.cos(pitch_rad) * np.cos(yaw_rad)])
        thrust_dir_body /= np.linalg.norm(thrust_dir_body)  # Normalize

        # Exhaust direction is opposite to thrust (exhaust flows away from rocket)
        exhaust_dir_body = -thrust_dir_body

        # Position
        pos = positions[i]

        # Plot arrow
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            exhaust_dir_body[0] * arrow_length,
            exhaust_dir_body[1] * arrow_length,
            exhaust_dir_body[2] * arrow_length,
            color="blue",
            linewidth=1.5,
            arrow_length_ratio=0.2,
        )

        # Annotation at tail
        ax.text(pos[0], pos[1], pos[2], f"E{i+1}", color="red", fontsize=10)

    # Set axis limits
    max_r = engine_radius * 1.2
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-arrow_length * 1.2, 0.2)  # Since exhaust ~ -Z

    # Labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Set view: +X right, +Y up, +Z out (viewing along -Z)
    ax.view_init(elev=90, azim=-90)

    # Title
    plt.title(f"Engine Exhaust Directions at t = {times[idx]:.2f} s \n" f"Applied Torque (N*m): {applied_torque}")
    plt.show()


def standard_plot_vs_time(field_names: list, structured_array: np.ndarray, y_axis_name: str, title: str):
    fig, ax = plt.subplots()

    for field_name in field_names:
        ax.plot(structured_array["time"], structured_array[field_name], label=field_name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_axis_name)
    ax.set_title(f"{title}")
    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    array = parse_log_to_structured_array("orbit.log")

    time = array["time"]
    y_list = [
        array["desired_torque"],
        array["applied_torque"],
        array["net_force"],
        array["error_angle"],
        array["ang_vel"],
        array["vel"],
    ]
    labels = [
        "desired_torque (X)",
        "applied_torque (X)",
        "net force (X)",
        "error_angle",
        "ang_vel (X)",
        "vel",
    ]

    # plot_six(time, y_list, labels)

    # plot_gimbal_angles(array["time"], array["engine_gimbal_angles"])

    plot_exhaust_flow_directions(15, array, exaggerate_factor=30000)

    # standard_plot_vs_time(["desired_pitch", "current_pitch"], array, "Pitch Angle (deg)", "Pitch Angle vs Time")

    # standard_plot_vs_time(["current_altitude", "orbital_vel"], array)

    # standard_plot_vs_time(["radial_vel", "tangential_vel", "orbital_vel"], array, "Velocity (m/s)", "Velocity vs Time")

    # standard_plot_vs_time(
    #     ["apoapsis_altitude", "periapsis_altitude", "current_altitude"], array, "Altitude (km)", "Altitude vs Time"
    # )
