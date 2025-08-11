import matplotlib.pyplot as plt
import numpy as np

from utils import compute_acceleration, quat_to_angle_axis


def plot_3D_trajectory(t_vals, state_vals):
    """

    Args:
        t_vals ():
        state_vals ():

    Returns:

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = state_vals[:, 0] / 1000
    y_vals = state_vals[:, 1] / 1000
    z_vals = state_vals[:, 2] / 1000

    ax.plot3D(
        x_vals,
        y_vals,
        z_vals,
        label="Rocket trajectory",
        linewidth=2,
        color="dodgerblue",
    )
    ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], color="green", label="Launch", s=50)
    ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], color="red", label="Final point", s=50)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("3D Rocket Trajectory")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_1D_position_velocity_acceleration(t_vals, state_vals, axis, environment):

    # Plot altitude vs time
    fig, axs = plt.subplots(3)

    position_dict = {"X": 0, "Y": 1, "Z": 2}
    velocity_dict = {"X": 3, "Y": 4, "Z": 5}

    altitude = state_vals[:, position_dict[axis]] - environment.earth_radius
    axs[0].plot(t_vals, altitude)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Altitude Above Surface (m)")
    axs[0].set_title(f"Altitude vs Time ({axis})")
    axs[0].grid()

    velocity = state_vals[:, velocity_dict[axis]]
    axs[1].plot(t_vals, velocity)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title(f"Velocity vs Time ({axis})")
    axs[1].grid()
    axs[1].sharex(axs[0])

    acceleration = compute_acceleration(t_vals, velocity)
    axs[2].plot(t_vals, acceleration)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Acceleration (m/s^2)")
    axs[2].set_title(f"Acceleration vs Time ({axis})")
    axs[2].grid()
    axs[2].sharex(axs[0])

    # drag_vals = np.array([
    #     np.linalg.norm(environment.drag_force(p, v, vehicle))
    #     for p, v in zip(state_vals[:, 2], state_vals[:, 5])
    # ])
    # axs[2].plot(t_vals, drag_vals)
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Acceleration (m/s^2)")
    # axs[2].set_title("Acceleration vs Time")
    # axs[2].grid()
    # axs[2].sharex(axs[0])

    plt.tight_layout()
    plt.show()


def plot_3D_integration_segments(t_vals, state_vals, phase_transitions=None, show_earth=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Define colors for phases
    phase_colors = {
        "Initial Ascent": "blue",
        "Kick": "green",
        "Stage 1 Pitch Program": "yellow",
        "Stage 2 Ascent Burn": "orange",
        "Coast": "purple",
        "Circularization": "red",
        "Orbit": "black",
        "Semi-Gravity Turn Stage 1": "cyan",
    }

    x_vals = state_vals[:, 0] / 1000
    y_vals = state_vals[:, 1] / 1000
    z_vals = state_vals[:, 2] / 1000

    if phase_transitions:
        # Segment by phase transitions
        for phase_index, (start_time, phase_name) in enumerate(phase_transitions):
            end_time = (
                phase_transitions[phase_index + 1][0] if phase_index + 1 < len(phase_transitions) else t_vals[-1]
            )

            mask = (t_vals >= start_time) & (t_vals < end_time)
            if np.any(mask):

                ax.plot3D(
                    x_vals[mask],
                    y_vals[mask],
                    z_vals[mask],
                    label=phase_name,
                    linewidth=2,
                    color=phase_colors.get(phase_name, "gray"),
                )
    else:
        # Fallback: Plot whole segment
        ax.plot3D(
            x_vals,
            y_vals,
            z_vals,
            label="Trajectory",
            linewidth=2,
            color="dodgerblue",
        )

    if show_earth:
        # Add Earth as a low-resolution wireframe sphere for efficiency
        earth_radius = 6371  # km
        u = np.linspace(0, 2 * np.pi, 40)  # Low res: 20 longitude points
        v = np.linspace(0, np.pi, 20)  # Low res: 10 latitude points
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="blue", rstride=2, cstride=2)  # Strides skip lines for even less lag

        # Optional: For a solid sphere (potentially laggier), replace plot_wireframe with:
        # ax.plot_surface(x, y, z, color='lightblue', alpha=1)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("3D Rocket Trajectory by Phase")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pitch_angle(t_vals, state_vals, guidance):

    fig, ax = plt.subplots()
    quats = state_vals[:, 6:10]
    pitches = []
    for quat in quats:
        angle_axis = quat_to_angle_axis(quat)
        angle = np.degrees(angle_axis[0])
        pitches.append(angle)
    ax.plot(t_vals, pitches)
    ax.axhline(guidance.kick_angle_deg, color="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Degrees)")
    ax.set_title("Pitch vs Time")
    ax.grid()
    plt.show()
