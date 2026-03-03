import numpy as np

from utils import rotate_vector_by_quaternion


class Environment:
    def __init__(self):
        self.earth_radius = 6371000
        self.earth_mass = 5.972e24
        self.gravitational_constant = 6.67430e-11
        self.earth_rotation_rate = 7.292115e-5  # rad/s (sidereal rotation)
        self.earth_angular_velocity_vector = np.array([0.0, 0.0, self.earth_rotation_rate])  # inertial

    def gravitational_force(self, position, vehicle_mass):
        orbital_radius = np.linalg.norm(position)
        if orbital_radius < 1e-3:
            return np.zeros(3)

        # Pure Newtonian
        newtonian_force = -self.gravitational_constant * self.earth_mass * vehicle_mass * position / orbital_radius**3

        # J2 perturbation (oblateness)
        j_2 = 1.08263e-3  # Earth's J2 coefficient
        x, y, z = position
        factor = (3 * j_2 * self.gravitational_constant * self.earth_mass * self.earth_radius**2) / (
            2 * orbital_radius**5
        )
        dx = factor * (5 * z**2 / orbital_radius**2 - 1) * x
        dy = factor * (5 * z**2 / orbital_radius**2 - 1) * y
        dz = factor * (5 * z**2 / orbital_radius**2 - 3) * z
        j2_perturbation = vehicle_mass * np.array([dx, dy, dz])  # Acceleration * mass = force

        return newtonian_force + j2_perturbation

    def atmospheric_density(self, altitude: float) -> float:
        sea_level_density = 1.225
        scale_height = 8500
        return sea_level_density * np.exp(-1 * altitude / scale_height)

    def drag_force(self, position, velocity, vehicle, quaternion):

        altitude = np.linalg.norm(position) - self.earth_radius

        # Protect against divide by zero
        if altitude < 0:
            altitude = 0

        density = self.atmospheric_density(altitude)

        # Compute atmospheric velocity due to Earth's rotation
        omega_cross_r = np.cross(self.earth_angular_velocity_vector, position)
        # Relative velocity (for accurate airspeed)
        relative_velocity = velocity - omega_cross_r
        relative_velocity_magnitude = np.linalg.norm(relative_velocity)

        # Protect against divide by zero
        if relative_velocity_magnitude < 1e-3:
            return np.zeros(3)

        # Direction of drag force (opposite of velocity)
        drag_unit_vector = -relative_velocity / relative_velocity_magnitude

        # Transform body frame Z axis to inertial frame
        body_frame_z_axis = np.array([0, 0, 1])
        inertial_frame_z_axis = rotate_vector_by_quaternion(body_frame_z_axis, quaternion)

        # Compute angle of attack (radians)
        cos_alpha = np.dot(relative_velocity, inertial_frame_z_axis) / relative_velocity_magnitude
        # Clamp for numerical safety
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        angle_of_attack = np.arccos(cos_alpha)

        # Adjust drag coefficient based on AoA
        total_drag_coefficient = (
            vehicle.base_drag_coefficient + vehicle.drag_scaling_coefficient * np.sin(angle_of_attack) ** 2
        )

        # Compute drag magnitude
        drag_magnitude = (
            0.5 * density * relative_velocity_magnitude**2 * total_drag_coefficient * vehicle.cross_sectional_area
        )

        return drag_magnitude * drag_unit_vector

    def aerodynamic_torque(self, position, velocity, quaternion, angular_velocity, vehicle):
        # Get control surface deflections
        deflections = vehicle.get_grid_fin_deflections(time=None, state=None)

        # Eventually add angle of attack math here for cross-sectional area
        # Placeholder: return zero for now
        return np.zeros(3)
