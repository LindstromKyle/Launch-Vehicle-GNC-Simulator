import numpy as np
import pytest

from simulator.environment import Environment
from simulator.vehicle import Vehicle


@pytest.fixture
def basic_environment():
    return Environment()


@pytest.fixture
def basic_vehicle():
    # Minimal Vehicle subclass for testing (overrides abstract methods)
    class TestVehicle(Vehicle):
        def _setup_propulsion_system(self):
            self.num_engines = 1
            self.thrust_per_engine = self.base_thrust_magnitude
            self.mdot_per_engine = self.mdot_max
            self.engines = [{"position": np.array([0.0, 0.0, -self.engine_lever_arm])}]
            self.rcs_thrusters = []

    return TestVehicle(
        dry_mass=1000.0,
        initial_prop_mass=500.0,
        base_thrust_magnitude=10000.0,
        average_isp=300.0,
        moment_of_inertia=np.eye(3) * 1000,
        base_drag_coefficient=0.5,
        drag_scaling_coefficient=0.1,
        cross_sectional_area=10.0,
        engine_gimbal_limit_deg=10.0,
        engine_gimbal_arm_len=5.0,
        dry_com_z=0.0,
        prop_com_z=-2.0,
    )
