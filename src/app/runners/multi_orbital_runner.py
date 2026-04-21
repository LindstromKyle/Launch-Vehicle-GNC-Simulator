"""Multi-satellite orbital simulation runner.

Simulates three W-series capsules in distinct LEO orbital planes concurrently,
emitting telemetry frames tagged with ``vehicle_id`` for each satellite.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from simulator.controller import Controller
from simulator.environment import Environment
from simulator.guidance import TimeBasedGuidancePhase
from simulator.mission import MissionPlanner
from simulator.simulator import Simulator
from simulator.utils import orbital_elements_to_state
from simulator.vehicle import WSeriesCapsule

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0
_MU = 3.986004418e14  # m³/s²  standard gravitational parameter

# ---------------------------------------------------------------------------
# Constellation definition
# (name, perigee_alt_km, eccentricity, inclination_deg, raan_deg, arg_perigee_deg, true_anomaly_deg)
# ---------------------------------------------------------------------------

CONSTELLATION: list[tuple[str, float, float, float, float, float, float]] = [
    ("W-4", 291.962, 0.0400635, 97.4615, 222.2763, 283.6748, 76.4488),
    ("W-6", 520.156, 0.0001664, 97.4483, 67.1823, 308.0006, 52.1227),
    ("W-7", 500.0, 0.020, 15.0, 240.0, 290.0, 260.0),
]

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

_T_FINAL = 6_000.0  # ~one full LEO orbital period (s)
_DELTA_T = 5.0  # Integration step size (s)
_TELEMETRY_INTERVAL = 10.0  # Minimum seconds between emitted frames
_STEP_DELAY_S = 0.05  # Real-time sleep between emitted frames (seconds)


class PassiveOrbitController(Controller):
    """No-op controller for passive orbital vehicles.

    The integrator expects a controller to always exist, so this controller
    returns zero actuator commands for thrust-vector and RCS channels.
    """

    def update(
        self,
        time: float,
        state_vector: np.ndarray,
        desired_quaternion: np.ndarray,
        throttle: float,
        log_flag: bool,
        attitude_mode: str,
    ) -> dict[str, Any]:
        _ = (time, state_vector, desired_quaternion, throttle, log_flag, attitude_mode)
        # Keep one neutral gimbal entry so verbose dynamics logging can index safely.
        return {
            "engine_gimbal_angles": [np.array([0.0, 0.0])],
            "rcs_levels": np.array([]),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_tagged_callback(
    vehicle_id: str,
    base_callback: Callable[[dict[str, Any]], None],
) -> Callable[[dict[str, Any]], None]:
    """Return a wrapper that stamps every frame with ``vehicle_id``."""

    def _tagged(frame: dict[str, Any]) -> None:
        frame["vehicle_id"] = vehicle_id
        base_callback(frame)
        if _STEP_DELAY_S > 0:
            time.sleep(_STEP_DELAY_S)

    return _tagged


def _run_satellite(
    name: str,
    perigee_alt_km: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    arg_perigee_deg: float,
    true_anomaly_deg: float,
    telemetry_callback: Callable[[dict[str, Any]], None],
) -> None:
    """Simulate a single W-series satellite and stream its telemetry."""
    env = Environment()
    vehicle = WSeriesCapsule()

    # Build semi-major axis from perigee radius so each orbit can have unique eccentricity.
    r_perigee = _EARTH_RADIUS_M + perigee_alt_km * 1_000.0
    a = r_perigee / (1.0 - eccentricity)

    initial_state = orbital_elements_to_state(
        semi_major_axis=a,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        arg_perigee_deg=arg_perigee_deg,
        true_anomaly_deg=true_anomaly_deg,
        mu=_MU,
        prop_mass=0.0,
    )

    # Single coast phase covering the full simulation window
    phases = [
        TimeBasedGuidancePhase(
            end_time=_T_FINAL,
            attitude_mode="passive",
            throttle=0.0,
            name="Orbit",
        )
    ]

    planner = MissionPlanner(phases, env, vehicle, start_time=0.0)

    sim = Simulator(
        vehicle=vehicle,
        environment=env,
        initial_state=initial_state,
        mission_planner=planner,
        t_0=0.0,
        t_final=_T_FINAL,
        delta_t=_DELTA_T,
        log_interval=300.0,
        log_name=f"constellation_{name.lower().replace('-', '_')}",
    )
    sim.add_controller(PassiveOrbitController())

    tagged_cb = _make_tagged_callback(name, telemetry_callback)
    sim.run(telemetry_callback=tagged_cb, telemetry_interval=_TELEMETRY_INTERVAL)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_constellation_simulation(
    telemetry_callback: Callable[[dict[str, Any]], None],
) -> None:
    """Simulate the full W-series constellation, running satellites in parallel.

    Each satellite's telemetry frames are tagged with ``vehicle_id`` so the
    frontend can route them to the correct 3D trace and 2D chart.

    Args:
        telemetry_callback: Called for every telemetry frame from every satellite.
    """
    with ThreadPoolExecutor(max_workers=len(CONSTELLATION)) as pool:
        futures = {
            pool.submit(
                _run_satellite,
                name,
                perigee_alt_km,
                eccentricity,
                inc_deg,
                raan_deg,
                arg_perigee_deg,
                ta_deg,
                telemetry_callback,
            ): name
            for (
                name,
                perigee_alt_km,
                eccentricity,
                inc_deg,
                raan_deg,
                arg_perigee_deg,
                ta_deg,
            ) in CONSTELLATION
        }

        for future in as_completed(futures):
            name = futures[future]
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(
                    f"Satellite {name} simulation failed: {exc}"
                ) from exc
