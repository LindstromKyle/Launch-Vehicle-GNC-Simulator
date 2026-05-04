"""Multi-satellite orbital simulation runner.

Simulates three W-series capsules in distinct LEO orbital planes concurrently,
emitting telemetry frames tagged with ``vehicle_id`` for each satellite.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from opentelemetry import context as otel_context

from app.observability import get_tracer
from simulator.controller import Controller, PIDAttitudeController
from simulator.environment import Environment
from simulator.guidance import (
    AwaitCommandGuidancePhase,
    BallisticReentryGuidancePhase,
    DeorbitBurnGuidancePhase,
    ParachuteDescentGuidancePhase,
    TimeBasedGuidancePhase,
)
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

_T_FINAL = 8_000.0  # Final sim time (s)
_DELTA_T = 1.0  # Integration step size (s)
_TELEMETRY_INTERVAL = 10.0  # Minimum seconds between emitted frames
_STEP_DELAY_S = 0.05  # Real-time sleep between emitted frames (seconds)
_TRACER = get_tracer()


@dataclass
class DeorbitCommandState:
    armed: bool = False
    command_id: str | None = None
    target_perigee_alt_km: float | None = None


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
    command_provider: Callable[[str, float], list[dict[str, Any]]] | None,
    command_event_callback: Callable[[dict[str, Any]], None] | None,
    command_state: DeorbitCommandState,
) -> Callable[[dict[str, Any]], None]:
    """Return a wrapper that stamps telemetry and updates command-driven state."""

    phase_to_event = {
        "Deorbit Burn": "burn_started",
        "Ballistic Reentry": "reentry_started",
        "Parachute Descent": "parachute_deployed",
        "Landed": "landed",
    }
    last_phase: str | None = None

    def _tagged(frame: dict[str, Any]) -> None:
        nonlocal last_phase
        frame["vehicle_id"] = vehicle_id
        base_callback(frame)

        sim_time_s = float(frame.get("time_s", 0.0))

        if command_provider is not None and command_event_callback is not None:
            due_commands = command_provider(vehicle_id, sim_time_s)
            for command in due_commands:
                command_state.armed = True
                command_state.command_id = str(command.get("command_id"))
                command_state.target_perigee_alt_km = float(
                    command.get("target_perigee_alt_km", 0.0)
                )

                command_event_callback(
                    {
                        "event_type": "command",
                        "event_name": "command_armed",
                        "action": command.get("action"),
                        "command_id": command_state.command_id,
                        "vehicle_id": vehicle_id,
                        "execute_at_sim_time_s": command.get("execute_at_sim_time_s"),
                        "executed_at_sim_time_s": sim_time_s,
                        "target_perigee_alt_km": command_state.target_perigee_alt_km,
                    }
                )

        current_phase = str(frame.get("phase") or "")
        if (
            command_event_callback is not None
            and current_phase != last_phase
            and current_phase in phase_to_event
        ):
            command_event_callback(
                {
                    "event_type": "command",
                    "event_name": phase_to_event[current_phase],
                    "vehicle_id": vehicle_id,
                    "command_id": command_state.command_id,
                    "time_s": sim_time_s,
                    "phase": current_phase,
                    "target_perigee_alt_km": command_state.target_perigee_alt_km,
                }
            )
        last_phase = current_phase

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
    command_provider: Callable[[str, float], list[dict[str, Any]]] | None = None,
    command_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Simulate a single satellite and stream its telemetry."""
    env = Environment()
    vehicle = WSeriesCapsule()
    vehicle.parachute_deployed = False

    command_state = DeorbitCommandState()

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
        prop_mass=vehicle.initial_propellant_mass,
    )

    phases = [
        AwaitCommandGuidancePhase(
            command_armed_fn=lambda: command_state.armed,
            name="Orbit Await Command",
        ),
        DeorbitBurnGuidancePhase(
            command_armed_fn=lambda: command_state.armed,
            target_perigee_alt_km_fn=lambda: command_state.target_perigee_alt_km,
            environment=env,
            burn_throttle=0.35,
            periapsis_tolerance_m=3_000.0,
            max_burn_duration_s=600.0,
            name="Deorbit Burn",
        ),
        BallisticReentryGuidancePhase(
            environment=env,
            parachute_deploy_alt_m=25_000.0,
            name="Ballistic Reentry",
        ),
        ParachuteDescentGuidancePhase(
            vehicle=vehicle,
            environment=env,
            landing_alt_threshold_m=100.0,
            name="Parachute Descent",
        ),
        TimeBasedGuidancePhase(
            end_time=_T_FINAL,
            attitude_mode="passive",
            throttle=0.0,
            name="Landed",
        ),
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
    sim.add_controller(
        PIDAttitudeController(
            kp=np.array([120.0, 120.0, 80.0]),
            ki=np.array([0.0, 0.0, 0.0]),
            kd=np.array([1500.0, 1500.0, 1000.0]),
            vehicle=vehicle,
        )
    )

    tagged_cb = _make_tagged_callback(
        name,
        telemetry_callback,
        command_provider,
        command_event_callback,
        command_state,
    )
    sim.run(telemetry_callback=tagged_cb, telemetry_interval=_TELEMETRY_INTERVAL)


def _run_satellite_with_span(
    parent_ctx: otel_context.Context,
    name: str,
    perigee_alt_km: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    arg_perigee_deg: float,
    true_anomaly_deg: float,
    telemetry_callback: Callable[[dict[str, Any]], None],
    command_provider: Callable[[str, float], list[dict[str, Any]]] | None = None,
    command_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    token = otel_context.attach(parent_ctx)
    try:
        with _TRACER.start_as_current_span(
            f"constellation.run.satellite.{name}"
        ) as span:
            span.set_attribute("vehicle_id", name)
            span.set_attribute("orbit.perigee_alt_km", perigee_alt_km)
            span.set_attribute("orbit.eccentricity", eccentricity)
            span.set_attribute("orbit.inclination_deg", inclination_deg)
            _run_satellite(
                name,
                perigee_alt_km,
                eccentricity,
                inclination_deg,
                raan_deg,
                arg_perigee_deg,
                true_anomaly_deg,
                telemetry_callback,
                command_provider,
                command_event_callback,
            )
    finally:
        otel_context.detach(token)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_constellation_simulation(
    telemetry_callback: Callable[[dict[str, Any]], None],
    command_provider: Callable[[str, float], list[dict[str, Any]]] | None = None,
    command_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Simulate the full W-series constellation, running satellites in parallel.

    Each satellite's telemetry frames are tagged with ``vehicle_id`` so the
    frontend can route them to the correct 3D trace and 2D chart.

    Args:
        telemetry_callback: Called for every telemetry frame from every satellite.
    """
    parent_ctx = otel_context.get_current()
    with ThreadPoolExecutor(max_workers=len(CONSTELLATION)) as pool:
        futures = {
            pool.submit(
                _run_satellite_with_span,
                parent_ctx,
                name,
                perigee_alt_km,
                eccentricity,
                inc_deg,
                raan_deg,
                arg_perigee_deg,
                ta_deg,
                telemetry_callback,
                command_provider,
                command_event_callback,
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
                tb = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                logging.error("Satellite %s simulation failed:\n%s", name, tb)
                raise RuntimeError(
                    f"Satellite {name} simulation failed: {exc}"
                ) from exc
