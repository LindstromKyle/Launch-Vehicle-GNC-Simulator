from concurrent.futures import ThreadPoolExecutor
from typing import Any

from starlette.requests import HTTPConnection

from app.runners.monte_carlo_runner import MonteCarloRunner
from app.settings import Settings
from app.storage.live_telemetry_storage import LiveTelemetryStorage


def get_settings(connection: HTTPConnection) -> Settings:
    return connection.app.state.settings


def get_executor(connection: HTTPConnection) -> ThreadPoolExecutor:
    return connection.app.state.executor


def get_monte_carlo_storage(connection: HTTPConnection) -> Any:
    return connection.app.state.mc_storage


def get_monte_carlo_runner(connection: HTTPConnection) -> MonteCarloRunner:
    return connection.app.state.mc_runner


def get_live_telemetry_storage(connection: HTTPConnection) -> LiveTelemetryStorage:
    return connection.app.state.live_telemetry_storage
