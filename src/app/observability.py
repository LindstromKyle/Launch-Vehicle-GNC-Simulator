from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from time import process_time

import psutil

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from prometheus_client import Counter, Gauge, generate_latest

_LOGGER_NAME = "app.observability"
_TRACER_NAME = "app.constellation"
_TRACE_BOOTSTRAPPED = False
_PROCESS = psutil.Process()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object]
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {"message": str(record.getMessage())}

        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        # Prefix level for easy terminal scanning while retaining JSON payload.
        return f"{record.levelname}: {json.dumps(payload)}"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def bootstrap_observability(otlp_endpoint: str = "") -> None:
    global _TRACE_BOOTSTRAPPED
    _get_logger()

    if _TRACE_BOOTSTRAPPED:
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": "launch-simulator-api"}),
    )
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACE_BOOTSTRAPPED = True


def get_tracer():
    return trace.get_tracer(_TRACER_NAME)


def log_event(level: int, event: str, **fields: object) -> None:
    logger = _get_logger()
    logger.log(level, {"event": event, **fields})


CONSTELLATION_RUN_OUTCOME_TOTAL = Counter(
    "constellation_run_outcome_total",
    "Constellation run outcomes.",
    labelnames=("status",),
)

APP_PROCESS_CPU_SECONDS_TOTAL = Gauge(
    "app_process_cpu_seconds_total",
    "CPU seconds consumed by the simulator API process.",
)

APP_PROCESS_RESIDENT_MEMORY_BYTES = Gauge(
    "app_process_resident_memory_bytes",
    "Resident memory size of the simulator API process in bytes.",
)


def increment_constellation_run_outcome(status: str) -> None:
    CONSTELLATION_RUN_OUTCOME_TOTAL.labels(status=status).inc()


def observe_process_metrics() -> None:
    APP_PROCESS_CPU_SECONDS_TOTAL.set(process_time())
    APP_PROCESS_RESIDENT_MEMORY_BYTES.set(float(_PROCESS.memory_info().rss))


def render_metrics_text() -> bytes:
    observe_process_metrics()
    return generate_latest()
