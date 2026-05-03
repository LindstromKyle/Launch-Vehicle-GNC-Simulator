from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LiveRun:
    run_id: str
    status: str
    created_at: str
    run_kind: str = "single_orbit"
    allowed_vehicle_ids: set[str] = field(default_factory=set)
    started_at: str | None = None
    finished_at: str | None = None
    latest_seq: int = 0
    error: str | None = None
    summary: dict[str, Any] | None = None
    latest_vehicle_time_s: dict[str, float] = field(default_factory=dict)
    pending_commands: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=500)
    )
    command_audit: list[dict[str, Any]] = field(default_factory=list)
    frames: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5000))


class LiveTelemetryStorage:
    """Thread-safe in-memory store for live simulation telemetry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, LiveRun] = {}

    def create_run(
        self,
        run_kind: str = "single_orbit",
        allowed_vehicle_ids: list[str] | None = None,
    ) -> str:
        run_id = str(uuid4())
        with self._lock:
            self._runs[run_id] = LiveRun(
                run_id=run_id,
                status="queued",
                created_at=_utc_now_iso(),
                run_kind=run_kind,
                allowed_vehicle_ids=set(allowed_vehicle_ids or []),
            )
        return run_id

    def mark_running(self, run_id: str) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.status = "running"
            run.started_at = _utc_now_iso()

    def mark_completed(
        self, run_id: str, summary: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.status = "completed"
            run.finished_at = _utc_now_iso()
            run.summary = summary

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.status = "failed"
            run.finished_at = _utc_now_iso()
            run.error = error

    def append_frame(self, run_id: str, frame: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            run = self._require_run(run_id)
            run.latest_seq += 1
            frame_with_meta = {
                "seq": run.latest_seq,
                "emitted_at": _utc_now_iso(),
                **frame,
            }
            run.frames.append(frame_with_meta)
            return frame_with_meta

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._require_run(run_id)
            return self._status_payload(run)

    def get_frames_after(
        self, run_id: str, after_seq: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        with self._lock:
            run = self._require_run(run_id)
            frames = [f for f in run.frames if int(f["seq"]) > after_seq][:limit]
            return {
                "run_id": run.run_id,
                "status": run.status,
                "after_seq": after_seq,
                "latest_seq": run.latest_seq,
                "frames": [dict(f) for f in frames],
            }

    def enqueue_deorbit_command(
        self,
        run_id: str,
        vehicle_id: str,
        execute_at_sim_time_s: float,
        target_perigee_alt_km: float,
    ) -> dict[str, Any]:
        with self._lock:
            received_at = _utc_now_iso()
            run = self._runs.get(run_id)
            if run is None:
                return {
                    "command_id": "",
                    "status": "rejected",
                    "reason": f"Live run not found: {run_id}",
                    "server_received_at": received_at,
                }

            if run.run_kind != "constellation":
                return {
                    "command_id": "",
                    "status": "rejected",
                    "reason": "Commands are only enabled for constellation runs",
                    "server_received_at": received_at,
                }

            if vehicle_id not in run.allowed_vehicle_ids:
                return {
                    "command_id": "",
                    "status": "rejected",
                    "reason": f"Unknown vehicle_id '{vehicle_id}' for run",
                    "server_received_at": received_at,
                }

            latest_time_s = run.latest_vehicle_time_s.get(vehicle_id, 0.0)
            if execute_at_sim_time_s < latest_time_s:
                return {
                    "command_id": "",
                    "status": "rejected",
                    "reason": (
                        "execute_at_sim_time_s is in the past for this vehicle "
                        f"(latest={latest_time_s:.3f}s)"
                    ),
                    "server_received_at": received_at,
                }

            command_id = str(uuid4())
            command_record = {
                "command_id": command_id,
                "status": "queued",
                "action": "deorbit_burn",
                "run_id": run_id,
                "vehicle_id": vehicle_id,
                "execute_at_sim_time_s": float(execute_at_sim_time_s),
                "target_perigee_alt_km": float(target_perigee_alt_km),
                "server_received_at": received_at,
                "executed_at_sim_time_s": None,
            }
            run.pending_commands.append(command_record)
            run.command_audit.append(dict(command_record))

            return {
                "command_id": command_id,
                "status": "accepted",
                "reason": None,
                "server_received_at": received_at,
            }

    def pop_due_commands(
        self,
        run_id: str,
        vehicle_id: str,
        sim_time_s: float,
    ) -> list[dict[str, Any]]:
        with self._lock:
            run = self._require_run(run_id)
            run.latest_vehicle_time_s[vehicle_id] = max(
                run.latest_vehicle_time_s.get(vehicle_id, 0.0),
                float(sim_time_s),
            )

            due: list[dict[str, Any]] = []
            remaining: deque[dict[str, Any]] = deque(maxlen=run.pending_commands.maxlen)

            for command in run.pending_commands:
                is_due = command["vehicle_id"] == vehicle_id and float(
                    command["execute_at_sim_time_s"]
                ) <= float(sim_time_s)
                if is_due:
                    updated = dict(command)
                    updated["status"] = "executed"
                    updated["executed_at_sim_time_s"] = float(sim_time_s)
                    due.append(updated)
                    self._update_audit_command(
                        run,
                        command_id=str(updated["command_id"]),
                        status="executed",
                        executed_at_sim_time_s=float(sim_time_s),
                    )
                else:
                    remaining.append(command)

            run.pending_commands = remaining
            due.sort(key=lambda cmd: float(cmd["execute_at_sim_time_s"]))
            return due

    def _status_payload(self, run: LiveRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "status": run.status,
            "run_kind": run.run_kind,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "latest_seq": run.latest_seq,
            "error": run.error,
        }

    def _update_audit_command(
        self,
        run: LiveRun,
        command_id: str,
        status: str,
        executed_at_sim_time_s: float | None = None,
    ) -> None:
        for idx, entry in enumerate(run.command_audit):
            if str(entry.get("command_id")) != command_id:
                continue
            updated = dict(entry)
            updated["status"] = status
            if executed_at_sim_time_s is not None:
                updated["executed_at_sim_time_s"] = executed_at_sim_time_s
            run.command_audit[idx] = updated
            break

    def _require_run(self, run_id: str) -> LiveRun:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Live run not found: {run_id}")
        return run
