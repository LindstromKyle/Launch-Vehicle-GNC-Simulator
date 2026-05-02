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
    started_at: str | None = None
    finished_at: str | None = None
    latest_seq: int = 0
    error: str | None = None
    summary: dict[str, Any] | None = None
    frames: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5000))


class LiveTelemetryStorage:
    """Thread-safe in-memory store for live simulation telemetry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, LiveRun] = {}

    def create_run(self) -> str:
        run_id = str(uuid4())
        with self._lock:
            self._runs[run_id] = LiveRun(
                run_id=run_id,
                status="queued",
                created_at=_utc_now_iso(),
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

    def _status_payload(self, run: LiveRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "status": run.status,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "latest_seq": run.latest_seq,
            "error": run.error,
            "summary": run.summary,
        }

    def _require_run(self, run_id: str) -> LiveRun:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Live run not found: {run_id}")
        return run
