"""PostgreSQL storage for Monte Carlo batch persistence."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import psycopg

from app.settings import Settings


class MonteCarloStorage:
    """Persists Monte Carlo batches in PostgreSQL using direct SQL."""

    def __init__(self):
        settings = Settings()
        self._connection_string = settings.db_connection_string
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with psycopg.connect(self._connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monte_carlo_batches (
                        batch_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        total_simulations INT NOT NULL,
                        base_params JSONB NOT NULL,
                        dispersions JSONB NOT NULL,
                        simulations JSONB NOT NULL DEFAULT '[]'::jsonb,
                        status TEXT NOT NULL,
                        summary JSONB
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_monte_carlo_batches_created_at
                    ON monte_carlo_batches (created_at DESC)
                    """
                )
            conn.commit()

    def create_batch(
        self,
        total_simulations: int,
        base_params: Dict[str, Any],
        dispersions: Dict[str, Dict[str, float]] | None = None,
    ) -> str:
        batch_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        with psycopg.connect(self._connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO monte_carlo_batches (
                        batch_id,
                        created_at,
                        total_simulations,
                        base_params,
                        dispersions,
                        simulations,
                        status,
                        summary
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                    """,
                    (
                        batch_id,
                        created_at,
                        total_simulations,
                        json.dumps(base_params),
                        json.dumps(dispersions or {}),
                        json.dumps([]),
                        "in_progress",
                        json.dumps(None),
                    ),
                )
            conn.commit()

        return batch_id

    def finalize_batch(
        self,
        batch_id: str,
        simulations: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        with psycopg.connect(self._connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE monte_carlo_batches
                    SET simulations = %s::jsonb,
                        status = %s,
                        summary = %s::jsonb
                    WHERE batch_id = %s
                    """,
                    (
                        json.dumps(simulations),
                        "completed",
                        json.dumps(summary),
                        batch_id,
                    ),
                )
            conn.commit()

    def mark_batch_failed(
        self,
        batch_id: str,
        simulations: List[Dict[str, Any]],
        error_msg: str,
    ) -> None:
        with psycopg.connect(self._connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE monte_carlo_batches
                    SET simulations = %s::jsonb,
                        status = %s,
                        summary = %s::jsonb
                    WHERE batch_id = %s
                    """,
                    (
                        json.dumps(simulations),
                        "failed",
                        json.dumps({"error": error_msg}),
                        batch_id,
                    ),
                )
            conn.commit()

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        with psycopg.connect(self._connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        batch_id,
                        created_at,
                        total_simulations,
                        base_params,
                        dispersions,
                        simulations,
                        status,
                        summary
                    FROM monte_carlo_batches
                    WHERE batch_id = %s
                    """,
                    (batch_id,),
                )
                row = cur.fetchone()

        if row is None:
            raise FileNotFoundError(f"Monte Carlo batch {batch_id} not found")

        return {
            "batch_id": row[0],
            "created_at": row[1].isoformat(),
            "total_simulations": row[2],
            "base_params": row[3],
            "dispersions": row[4],
            "simulations": row[5],
            "status": row[6],
            "summary": row[7],
        }

    def list_batches(self) -> List[Dict[str, Any]]:
        with psycopg.connect(self._connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        batch_id,
                        created_at,
                        status,
                        total_simulations,
                        jsonb_array_length(simulations) AS completed_simulations
                    FROM monte_carlo_batches
                    ORDER BY created_at DESC
                    """
                )
                rows = cur.fetchall()

        return [
            {
                "batch_id": row[0],
                "created_at": row[1].isoformat(),
                "status": row[2],
                "total_simulations": row[3],
                "completed_simulations": row[4],
            }
            for row in rows
        ]
