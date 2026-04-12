"""
Simple JSON-based storage for Monte Carlo simulation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import uuid


class MonteCarloStorage:
    """
    Handles persistence of Monte Carlo results to JSON files.
    """

    def __init__(self, storage_dir: str = None):
        """
        Args:
            storage_dir: Directory to store MC results. Defaults to ./mc_results/
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent.parent / "mc_results"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_batch(
        self,
        total_simulations: int,
        base_params: Dict[str, Any],
        dispersions: Dict[str, Dict[str, float]] | None = None,
    ) -> str:
        """
        Create a new Monte Carlo batch record with status "in_progress".

        Args:
            total_simulations: Number of individual simulations to perform
            base_params: The base simulation parameters
            dispersions: Parameter dispersions applied, if any

        Returns:
            batch_id: Unique identifier for this batch
        """
        batch_id = str(uuid.uuid4())

        batch_data = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "total_simulations": total_simulations,
            "base_params": base_params,
            "dispersions": dispersions or {},
            "simulations": [],
            "status": "in_progress",
            "summary": None,
        }

        self._save_batch(batch_id, batch_data)
        return batch_id

    def finalize_batch(
        self,
        batch_id: str,
        simulations: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        """
        Update a Monte Carlo batch with completed results and statistics.

        Args:
            batch_id: The Monte Carlo batch identifier
            simulations: Full list of simulation results
            summary: Dictionary of summary statistics
        """
        batch_data = self._load_batch(batch_id)
        batch_data["simulations"] = simulations
        batch_data["status"] = "completed"
        batch_data["summary"] = summary
        self._save_batch(batch_id, batch_data)

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Retrieve a Monte Carlo batch by ID."""
        return self._load_batch(batch_id)

    def list_batches(self) -> List[Dict[str, Any]]:
        """List all Monte Carlo batches with basic info."""
        batches = []
        for json_file in self.storage_dir.glob("*.json"):
            data = self._load_batch(json_file.stem)
            batches.append(
                {
                    "batch_id": data["batch_id"],
                    "created_at": data["created_at"],
                    "status": data["status"],
                    "total_simulations": data["total_simulations"],
                    "completed_simulations": len(data["simulations"]),
                }
            )
        return sorted(batches, key=lambda x: x["created_at"], reverse=True)

    def _save_batch(self, batch_id: str, data: Dict) -> None:
        """Save batch data to JSON file."""
        filepath = self.storage_dir / f"{batch_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _load_batch(self, batch_id: str) -> Dict:
        """Load batch data from JSON file."""
        filepath = self.storage_dir / f"{batch_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"MC batch {batch_id} not found")
        with open(filepath, "r") as f:
            return json.load(f)
