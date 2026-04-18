"""
Monte Carlo simulation service for running dispersed orbital launch simulations.
"""

from typing import Any, Dict, List

import numpy as np

from app.models.simulation_models import SimulationRequest
from app.runners.simulation_runner import run_full_orbit_simulation


class MonteCarloRunner:
    """
    Monte Carlo dispersion analysis for launch simulations.
    """

    def __init__(self):
        """Initialize the Monte Carlo runner."""
        pass

    def run_monte_carlo(
        self,
        base_request: SimulationRequest,
        num_simulations: int,
        dispersions: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """
        Run multiple simulations with parameter dispersions.

        Args:
            base_request: Base SimulationRequest to build off of
            num_simulations: Number of individual simulations in this batch
            dispersions: Dict mapping parameter names to {mean, std_dev}
                Example: {"stage1_base_thrust_magnitude": {"mean": 7600000, "std_dev": 100000}}

        Returns:
            List of simulation results
        """
        results = []

        for i in range(num_simulations):
            try:
                # Create a dispersed copy of the request and run simulation.
                dispersed_request = self._create_dispersed_request(
                    base_request, dispersions
                )
                result = run_full_orbit_simulation(dispersed_request)
                result["simulation_index"] = i + 1
                result["dispersions_applied"] = self._get_applied_dispersions(
                    base_request, dispersed_request
                )
                results.append(result)
            except Exception as e:
                results.append(
                    {"simulation_index": i + 1, "error": str(e), "success": False}
                )

        return results

    def _create_dispersed_request(
        self, base_request: SimulationRequest, dispersions: Dict[str, Dict[str, float]]
    ) -> SimulationRequest:
        """
        Create a new request with dispersed parameters sampled from normal distributions.

        Args:
            base_request: Base request to modify
            dispersions: Parameter dispersions

        Returns:
            New SimulationRequest with dispersed values
        """
        # Convert request to dict
        request_dict = base_request.dict()

        # Apply dispersions
        for param_name, dispersion_spec in dispersions.items():
            if param_name in request_dict:
                mean = dispersion_spec.get("mean", request_dict[param_name])
                std_dev = dispersion_spec.get("std_dev", 0)

                # Sample from normal distribution
                dispersed_value = np.random.normal(mean, std_dev)
                request_dict[param_name] = dispersed_value

        # Create new request from modified dict
        return SimulationRequest(**request_dict)

    def _get_applied_dispersions(
        self, base_request: SimulationRequest, modified_request: SimulationRequest
    ) -> Dict[str, float]:
        """
        Extract which dispersions were actually applied (non-zero differences).

        Args:
            base_request: Original request
            modified_request: Modified request

        Returns:
            Dict of parameter_name: dispersed_value
        """
        base_dict = base_request.dict()
        modified_dict = modified_request.dict()

        applied = {}
        for key in base_dict:
            if base_dict[key] != modified_dict[key]:
                applied[key] = modified_dict[key]

        return applied

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistical summary of Monte Carlo results.

        Args:
            results: List of simulation results

        Returns:
            Dictionary of aggregated statistics
        """
        successful_runs = [
            r
            for r in results
            if not r.get("error") and isinstance(r.get("summary"), dict)
        ]
        failed_runs = [
            r
            for r in results
            if r.get("error") or not isinstance(r.get("summary"), dict)
        ]

        if not successful_runs:
            return {
                "success_rate": 0.0,
                "total_simulations": len(results),
                "completed_simulations": 0,
                "failed_simulations": len(failed_runs),
                "error": "All runs failed",
            }

        # Extract key metrics
        altitudes = self._get_valid_array(successful_runs, "final_altitude_km")
        apoapsis = self._get_valid_array(successful_runs, "apoapsis_altitude_km")
        periapsis = self._get_valid_array(successful_runs, "periapsis_altitude_km")
        eccentricity = self._get_valid_array(successful_runs, "eccentricity")

        return {
            "total_simulations": len(results),
            "completed_simulations": len(successful_runs),
            "failed_simulations": len(failed_runs),
            "success_rate": len(successful_runs) / len(results),
            "final_altitude_km": self._construct_statistics_dict(altitudes),
            "apoapsis_altitude_km": self._construct_statistics_dict(apoapsis),
            "periapsis_altitude_km": self._construct_statistics_dict(periapsis),
            "eccentricity": self._construct_statistics_dict(eccentricity),
            "failure_modes": [r.get("error", "Unknown") for r in failed_runs],
        }

    def _get_valid_array(self, results: List[Dict[str, Any]], key: str) -> np.ndarray:
        """
        Helper to extract a valid numpy array for a given summary key across results.

        Args:
            results: List of simulation results
            key: Summary key to extract

        Returns:
            Numpy array of valid values for the specified key
        """
        return np.array(
            [
                r["summary"][key]
                for r in results
                if r.get("summary")
                and r["summary"].get(key) is not None
                and np.isfinite(r["summary"][key])
            ]
        )

    def _construct_statistics_dict(self, array: np.ndarray) -> Dict[str, Any]:
        """
        Construct a dictionary of statistics from the Monte Carlo results.

        Args:
            array: Numpy array of simulation results

        Returns:
            Dictionary of statistics
        """
        return (
            {
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "percentile_5": float(np.percentile(array, 5)),
                "percentile_95": float(np.percentile(array, 95)),
            }
            if len(array) > 0
            else {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "percentile_5": None,
                "percentile_95": None,
            }
        )
