import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_batch(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_sim_rows(simulations: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for sim in simulations:
        summary = sim.get("summary") or {}
        row: Dict[str, float] = {
            "simulation_index": sim.get("simulation_index", np.nan),
            "final_altitude_km": summary.get("final_altitude_km", np.nan),
            "apoapsis_altitude_km": summary.get("apoapsis_altitude_km", np.nan),
            "periapsis_altitude_km": summary.get("periapsis_altitude_km", np.nan),
            "semi_major_axis_km": summary.get("semi_major_axis_km", np.nan),
            "eccentricity": summary.get("eccentricity", np.nan),
            "inclination": summary.get("inclination", np.nan),
        }

        dispersions_applied = sim.get("dispersions_applied") or {}
        for k, v in dispersions_applied.items():
            row[k] = v

        # Hyperbolic/parabolic flag for quick classification.
        ecc = row["eccentricity"]
        sma = row["semi_major_axis_km"]
        apo = row["apoapsis_altitude_km"]
        row["is_escape_orbit"] = float(
            (np.isfinite(ecc) and ecc >= 1.0)
            or (np.isfinite(sma) and sma <= 0.0)
            or (not np.isfinite(apo))
        )

        rows.append(row)
    return rows


def get_dispersion_keys(batch: Dict[str, Any]) -> List[str]:
    dispersions = batch.get("dispersions") or {}
    return list(dispersions.keys())


def to_array(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def plot_metric_distributions(rows: List[Dict[str, float]], batch_id: str) -> None:
    metrics = [
        ("final_altitude_km", "Final Altitude (km)"),
        ("apoapsis_altitude_km", "Apoapsis Altitude (km)"),
        ("periapsis_altitude_km", "Periapsis Altitude (km)"),
        ("eccentricity", "Eccentricity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Monte Carlo Metric Distributions | Batch {batch_id}")

    for ax, (key, label) in zip(axes.flat, metrics):
        vals = finite(to_array(rows, key))
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
            ax.set_title(label)
            continue

        # bins = min(20, max(8, int(np.sqrt(vals.size))))
        bins = 40
        ax.hist(vals, bins=bins, alpha=0.8, edgecolor="black")

        ax.axvline(
            np.mean(vals),
            linestyle="--",
            linewidth=1.5,
            label=f"mean={np.mean(vals):.3g}",
        )
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()


def plot_dispersion_sensitivity(
    rows: List[Dict[str, float]],
    dispersion_key: str,
    batch_id: str,
) -> None:
    x = to_array(rows, dispersion_key)
    y_apo = to_array(rows, "apoapsis_altitude_km")
    y_peri = to_array(rows, "periapsis_altitude_km")
    y_ecc = to_array(rows, "eccentricity")
    escape = to_array(rows, "is_escape_orbit") > 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"Sensitivity to {dispersion_key} | Batch {batch_id}")

    plots = [
        (y_apo, "Apoapsis Altitude (km)"),
        (y_peri, "Periapsis Altitude (km)"),
        (y_ecc, "Eccentricity"),
    ]

    for ax, (y, y_label) in zip(axes, plots):
        normal_mask = np.isfinite(x) & np.isfinite(y) & (~escape)
        escape_mask = np.isfinite(x) & np.isfinite(y) & escape

        ax.scatter(x[normal_mask], y[normal_mask], s=26, alpha=0.8)
        if np.any(escape_mask):
            ax.scatter(
                x[escape_mask],
                y[escape_mask],
                s=28,
                alpha=0.85,
                marker="x",
                label="Escape/hyperbolic",
            )

        ax.set_xlabel(dispersion_key)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)

    fig.tight_layout()


if __name__ == "__main__":
    batch_file = Path(
        "C:\\Users\\linds\\code\\Launch-Vehicle-GNC-Simulator\\src\\mc_results\\80831c57-a485-4ecd-b8f9-493a8b6b75ab.json"
    )

    batch = load_batch(batch_file)
    batch_id = batch.get("batch_id", "unknown_batch")
    simulations = batch.get("simulations") or []

    if not simulations:
        raise ValueError("No simulations found in batch file.")

    rows = extract_sim_rows(simulations)
    dispersion_keys = get_dispersion_keys(batch)

    plot_metric_distributions(rows, batch_id)

    if dispersion_keys:
        primary_key = dispersion_keys[0]
        plot_dispersion_sensitivity(rows, primary_key, batch_id)

    print(f"Loaded batch: {batch_file}")
    print(f"Batch ID: {batch_id}")
    print(f"Simulations: {len(simulations)}")
    print("Close the matplotlib windows to end the script.")
    plt.show()
