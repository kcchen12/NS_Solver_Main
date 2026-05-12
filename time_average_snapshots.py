"""Compute time-averaged and RMS flow fields from saved snapshots.

This script reads ``snap_*.npz`` files from an output directory, converts the
staggered velocity fields to cell centers, and saves:

- mean streamwise velocity ``u_mean``
- mean vertical velocity ``v_mean``
- mean pressure ``p_mean``
- streamwise velocity RMS ``u_rms``
- vertical velocity RMS ``v_rms``

The averages are time-weighted using trapezoidal weights derived from the
snapshot times.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from analyze_aerodynamics import (
    _collect_snapshots,
    _load_grid_faces_for_snapshot,
    _load_snapshot_fields,
)


DEFAULT_RESULTS_DIR = "results"


def _compute_time_weights(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Return trapezoidal-integration weights and total averaging duration."""
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")
    if times.size == 1:
        return np.ones(1, dtype=float), 1.0

    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("Snapshot times must be strictly increasing.")

    weights = np.empty_like(times, dtype=float)
    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]
    if times.size > 2:
        weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    total_duration = float(np.sum(weights))
    if total_duration <= 0.0:
        raise ValueError("Total averaging duration must be positive.")
    return weights, total_duration


def _cell_center_velocities(
    u_face: np.ndarray,
    v_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MAC face-centered velocities to cell centers."""
    u_center = 0.5 * (u_face[:-1, :] + u_face[1:, :])
    v_center = 0.5 * (v_face[:, :-1] + v_face[:, 1:])
    return u_center, v_center


def compute_time_averaged_fields(
    indir: str = "output",
    pattern: str = "snap_*.npz",
    t_min: float = 0.0,
    t_max: Optional[float] = None,
) -> dict[str, np.ndarray | float | int]:
    """Compute mean and RMS flow statistics from snapshot files."""
    snapshots = _collect_snapshots(indir, pattern)
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshots found in {indir!r} matching pattern {pattern!r}."
        )

    if t_max is None:
        filtered = [(t, path) for t, path in snapshots if t >= float(t_min)]
    else:
        filtered = [
            (t, path)
            for t, path in snapshots
            if float(t_min) <= t <= float(t_max)
        ]
    if not filtered:
        raise ValueError("No snapshots remain after applying the requested time window.")

    times = np.array([t for t, _ in filtered], dtype=float)
    weights, total_duration = _compute_time_weights(times)

    mean_u = mean_v = mean_p = None
    mean_u2 = mean_v2 = None
    xc = yc = None

    for weight, (_, path) in zip(weights, filtered):
        u_face, v_face, p = _load_snapshot_fields(path, "u", "v", "p")
        u_center, v_center = _cell_center_velocities(u_face, v_face)

        if mean_u is None:
            nx, ny = p.shape
            xf, yf = _load_grid_faces_for_snapshot(path, nx=nx, ny=ny)
            xc = 0.5 * (xf[:-1] + xf[1:])
            yc = 0.5 * (yf[:-1] + yf[1:])
            mean_u = np.zeros_like(u_center, dtype=float)
            mean_v = np.zeros_like(v_center, dtype=float)
            mean_p = np.zeros_like(p, dtype=float)
            mean_u2 = np.zeros_like(u_center, dtype=float)
            mean_v2 = np.zeros_like(v_center, dtype=float)

        mean_u += weight * u_center
        mean_v += weight * v_center
        mean_p += weight * p
        mean_u2 += weight * (u_center ** 2)
        mean_v2 += weight * (v_center ** 2)

    assert mean_u is not None
    assert mean_v is not None
    assert mean_p is not None
    assert mean_u2 is not None
    assert mean_v2 is not None
    assert xc is not None
    assert yc is not None

    mean_u /= total_duration
    mean_v /= total_duration
    mean_p /= total_duration
    mean_u2 /= total_duration
    mean_v2 /= total_duration

    u_rms = np.sqrt(np.maximum(mean_u2 - mean_u ** 2, 0.0))
    v_rms = np.sqrt(np.maximum(mean_v2 - mean_v ** 2, 0.0))

    return {
        "xc": xc,
        "yc": yc,
        "u_mean": mean_u,
        "v_mean": mean_v,
        "p_mean": mean_p,
        "u_rms": u_rms,
        "v_rms": v_rms,
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
        "n_snapshots": int(times.size),
        "averaging_duration": total_duration,
    }


def save_time_averaged_fields(
    save_name: str = "time_averaged_fields.npz",
    results_dir: str = DEFAULT_RESULTS_DIR,
    **kwargs,
) -> str:
    """Compute and save time-averaged field statistics to a compressed NPZ."""
    stats = compute_time_averaged_fields(**kwargs)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_name)
    np.savez_compressed(save_path, **stats)
    return save_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute time-averaged mean and RMS fields from output snapshots."
    )
    parser.add_argument("--indir", type=str, default="output",
                        help="directory containing snap_*.npz files")
    parser.add_argument("--pattern", type=str, default="snap_*.npz",
                        help="snapshot filename glob pattern")
    parser.add_argument("--t-min", type=float, default=0.0,
                        help="minimum snapshot time to include")
    parser.add_argument("--t-max", type=float, default=None,
                        help="maximum snapshot time to include")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="directory to receive the averaged output file")
    parser.add_argument("--save-name", type=str, default="time_averaged_fields.npz",
                        help="output NPZ filename")
    args = parser.parse_args(argv)

    save_path = save_time_averaged_fields(
        indir=args.indir,
        pattern=args.pattern,
        t_min=args.t_min,
        t_max=args.t_max,
        results_dir=args.results_dir,
        save_name=args.save_name,
    )
    print(f"Saved time-averaged fields: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
