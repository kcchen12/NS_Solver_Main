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


def _validate_axis_scales(
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> tuple[float, float]:
    """Validate independent display scale factors for x and y."""
    x_scale = float(x_scale)
    y_scale = float(y_scale)
    if x_scale <= 0.0 or y_scale <= 0.0:
        raise ValueError("x_scale and y_scale must both be positive.")
    return x_scale, y_scale


def _axis_aspect(x_scale: float = 1.0, y_scale: float = 1.0) -> float:
    """Return the matplotlib aspect ratio for requested x/y display scaling."""
    x_scale, y_scale = _validate_axis_scales(x_scale=x_scale, y_scale=y_scale)
    return y_scale / x_scale


def _scaled_figsize(
    width: float,
    height: float,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> tuple[float, float]:
    """Scale figure width/height independently to match the requested view."""
    x_scale, y_scale = _validate_axis_scales(x_scale=x_scale, y_scale=y_scale)
    return width * x_scale, height * y_scale


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


def plot_time_averaged_fields(
    stats_path: str,
    save_name: str = "time_averaged_fields.png",
    results_dir: str = DEFAULT_RESULTS_DIR,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> str:
    """Render a readable summary plot from a saved time-averaged-field NPZ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to plot time-averaged fields.") from exc

    with np.load(stats_path, allow_pickle=False) as data:
        xc = np.asarray(data["xc"], dtype=float)
        yc = np.asarray(data["yc"], dtype=float)
        u_mean = np.asarray(data["u_mean"], dtype=float)
        v_mean = np.asarray(data["v_mean"], dtype=float)
        p_mean = np.asarray(data["p_mean"], dtype=float)
        u_rms = np.asarray(data["u_rms"], dtype=float)
        v_rms = np.asarray(data["v_rms"], dtype=float)

    dx = float(xc[1] - xc[0]) if xc.size > 1 else 1.0
    dy = float(yc[1] - yc[0]) if yc.size > 1 else 1.0
    extent = (
        float(xc[0] - 0.5 * dx),
        float(xc[-1] + 0.5 * dx),
        float(yc[0] - 0.5 * dy),
        float(yc[-1] + 0.5 * dy),
    )
    mean_speed = np.sqrt(u_mean ** 2 + v_mean ** 2)

    figsize = _scaled_figsize(15.0, 8.0, x_scale=x_scale, y_scale=y_scale)
    aspect = _axis_aspect(x_scale=x_scale, y_scale=y_scale)
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    panels = [
        ("Mean u", u_mean, "seismic"),
        ("Mean p", p_mean, "RdBu_r"),
        ("Mean speed", mean_speed, "viridis"),
        ("u RMS", u_rms, "magma"),
        ("v RMS", v_rms, "magma"),
    ]

    for ax, (label, field, cmap) in zip(axes.flat, panels):
        if label == "Mean u":
            vmax = max(float(np.percentile(np.abs(field), 99.0)), 1e-12)
            vmin = -vmax
        elif label == "Mean p":
            vmax = max(float(np.percentile(np.abs(field), 99.0)), 1e-12)
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = max(float(np.percentile(field, 99.0)), 1e-12)
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        colorbar = fig.colorbar(im, ax=ax, shrink=0.86)
        colorbar.set_label(label)

    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_name)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
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
    parser.add_argument("--plot", action="store_true",
                        help="also save a readable PNG summary of the averaged fields")
    parser.add_argument("--plot-save-name", type=str, default="time_averaged_fields.png",
                        help="output PNG filename when --plot is used")
    parser.add_argument("--x-scale", type=float, default=1.0,
                        help="horizontal display scale for the saved PNG plot")
    parser.add_argument("--y-scale", type=float, default=1.0,
                        help="vertical display scale for the saved PNG plot")
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
    if args.plot:
        plot_path = plot_time_averaged_fields(
            save_path,
            save_name=args.plot_save_name,
            results_dir=args.results_dir,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
        )
        print(f"Saved time-averaged field plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
