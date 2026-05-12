#!/usr/bin/env python3
"""Simple viewer for `.npz` snapshot files produced by this project.

Usage examples:
  # show keys and shapes in the latest snapshot
  python view_snapshot_viewer.py --list

  # plot a named variable from a snapshot file
    python view_snapshot_viewer.py output/snap_005.0000.npz -k u

    # plot the latest pressure snapshot and save to PNG
  python view_snapshot_viewer.py -k pressure --save out.png

    # plot drag/lift coefficient histories from output CSVs
    # (default trims startup and plots from t >= 0.5)
    python view_snapshot_viewer.py --plot-coeffs --save coeff_history.png

    # override startup trim if needed
    python view_snapshot_viewer.py --plot-coeffs --coeff-t-min 0.25 --save coeff_history.png

The script attempts to guess array layouts but accepts explicit --slice and --comp parameters.
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from src.config import ConfigParser


DEFAULT_RESULTS_DIR = "results"


def _load_cylinder_overlay_geometry(
    config_path: str = "config.txt",
) -> tuple[float, float, float] | None:
    """Load cylinder center/radius for hollow plot overlays from config."""
    if not os.path.exists(config_path):
        return None

    cfg = ConfigParser(config_path)
    if not cfg.get("cylinder", False, bool):
        return None

    lx = cfg.get("lx", 10.0, float)
    ly = cfg.get("ly", 1.0, float)
    x_min = cfg.get("x_min", 0.0, float)
    y_min = cfg.get("y_min", 0.0, float)
    cx = cfg.get("cylinder_center_x", -1.0, float)
    cy = cfg.get("cylinder_center_y", -1.0, float)
    radius = cfg.get("cylinder_radius", -1.0, float)

    if cx < 0.0:
        cx = x_min + 0.25 * lx
    if cy < 0.0:
        cy = y_min + 0.5 * ly
    if radius <= 0.0:
        radius = ly / 8.0

    return float(cx), float(cy), float(radius)


def ensure_results_dir(results_dir: str = "results") -> str:
    """Ensure the results directory exists."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def _result_path(filename: str, results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    return os.path.join(ensure_results_dir(results_dir), filename)


def _load_npz_keys(path: str) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return list(data.keys())


def find_latest_snapshot(dirpath: str = "output", pattern: str = "snap_*.npz") -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(dirpath, pattern)))
    return paths[-1] if paths else None


def find_coeff_series_file(indir: str = "output") -> Optional[str]:
    """Find a coefficient time-series CSV, preferring forces.csv then aero.csv."""
    candidates = [
        os.path.join(indir, "forces.csv"),
        os.path.join(indir, "aero.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_coeff_series(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load t, c_d, c_l from a CSV produced by aerodynamic analysis scripts."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.size == 0:
        raise ValueError(f"No rows found in coefficient file: {path}")

    names = arr.dtype.names or ()
    required = {"t", "c_d", "c_l"}
    if not required.issubset(set(names)):
        raise ValueError(
            f"CSV missing required columns {sorted(required)}. Found: {list(names)}"
        )

    t = np.atleast_1d(arr["t"]).astype(float)
    c_d = np.atleast_1d(arr["c_d"]).astype(float)
    c_l = np.atleast_1d(arr["c_l"]).astype(float)
    return t, c_d, c_l


def _derive_oscillation_save_name(save_name: str) -> str:
    """Append an oscillation-only suffix before file extension."""
    root, ext = os.path.splitext(save_name)
    ext = ext if ext else ".png"
    return f"{root}_oscillation_only{ext}"


def _count_sign_changes(values: np.ndarray) -> int:
    """Count sign flips in a 1-D signal, skipping zeros."""
    signs = np.sign(values)
    signs = signs[signs != 0]
    if signs.size <= 1:
        return 0
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def detect_oscillation_start_index(
    t: np.ndarray,
    c_l: np.ndarray,
    c_d: Optional[np.ndarray] = None,
) -> int:
    """Estimate where oscillation amplitude has reached a stable regime."""
    n = len(c_l)
    if n < 30:
        return 0

    # Remove mean so amplitude/zero-crossing checks are transient-insensitive.
    c_l_centered = c_l - np.mean(c_l)
    c_d_centered = None
    if c_d is not None:
        c_d_centered = c_d - np.mean(c_d)

    w = max(18, n // 24)
    step = max(4, w // 4)
    starts = list(range(0, n - w + 1, step))
    if not starts:
        return 0

    amp_l = np.zeros(len(starts), dtype=float)
    amp_d = np.zeros(len(starts), dtype=float)
    osc_ok = np.zeros(len(starts), dtype=bool)

    for k, i0 in enumerate(starts):
        i1 = i0 + w
        seg_l = c_l_centered[i0:i1]
        amp_l[k] = float(np.percentile(seg_l, 95) - np.percentile(seg_l, 5))
        osc_ok[k] = _count_sign_changes(seg_l) >= 3

        if c_d_centered is not None:
            seg_d = c_d_centered[i0:i1]
            amp_d[k] = float(np.percentile(seg_d, 95) -
                             np.percentile(seg_d, 5))

    valid = np.where(osc_ok)[0]
    if valid.size == 0:
        return 0

    tail_count = min(6, valid.size)
    tail_idx = valid[-tail_count:]
    target_l = max(float(np.median(amp_l[tail_idx])), 1e-12)
    if c_d_centered is not None:
        target_d = max(float(np.median(amp_d[tail_idx])), 1e-12)
    else:
        target_d = 1.0

    hold = 4
    tol = 0.15
    max_k = len(starts) - hold
    for k in range(max(1, valid[0]), max_k):
        if not np.all(osc_ok[k:k + hold]):
            continue

        rel_l = np.abs(amp_l[k:k + hold] - target_l) / target_l
        if np.any(rel_l > tol):
            continue

        if c_d_centered is not None:
            rel_d = np.abs(amp_d[k:k + hold] - target_d) / target_d
            if np.any(rel_d > 0.25):
                continue

        return starts[k]

    # Fallback: use first oscillatory window if stable-amplitude criterion is not met.
    return starts[valid[0]]


def detect_startup_trim_index(
    t: np.ndarray,
    c_d: np.ndarray,
    c_l: np.ndarray,
    coeff_t_min: Optional[float] = None,
) -> int:
    """Return index to start plotting, trimming startup transients/outliers."""
    if t.size == 0:
        return 0

    if coeff_t_min is not None:
        return int(np.searchsorted(t, coeff_t_min, side="left"))

    # Auto-trim a single extreme startup point if it is a clear outlier.
    n = len(t)
    if n < 8:
        return 0

    w = min(20, n - 1)
    cd_base = c_d[1:1 + w]
    cl_base = c_l[1:1 + w]

    cd_scale = float(np.std(cd_base)) + 1e-12
    cl_scale = float(np.std(cl_base)) + 1e-12
    cd_z = abs(float(c_d[0] - np.median(cd_base))) / cd_scale
    cl_z = abs(float(c_l[0] - np.median(cl_base))) / cl_scale

    return 1 if max(cd_z, cl_z) >= 8.0 else 0


def detect_tail_fraction_start_index(t: np.ndarray, fraction: float = 0.2) -> int:
    """Return index corresponding to the final `fraction` of the time span."""
    if t.size == 0:
        return 0
    if t.size == 1:
        return 0

    frac = float(np.clip(fraction, 1e-6, 1.0))
    t_start = float(t[0]) + (1.0 - frac) * float(t[-1] - t[0])
    return int(np.searchsorted(t, t_start, side="left"))


def plot_coeff_history(
    csv_path: str,
    save_name: Optional[str] = None,
    coeff_t_min: Optional[float] = None,
) -> None:
    """Plot drag and lift coefficients as functions of time."""
    t, c_d, c_l = load_coeff_series(csv_path)
    i_plot_start = detect_startup_trim_index(
        t, c_d, c_l, coeff_t_min=coeff_t_min)
    if i_plot_start >= len(t):
        raise ValueError("No coefficient samples left after startup trimming.")

    t_plot = t[i_plot_start:]
    c_d_plot = c_d[i_plot_start:]
    c_l_plot = c_l[i_plot_start:]

    i_start = detect_tail_fraction_start_index(t, fraction=0.2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t_plot, c_d_plot, color="tab:blue", linewidth=1.6)
    ax1.set_ylabel("C_d")
    ax1.set_title(f"Drag/Lift Coefficient History ({os.path.basename(csv_path)})",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_plot, c_l_plot, color="tab:orange", linewidth=1.6)
    ax2.set_xlabel("time")
    ax2.set_ylabel("C_l")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_name:
        results_dir = ensure_results_dir()
        save_path = os.path.join(results_dir, save_name)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    else:
        plt.show()

    # Also create oscillation-only view from the last 20% of the time range.
    i_start = max(i_start, i_plot_start)
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax3.plot(t[i_start:], c_d[i_start:], color="tab:blue", linewidth=1.6)
    ax3.set_ylabel("C_d")
    ax3.set_title(
        f"Drag/Lift Coefficients (Last 20% Time Window) ({os.path.basename(csv_path)})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)

    ax4.plot(t[i_start:], c_l[i_start:], color="tab:orange", linewidth=1.6)
    ax4.set_xlabel("time")
    ax4.set_ylabel("C_l")
    ax4.grid(True, alpha=0.3)

    fig2.tight_layout()

    if save_name:
        osc_name = _derive_oscillation_save_name(save_name)
        osc_path = _result_path(osc_name)
        fig2.savefig(osc_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {osc_path}")
    else:
        plt.show()


def inspect_npz(path: str):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        print(f"File: {path}")
        if not keys:
            print("(no arrays found)")
            return
        for k in keys:
            print(f"- {k}: shape={data[k].shape}, dtype={data[k].dtype}")


def pick_slice_and_component(arr: np.ndarray, slice_idx: Optional[int], comp_idx: Optional[int]):
    # Try to handle common shapes: (ny,nx), (ny,nx,comp), (comp,ny,nx), (nz,ny,nx), (comp,nz,ny,nx)
    ndim = arr.ndim
    if ndim == 2:
        return arr
    if ndim == 3:
        # If last dim reasonably small, treat as components
        if arr.shape[2] <= 4:
            c = 0 if comp_idx is None else comp_idx
            return arr[:, :, c]
        # If first dim small, treat as (comp,ny,nx)
        if arr.shape[0] <= 4:
            c = 0 if comp_idx is None else comp_idx
            return arr[c, :, :]
        # otherwise treat as (nz,ny,nx) -> take slice
        z = arr.shape[0] // 2 if slice_idx is None else slice_idx
        return arr[z, :, :]
    if ndim == 4:
        # common: (comp,nz,ny,nx) or (nz,ny,nx,comp)
        if arr.shape[0] <= 4:
            c = 0 if comp_idx is None else comp_idx
            z = arr.shape[1] // 2 if slice_idx is None else slice_idx
            return arr[c, z, :, :]
        if arr.shape[3] <= 4:
            c = 0 if comp_idx is None else comp_idx
            z = arr.shape[0] // 2 if slice_idx is None else slice_idx
            return arr[z, :, :, c]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def plot_snapshot_key(
    file_path: str,
    key: str,
    save_name: Optional[str] = None,
    slice_idx: Optional[int] = None,
    comp_idx: Optional[int] = None,
) -> None:
    """Plot one array key from a snapshot file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with np.load(file_path, allow_pickle=False) as data:
        keys = list(data.keys())
        if key not in keys:
            raise KeyError(f"Key '{key}' not found. Available: {keys}")
        arr = np.array(data[key])

    img = pick_slice_and_component(arr, slice_idx, comp_idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, origin="lower", cmap="viridis")
    ax.set_title(f"{os.path.basename(file_path)} : {key}",
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label=key)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)

    if save_name:
        save_path = _result_path(save_name)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _load_snapshot_velocity_fields(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load face-centered velocity arrays from a snapshot."""
    with np.load(file_path, allow_pickle=False) as data:
        if "u" not in data or "v" not in data:
            raise KeyError(f"Snapshot {file_path} must contain 'u' and 'v' arrays.")
        u = np.asarray(data["u"], dtype=float)
        v = np.asarray(data["v"], dtype=float)
    return u, v


def _compute_snapshot_vorticity(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cell-centered vorticity from one saved snapshot."""
    u, v = _load_snapshot_velocity_fields(file_path)
    nx, ny = u.shape[0] - 1, v.shape[1] - 1
    xf, yf = _load_grid_faces_for_snapshot(file_path, nx=nx, ny=ny)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])

    u_c = 0.5 * (u[:-1, :] + u[1:, :])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])

    edge_x = 2 if nx >= 3 else 1
    edge_y = 2 if ny >= 3 else 1
    dv_dx = np.gradient(v_c, xc, axis=0, edge_order=edge_x)
    du_dy = np.gradient(u_c, yc, axis=1, edge_order=edge_y)
    omega = dv_dx - du_dy
    return xc, yc, omega


def plot_vorticity_video(
    snapshot_dir: str = "output",
    pattern: str = "snap_*.npz",
    save_name: str = "vorticity.gif",
    fps: int = 12,
    frame_stride: int = 1,
    verbose: bool = False,
    results_dir: str = DEFAULT_RESULTS_DIR,
    config_path: str = "config.txt",
) -> None:
    """Create an animated GIF of vorticity over all saved snapshots."""
    if save_name:
        plt.switch_backend("Agg")

    paths = sorted(glob.glob(os.path.join(snapshot_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No snapshots found in '{snapshot_dir}' matching '{pattern}'."
        )
    stride = max(int(frame_stride), 1)
    paths = paths[::stride]
    total_frames = len(paths)
    progress_interval = max(1, total_frames // 10)

    if verbose:
        print(
            "  Building vorticity video: "
            f"{total_frames} frame(s) from '{snapshot_dir}' "
            f"(stride={stride}, fps={max(int(fps), 1)})"
        )

    frames: list[np.ndarray] = []
    times: list[float] = []
    skipped: list[str] = []
    xc = yc = None
    max_abs_omega = 0.0

    for idx, path in enumerate(paths, start=1):
        try:
            xc_i, yc_i, omega = _compute_snapshot_vorticity(path)
            frames.append(omega)
            xc = xc_i
            yc = yc_i
            max_abs_omega = max(max_abs_omega, float(np.max(np.abs(omega))))
            with np.load(path, allow_pickle=False) as data:
                times.append(float(data["t"]) if "t" in data else float(len(times)))
            if verbose and (idx == 1 or idx == total_frames or idx % progress_interval == 0):
                print(f"    Loaded vorticity frame {idx}/{total_frames}")
        except Exception:
            skipped.append(os.path.basename(path))

    if skipped:
        preview = ", ".join(skipped[:5])
        suffix = "" if len(skipped) <= 5 else ", ..."
        print(
            f"Skipped {len(skipped)} vorticity frame(s) due to incomplete snapshots: "
            f"{preview}{suffix}"
        )

    if xc is None or yc is None or not frames:
        raise RuntimeError("Failed to assemble vorticity frames.")

    # Tighten the displayed range so the vorticity colors read darker/more saturated.
    max_abs_omega = max(0.5 * max_abs_omega, 1e-12)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    dx = float(xc[1] - xc[0]) if len(xc) > 1 else 1.0
    dy = float(yc[1] - yc[0]) if len(yc) > 1 else 1.0
    extent = (
        float(xc[0] - 0.5 * dx),
        float(xc[-1] + 0.5 * dx),
        float(yc[0] - 0.5 * dy),
        float(yc[-1] + 0.5 * dy),
    )
    image = ax.imshow(
        frames[0].T,
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=-max_abs_omega,
        vmax=max_abs_omega,
        aspect="auto",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$\omega_z$")
    title = ax.set_title(f"Vorticity t={times[0]:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    circle_geometry = _load_cylinder_overlay_geometry(config_path=config_path)
    if circle_geometry is not None:
        cx, cy, radius = circle_geometry
        ax.add_patch(
            Circle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor="black",
                linewidth=1.6,
                zorder=6,
            )
        )

    def _update(frame_idx: int):
        image.set_data(frames[frame_idx].T)
        title.set_text(f"Vorticity t={times[frame_idx]:.4f}")
        return [image, title]

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000.0 / max(fps, 1), blit=False)
    save_path = _result_path(save_name, results_dir=results_dir)
    if verbose:
        print(f"  Encoding vorticity GIF to {save_path}")
    anim.save(save_path, writer=PillowWriter(fps=max(int(fps), 1)))
    print(f"Saved animation: {save_path}")
    plt.close(fig)


def _load_ibm_cell_forcing(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load cell-centered IBM forcing from snapshot, with face-based fallback."""
    with np.load(file_path, allow_pickle=False) as data:
        keys = set(data.keys())

        if "ibm_forcing_x_cell" in keys and "ibm_forcing_y_cell" in keys:
            fx = np.array(data["ibm_forcing_x_cell"], dtype=float)
            fy = np.array(data["ibm_forcing_y_cell"], dtype=float)
            return fx, fy

        if "ibm_forcing_u_face" in keys and "ibm_forcing_v_face" in keys:
            fu = np.array(data["ibm_forcing_u_face"], dtype=float)
            fv = np.array(data["ibm_forcing_v_face"], dtype=float)
            fx = 0.5 * (fu[:-1, :] + fu[1:, :])
            fy = 0.5 * (fv[:, :-1] + fv[:, 1:])
            return fx, fy

        raise KeyError(
            "IBM forcing fields not found. Expected ibm_forcing_x_cell/ibm_forcing_y_cell "
            "or ibm_forcing_u_face/ibm_forcing_v_face in snapshot."
        )


def _load_grid_faces_for_snapshot(
    file_path: str,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load physical grid faces from prepared-grid metadata when available."""
    indir = os.path.dirname(os.path.abspath(file_path))
    candidates = [
        os.path.join(indir, "nonuniform_grid.npz"),
        os.path.join(indir, "uniform_grid.npz"),
    ]

    for grid_path in candidates:
        if not os.path.exists(grid_path):
            continue
        try:
            with np.load(grid_path, allow_pickle=False) as meta:
                if "xf" in meta and "yf" in meta:
                    xf = np.asarray(meta["xf"], dtype=float)
                    yf = np.asarray(meta["yf"], dtype=float)
                    if xf.shape == (nx + 1,) and yf.shape == (ny + 1,):
                        return xf, yf
        except Exception:
            continue

    # Fallback to index-like coordinates if no prepared grid metadata is found.
    return np.arange(nx + 1, dtype=float), np.arange(ny + 1, dtype=float)


def plot_ibm_forcing(
    file_path: str,
    save_name: Optional[str] = None,
) -> None:
    """Plot IBM forcing x/y components and magnitude from a snapshot."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    fx, fy = _load_ibm_cell_forcing(file_path)
    fmag = np.sqrt(fx**2 + fy**2)
    nx, ny = fx.shape
    xf, yf = _load_grid_faces_for_snapshot(file_path, nx=nx, ny=ny)

    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")

    vmax_comp = max(float(np.percentile(
        np.abs(np.concatenate([fx.ravel(), fy.ravel()])), 99.0)), 1e-12)
    vmax_mag = max(float(np.percentile(fmag, 99.0)), 1e-12)

    active_threshold = max(1e-12, 0.01 * float(np.max(fmag)))
    active = fmag >= active_threshold

    if np.any(active):
        ii, jj = np.where(active)
        pad_cells = 3
        i0 = max(int(np.min(ii)) - pad_cells, 0)
        i1 = min(int(np.max(ii)) + pad_cells, nx - 1)
        j0 = max(int(np.min(jj)) - pad_cells, 0)
        j1 = min(int(np.max(jj)) + pad_cells, ny - 1)
        xlim = (float(xf[i0]), float(xf[i1 + 1]))
        ylim = (float(yf[j0]), float(yf[j1 + 1]))
    else:
        xlim = (float(xf[0]), float(xf[-1]))
        ylim = (float(yf[0]), float(yf[-1]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    im0 = axes[0].pcolormesh(Xf, Yf, fx, shading="flat", cmap="seismic",
                             vmin=-vmax_comp, vmax=vmax_comp)
    axes[0].set_title("IBM Forcing x (cell)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(Xf, Yf, fy, shading="flat", cmap="seismic",
                             vmin=-vmax_comp, vmax=vmax_comp)
    axes[1].set_title("IBM Forcing y (cell)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(Xf, Yf, fmag, shading="flat",
                             cmap="inferno", vmin=0.0, vmax=vmax_mag)
    axes[2].set_title("IBM Forcing Magnitude", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")

    fig.suptitle(
        f"IBM Forcing Fields: {os.path.basename(file_path)}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_name:
        save_path = _result_path(save_name)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="View .npz snapshot files with velocity visualization (matplotlib viewer)")
    parser.add_argument("file", nargs="?",
                        help=".npz snapshot file (default: latest in output/)")
    parser.add_argument("-l", "--list", action="store_true",
                        help="list keys and shapes then exit")
    parser.add_argument(
        "-k", "--key", help="variable name (npz archive key) to plot")
    parser.add_argument("-s", "--slice", type=int,
                        help="slice index for 3D arrays (nz dimension)")
    parser.add_argument("-c", "--comp", type=int,
                        help="component index for vector fields")
    parser.add_argument(
        "--save", help="save plot to this path instead of showing")
    parser.add_argument("--plot-coeffs", action="store_true",
                        help="plot drag/lift coefficients vs time from forces.csv or aero.csv")
    parser.add_argument("--plot-ibm", action="store_true",
                        help="plot IBM forcing x/y/magnitude from a snapshot")
    parser.add_argument("--plot-vorticity-video", action="store_true",
                        help="create an animated vorticity GIF from output snapshots")
    parser.add_argument("--coeff-file", type=str,
                        help="path to coefficient CSV (default: auto from output/)")
    parser.add_argument("--coeff-indir", type=str, default="output",
                        help="directory searched for forces.csv/aero.csv (default: output)")
    parser.add_argument("--coeff-t-min", type=float, default=0.5,
                        help="minimum time for coefficient plots (default: 0.5)"
                        )
    parser.add_argument("--snapshot-dir", type=str, default="output",
                        help="directory searched for snap_*.npz when building a vorticity video")
    parser.add_argument("--config", type=str, default="config.txt",
                        help="solver configuration file used for cylinder overlays")
    parser.add_argument("--video-fps", type=int, default=12,
                        help="frames per second for the animated vorticity GIF")
    parser.add_argument("--video-frame-stride", type=int, default=1,
                        help="use every nth snapshot when building the vorticity GIF")
    parser.add_argument("--verbose", action="store_true",
                        help="print progress while building derived outputs")
    args = parser.parse_args(argv)

    if args.plot_coeffs:
        coeff_path = args.coeff_file or find_coeff_series_file(
            args.coeff_indir)
        if coeff_path is None:
            print("Could not find coefficient CSV. Expected forces.csv or aero.csv.",
                  file=sys.stderr)
            sys.exit(7)
        try:
            plot_coeff_history(coeff_path, args.save,
                               coeff_t_min=args.coeff_t_min)
        except Exception as e:
            print(f"Error plotting coefficients: {e}", file=sys.stderr)
            sys.exit(8)
        return

    if args.plot_ibm:
        if args.file is None:
            latest = find_latest_snapshot()
            if latest is None:
                print("No snapshot file specified and none found in 'output/'",
                      file=sys.stderr)
                parser.print_help()
                sys.exit(1)
            args.file = latest

        if not os.path.exists(args.file):
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(2)

        save_name = args.save or "ibm_forcing.png"
        try:
            plot_ibm_forcing(args.file, save_name=save_name)
        except Exception as e:
            print(f"Error plotting IBM forcing: {e}", file=sys.stderr)
            sys.exit(9)
        return

    if args.plot_vorticity_video:
        save_name = args.save or "vorticity.gif"
        try:
            plot_vorticity_video(
                snapshot_dir=args.snapshot_dir,
                save_name=save_name,
                fps=args.video_fps,
                frame_stride=args.video_frame_stride,
                verbose=args.verbose,
                config_path=args.config,
            )
        except Exception as e:
            print(f"Error creating vorticity video: {e}", file=sys.stderr)
            sys.exit(10)
        return

    if args.file is None:
        latest = find_latest_snapshot()
        if latest is None:
            print("No snapshot file specified and none found in 'output/'",
                  file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        args.file = latest

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    keys = _load_npz_keys(args.file)
    if args.list:
        inspect_npz(args.file)
        return
    if not keys:
        print("No arrays found in file", file=sys.stderr)
        sys.exit(3)
    key = args.key or keys[0]

    try:
        plot_snapshot_key(
            args.file,
            key,
            save_name=args.save,
            slice_idx=args.slice,
            comp_idx=args.comp,
        )
    except Exception as e:
        print(f"Error plotting snapshot key: {e}", file=sys.stderr)
        sys.exit(6)


if __name__ == "__main__":
    main()
