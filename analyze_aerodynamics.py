#!/usr/bin/env python3
"""Comprehensive aerodynamic analysis from snapshot data.

This script combines Strouhal number and drag/lift coefficient analysis.
It computes force-based drag/lift histories from snapshots and estimates
Strouhal number from the lift-coefficient signal. Optional point-probe
sampling is retained for CSV export and inspection.

Outputs:
    - Time series CSV with probe values, forces, and coefficients
    - Comprehensive text report with all statistics
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.config import ConfigParser


DEFAULT_RESULTS_DIR = "results"


@dataclass
class SpectralResult:
    freq: float
    peak_power: float
    st: float


@dataclass(frozen=True)
class BilinearPlan:
    i: int
    j: int
    w11: float
    w21: float
    w12: float
    w22: float


@dataclass(frozen=True)
class CylinderGeometry:
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class SurfaceForcePlan:
    theta: np.ndarray
    normals_x: np.ndarray
    normals_y: np.ndarray
    arc_length: float
    bilinear_plans: Tuple[BilinearPlan, ...]


def _time_from_filename(path: str) -> Optional[float]:
    """Extract snapshot time from filename pattern snap_<time>.npz."""
    name = os.path.basename(path)
    m = re.match(r"^snap_([-+0-9.eE]+)\.npz$", name)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _collect_snapshots(indir: str, pattern: str) -> List[Tuple[float, str]]:
    """Return unique snapshot list as (time, path), sorted by time."""
    candidates = sorted(glob.glob(os.path.join(indir, pattern)))
    if not candidates:
        return []

    by_time: Dict[float, str] = {}
    for path in candidates:
        t = _time_from_filename(path)
        if t is None:
            with np.load(path, allow_pickle=False) as data:
                t = float(data["t"])
        by_time[t] = path

    return [(t, by_time[t]) for t in sorted(by_time.keys())]


def _safe_scalar(data: np.lib.npyio.NpzFile, key: str) -> Optional[float]:
    if key not in data.files:
        return None
    return float(np.array(data[key]).item())


def _load_snapshot_grid_metadata(first_path: str) -> Tuple[int, int, float, float]:
    """Read grid dimensions and domain size from one snapshot."""
    with np.load(first_path, allow_pickle=False) as data:
        p = data["p"]
        nx = int(p.shape[0])
        ny = int(p.shape[1])
        lx = _safe_scalar(data, "meta_lx")
        ly = _safe_scalar(data, "meta_ly")

    if lx is None or ly is None:
        raise ValueError(
            "Snapshot metadata does not include meta_lx/meta_ly. "
            "Please provide snapshots written by main.py with metadata."
        )

    return nx, ny, float(lx), float(ly)


def _read_config(config_path: str) -> Dict[str, float]:
    """Parse numeric/bool config values that aerodynamic post-processing uses."""
    parser = ConfigParser(config_path)
    config: Dict[str, float] = {}
    for key, raw_value in parser.get_all().items():
        value_str = str(raw_value).strip().lower()
        if value_str in {"true", "1", "yes", "on"}:
            config[key] = 1.0
            continue
        if value_str in {"false", "0", "no", "off"}:
            config[key] = 0.0
            continue
        try:
            int_value = int(raw_value)
        except (TypeError, ValueError):
            int_value = None
        if int_value is not None:
            config[key] = float(int_value)
            continue
        try:
            config[key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return config


def _build_uniform_face_and_center_coords(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xf = np.linspace(0.0, lx, nx + 1)
    yf = np.linspace(0.0, ly, ny + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])
    return xf, xc, yf, yc


def _load_snapshot_fields(path: str, *field_names: str) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in field_names if name not in data.files]
        if missing:
            available = ", ".join(sorted(data.files))
            missing_str = ", ".join(missing)
            raise KeyError(
                f"Snapshot {os.path.basename(path)!r} is missing field(s): "
                f"{missing_str}. Available fields: {available}"
            )
        return tuple(np.array(data[name], copy=True) for name in field_names)


def _resolve_cylinder_radius(config: Dict[str, float], ly: float) -> float:
    """Resolve cylinder radius from config with a positive fallback."""
    radius = float(config.get("cylinder_radius", ly / 8.0))
    return radius if radius > 0.0 else float(ly / 8.0)


def _resolve_cylinder_center(
    config: Dict[str, float],
    lx: float,
    ly: float,
) -> Tuple[float, float]:
    """Resolve cylinder center from config with solver-default fallback."""
    center_x = float(config.get("cylinder_center_x", lx / 4.0))
    center_y = float(config.get("cylinder_center_y", ly / 2.0))
    return center_x, center_y


def _estimate_scales(
    first_path: str,
    length_scale: Optional[float],
    use_cylinder_diameter: bool,
    u_ref: float,
    config_path: Optional[str] = None,
) -> Tuple[float, float, int, int, float, float]:
    """Read metadata and return (L, U, nx, ny, lx, ly)."""
    nx, ny, lx, ly = _load_snapshot_grid_metadata(first_path)

    config = _read_config(config_path) if config_path else {}
    cfg_length_scale = config.get("aero_length_scale", None)
    cfg_use_cyl_d = bool(config.get("aero_use_cylinder_diameter", 0.0))
    cylinder_radius = _resolve_cylinder_radius(config, ly)

    if length_scale is not None:
        l_char = float(length_scale)
    elif cfg_length_scale is not None and float(cfg_length_scale) > 0.0:
        l_char = float(cfg_length_scale)
    elif use_cylinder_diameter or cfg_use_cyl_d:
        l_char = 2.0 * cylinder_radius
    else:
        if "cylinder" in config and config["cylinder"] != 0:
            l_char = 2.0 * cylinder_radius
        else:
            l_char = 1.0

    if l_char <= 0.0:
        raise ValueError("Characteristic length must be positive.")
    if u_ref <= 0.0:
        raise ValueError("Reference velocity must be positive.")

    return l_char, float(u_ref), nx, ny, float(lx), float(ly)


def _estimate_cylinder_geometry(
    first_path: str,
    config_path: Optional[str] = None,
    cylinder_radius: Optional[float] = None,
) -> CylinderGeometry:
    """Estimate cylinder geometry from snapshot metadata and config."""
    _, _, lx, ly = _load_snapshot_grid_metadata(first_path)
    config = _read_config(config_path) if config_path else {}

    if cylinder_radius is not None:
        r = float(cylinder_radius)
    else:
        r = _resolve_cylinder_radius(config, ly)

    center_x, center_y = _resolve_cylinder_center(config, lx, ly)

    return CylinderGeometry(
        center_x=center_x,
        center_y=center_y,
        radius=r,
    )


def _build_bilinear_plan(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
) -> BilinearPlan:
    """Precompute cell indices and bilinear weights for one fixed probe."""
    i = int(np.searchsorted(x_grid, x) - 1)
    j = int(np.searchsorted(y_grid, y) - 1)

    i = int(np.clip(i, 0, len(x_grid) - 2))
    j = int(np.clip(j, 0, len(y_grid) - 2))

    x0, x1 = x_grid[i], x_grid[i + 1]
    y0, y1 = y_grid[j], y_grid[j + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    w11 = (1.0 - tx) * (1.0 - ty)
    w21 = tx * (1.0 - ty)
    w12 = (1.0 - tx) * ty
    w22 = tx * ty

    return BilinearPlan(i=i, j=j, w11=w11, w21=w21, w12=w12, w22=w22)


def _apply_bilinear_plan(values: np.ndarray, plan: BilinearPlan) -> float:
    """Apply precomputed bilinear interpolation weights to a field array."""
    i = plan.i
    j = plan.j
    return float(
        plan.w11 * values[i, j]
        + plan.w21 * values[i + 1, j]
        + plan.w12 * values[i, j + 1]
        + plan.w22 * values[i + 1, j + 1]
    )


def _extract_probe_series(
    snapshots: Iterable[Tuple[float, str]],
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    probe_x: Optional[float],
    probe_y: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays: t, u_probe, v_probe, p_probe."""
    snapshots_list = list(snapshots)
    n = len(snapshots_list)

    if probe_x is None or probe_y is None:
        times = np.array([t for t, _ in snapshots_list], dtype=float)
        nan_vals = np.full(n, np.nan, dtype=float)
        return times, nan_vals.copy(), nan_vals.copy(), nan_vals.copy()

    xf, xc, yf, yc = _build_uniform_face_and_center_coords(nx, ny, lx, ly)

    u_plan = _build_bilinear_plan(xf, yc, probe_x, probe_y)
    v_plan = _build_bilinear_plan(xc, yf, probe_x, probe_y)
    p_plan = _build_bilinear_plan(xc, yc, probe_x, probe_y)

    times = np.empty(n, dtype=float)
    u_vals = np.empty(n, dtype=float)
    v_vals = np.empty(n, dtype=float)
    p_vals = np.empty(n, dtype=float)

    for k, (t, path) in enumerate(snapshots_list):
        u, v, p = _load_snapshot_fields(path, "u", "v", "p")

        u_probe = _apply_bilinear_plan(u, u_plan)
        v_probe = _apply_bilinear_plan(v, v_plan)
        p_probe = _apply_bilinear_plan(p, p_plan)

        times[k] = t
        u_vals[k] = u_probe
        v_vals[k] = v_probe
        p_vals[k] = p_probe

    return times, u_vals, v_vals, p_vals


def _dominant_frequency(
    t: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    f_min: float,
    f_max: float,
) -> Optional[Tuple[float, float]]:
    """Return (f_peak, peak_power) from a one-sided Fourier power spectrum."""
    out = _compute_fourier_power_spectrum(
        t,
        signal,
        t_min=t_min,
        f_min=f_min,
        f_max=f_max,
    )
    if out is None:
        return None

    _, _, freq_band, power_band = out
    idx = int(np.argmax(power_band))
    return float(freq_band[idx]), float(power_band[idx])


def _compute_fourier_power_spectrum(
    t: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    f_min: float,
    f_max: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (t_uniform, signal_uniform, freq_band, power_band) for a one-sided FFT."""
    mask = t >= t_min
    ts = t[mask]
    ys = signal[mask]

    if ts.size < 8:
        return None

    ys = ys - np.mean(ys)
    sigma = float(np.std(ys))
    if sigma < 1e-12:
        return None

    dt = np.diff(ts)
    dt_ref = float(np.median(dt)) if dt.size else np.nan
    if not np.isfinite(dt_ref) or dt_ref <= 0.0:
        return None

    if not np.allclose(dt, dt_ref, rtol=1e-4, atol=1e-10):
        # Snapshot times can drift under adaptive stepping even when the saved
        # history is otherwise smooth. Resample onto a uniform grid before FFT.
        n_uniform = int(np.floor((ts[-1] - ts[0]) / dt_ref)) + 1
        if n_uniform < 8:
            return None
        ts_uniform = ts[0] + dt_ref * np.arange(n_uniform, dtype=float)
        ys = np.interp(ts_uniform, ts, ys)
        ts = ts_uniform

    freq = np.fft.rfftfreq(ts.size, d=dt_ref)
    spectrum = np.fft.rfft(ys)
    power = (np.abs(spectrum) ** 2) / float(ts.size**2)

    band = (freq >= f_min) & (freq <= f_max)
    if not np.any(band):
        return None

    freq_band = freq[band]
    power_band = power[band]
    return ts, ys, freq_band, power_band


def _find_top_spectral_peaks(
    freq: np.ndarray,
    power: np.ndarray,
    max_peaks: int = 3,
    min_relative_power: float = 0.10,
) -> list[int]:
    """Return indices of the strongest local spectral peaks."""
    if freq.size == 0 or power.size == 0:
        return []

    if power.size == 1:
        return [0]

    peak_indices: list[int] = []
    threshold = float(np.max(power)) * float(min_relative_power)

    for idx in range(power.size):
        left = power[idx - 1] if idx > 0 else -np.inf
        right = power[idx + 1] if idx + 1 < power.size else -np.inf
        if power[idx] >= threshold and power[idx] >= left and power[idx] >= right:
            peak_indices.append(idx)

    if not peak_indices:
        peak_indices = [int(np.argmax(power))]

    peak_indices.sort(key=lambda idx: power[idx], reverse=True)
    unique: list[int] = []
    for idx in peak_indices:
        if idx not in unique:
            unique.append(idx)
        if len(unique) >= max_peaks:
            break
    return unique


def _is_edge_frequency(f: float, f_min: float, f_max: float) -> bool:
    """Return True if f is effectively at the search-window edge."""
    width = max(f_max - f_min, 1e-12)
    tol = 1e-3 * width
    return (f - f_min) <= tol or (f_max - f) <= tol


def _build_surface_force_plan(
    xc: np.ndarray,
    yc: np.ndarray,
    geom: CylinderGeometry,
    n_samples: int = 720,
) -> SurfaceForcePlan:
    """Precompute interpolation plans for a line integral on the cylinder surface."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    x_surf = geom.center_x + geom.radius * np.cos(theta)
    y_surf = geom.center_y + geom.radius * np.sin(theta)
    plans = tuple(
        _build_bilinear_plan(xc, yc, float(x), float(y))
        for x, y in zip(x_surf, y_surf)
    )
    return SurfaceForcePlan(
        theta=theta,
        normals_x=np.cos(theta),
        normals_y=np.sin(theta),
        arc_length=2.0 * np.pi * geom.radius / float(n_samples),
        bilinear_plans=plans,
    )


def _sample_surface_pressure(
    p: np.ndarray,
    force_plan: SurfaceForcePlan,
) -> np.ndarray:
    """Sample pressure along the cylinder surface interpolation plan."""
    return np.array(
        [_apply_bilinear_plan(p, plan) for plan in force_plan.bilinear_plans],
        dtype=float,
    )


def _compute_pressure_forces(
    p: np.ndarray,
    force_plan: SurfaceForcePlan,
) -> Tuple[float, float]:
    """Compute pressure forces on the cylinder via a contour integral."""
    pressure_samples = _sample_surface_pressure(p, force_plan)
    pressure_samples -= np.mean(pressure_samples)

    # Force on the body is - integral(p * n ds) over the body surface.
    f_x = -force_plan.arc_length * np.sum(
        pressure_samples * force_plan.normals_x
    )
    f_y = -force_plan.arc_length * np.sum(
        pressure_samples * force_plan.normals_y
    )
    return float(f_x), float(f_y)


def _compute_forces(
    snapshot_path: str,
    force_plan: SurfaceForcePlan,
) -> Tuple[float, float]:
    """Compute x and y forces from a single snapshot."""
    with np.load(snapshot_path, allow_pickle=False) as data:
        fx_meta = _safe_scalar(data, "meta_ibm_force_x")
        fy_meta = _safe_scalar(data, "meta_ibm_force_y")
        if fx_meta is not None and fy_meta is not None:
            return float(fx_meta), float(fy_meta)
        if "p" not in data.files:
            available = ", ".join(sorted(data.files))
            raise KeyError(
                f"Snapshot {os.path.basename(snapshot_path)!r} is missing field "
                f"'p'. Available fields: {available}"
            )
        p = data["p"]
    return _compute_pressure_forces(p, force_plan)


def _load_grid_faces_for_snapshot(
    snapshot_path: str,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load physical grid faces from prepared-grid metadata when available."""
    indir = os.path.dirname(os.path.abspath(snapshot_path))
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

    with np.load(snapshot_path, allow_pickle=False) as data:
        lx = _safe_scalar(data, "meta_lx")
        ly = _safe_scalar(data, "meta_ly")
        x_min = _safe_scalar(data, "meta_x_min")
        y_min = _safe_scalar(data, "meta_y_min")

    if lx is not None and ly is not None:
        x0 = 0.0 if x_min is None else float(x_min)
        y0 = 0.0 if y_min is None else float(y_min)
        xf = np.linspace(x0, x0 + float(lx), nx + 1)
        yf = np.linspace(y0, y0 + float(ly), ny + 1)
        return xf, yf

    return np.arange(nx + 1, dtype=float), np.arange(ny + 1, dtype=float)


def plot_shedding_spectrum(
    csv_path: str,
    save_name: str = "shedding_spectrum.png",
    t_min: float = 1.0,
    f_min: float = 0.05,
    f_max: float = 2.0,
    char_length: Optional[float] = None,
    u_ref: Optional[float] = None,
) -> None:
    """Plot the Fourier energy spectrum of C_l and mark dominant shedding peaks."""
    plt.switch_backend("Agg")
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    if arr.size == 0:
        raise ValueError(f"No rows found in coefficient file: {csv_path}")

    names = arr.dtype.names or ()
    if "t" not in names or "c_l" not in names:
        raise ValueError(
            f"CSV missing required columns ['t', 'c_l']. Found: {list(names)}"
        )

    t = np.atleast_1d(arr["t"]).astype(float)
    c_l = np.atleast_1d(arr["c_l"]).astype(float)
    out = _compute_fourier_power_spectrum(
        t,
        c_l,
        t_min=t_min,
        f_min=f_min,
        f_max=f_max,
    )
    if out is None:
        raise ValueError("Insufficient oscillatory data to compute a Fourier spectrum.")

    _, _, freq_band, power_band = out
    peak_indices = _find_top_spectral_peaks(freq_band, power_band, max_peaks=3)
    power_floor = max(float(np.max(power_band)) * 1e-12, np.finfo(float).tiny)
    power_display = np.maximum(power_band, power_floor)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(freq_band, power_display, color="tab:blue", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Fourier energy")
    ax.set_title(
        f"Lift Spectrum / Shedding Frequencies ({os.path.basename(csv_path)})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    ymax = float(np.max(power_display)) if power_display.size else 1.0
    for rank, idx in enumerate(peak_indices, start=1):
        f_peak = float(freq_band[idx])
        p_peak = float(power_display[idx])
        ax.scatter([f_peak], [p_peak], color="crimson", zorder=3)
        label = f"#{rank}: f={f_peak:.4g}"
        if char_length is not None and u_ref is not None and u_ref > 0.0:
            st = f_peak * float(char_length) / float(u_ref)
            label += f", St={st:.4g}"
        ax.annotate(
            label,
            xy=(f_peak, p_peak),
            xytext=(8, 8 + 16 * (rank - 1)),
            textcoords="offset points",
            fontsize=9,
            color="crimson",
            arrowprops={"arrowstyle": "-", "color": "crimson", "lw": 0.8},
        )

    ax.set_xlim(float(freq_band[0]), float(freq_band[-1]))
    ax.set_ylim(bottom=power_floor, top=max(1.05 * ymax, power_floor * 10.0))
    fig.tight_layout()

    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(DEFAULT_RESULTS_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def save_pressure_coefficient_report(
    snapshot_path: str,
    u_ref: float,
    save_csv: str = "pressure_coefficient_theta.csv",
    save_plot: str = "pressure_coefficient_theta.png",
    config_path: Optional[str] = None,
    cylinder_center: Optional[Tuple[float, float]] = None,
    cylinder_radius: Optional[float] = None,
    n_samples: int = 720,
) -> None:
    """Save C_p(theta) from one snapshot as CSV and PNG."""
    plt.switch_backend("Agg")
    with np.load(snapshot_path, allow_pickle=False) as data:
        if "p" not in data.files:
            available = ", ".join(sorted(data.files))
            raise KeyError(
                f"Snapshot {os.path.basename(snapshot_path)!r} is missing field "
                f"'p'. Available fields: {available}"
            )
        p = np.asarray(data["p"], dtype=float)

    nx, ny = p.shape
    xf, yf = _load_grid_faces_for_snapshot(snapshot_path, nx=nx, ny=ny)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])

    if cylinder_center is None or cylinder_radius is None:
        geom = _estimate_cylinder_geometry(
            snapshot_path,
            config_path=config_path,
            cylinder_radius=cylinder_radius,
        )
    else:
        geom = CylinderGeometry(
            center_x=float(cylinder_center[0]),
            center_y=float(cylinder_center[1]),
            radius=float(cylinder_radius),
        )

    if u_ref <= 0.0:
        raise ValueError("Reference velocity must be positive for pressure coefficient.")

    force_plan = _build_surface_force_plan(xc, yc, geom, n_samples=n_samples)
    pressure_samples = _sample_surface_pressure(p, force_plan)
    pressure_ref = float(np.mean(pressure_samples))
    c_p = (pressure_samples - pressure_ref) / (0.5 * float(u_ref) ** 2)
    theta_rad = force_plan.theta
    theta_deg = np.rad2deg(theta_rad)

    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(DEFAULT_RESULTS_DIR, save_csv)
    plot_path = os.path.join(DEFAULT_RESULTS_DIR, save_plot)

    out = np.column_stack((theta_deg, theta_rad, pressure_samples, c_p))
    header = "theta_deg,theta_rad,pressure_surface,c_p_zero_mean_surface"
    np.savetxt(csv_path, out, delimiter=",", header=header, comments="")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(theta_deg, c_p, color="tab:purple", linewidth=1.8)
    ax.set_xlabel("Theta [deg]")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(
        f"Surface Pressure Coefficient vs Theta ({os.path.basename(snapshot_path)})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 360.0)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pressure-coefficient CSV: {csv_path}")
    print(f"Saved figure: {plot_path}")


def _filter_valid_snapshots(
    snapshots: List[Tuple[float, str]],
) -> Tuple[List[Tuple[float, str]], List[str]]:
    """Keep only snapshots that include the fields aerodynamic analysis needs."""
    valid: List[Tuple[float, str]] = []
    skipped: List[str] = []

    for t, path in snapshots:
        try:
            with np.load(path, allow_pickle=False) as data:
                has_velocity = "u" in data.files and "v" in data.files
                has_force_meta = (
                    "meta_ibm_force_x" in data.files and
                    "meta_ibm_force_y" in data.files
                )
                has_pressure = "p" in data.files
                if has_velocity and (has_force_meta or has_pressure):
                    valid.append((t, path))
                else:
                    skipped.append(os.path.basename(path))
        except Exception:
            skipped.append(os.path.basename(path))

    return valid, skipped


def _compute_coefficients(
    f_x: float,
    f_y: float,
    u_ref: float,
    char_length: float,
    rho: float = 1.0,
) -> Tuple[float, float]:
    """Convert forces to non-dimensional coefficients."""
    if u_ref <= 0:
        return 0.0, 0.0

    q = 0.5 * rho * u_ref**2
    area = char_length
    c_d = 2.0 * f_x / (q * area) if area > 0 else 0.0
    c_l = 2.0 * f_y / (q * area) if area > 0 else 0.0

    return c_d, c_l


def _extract_combined_series(
    snapshots: List[Tuple[float, str]],
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    probe_x: Optional[float],
    probe_y: Optional[float],
    geom: CylinderGeometry,
    u_ref: float,
    char_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract optional probe series and force/coefficient histories.

    Returns:
        (t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l)
    """
    t, u_probe, v_probe, p_probe = _extract_probe_series(
        snapshots, nx, ny, lx, ly, probe_x, probe_y
    )
    _, xc, _, yc = _build_uniform_face_and_center_coords(nx, ny, lx, ly)
    force_plan = _build_surface_force_plan(xc, yc, geom)

    n = len(snapshots)
    f_x_arr = np.empty(n, dtype=float)
    f_y_arr = np.empty(n, dtype=float)
    c_d_arr = np.empty(n, dtype=float)
    c_l_arr = np.empty(n, dtype=float)

    for k, (_, path) in enumerate(snapshots):
        f_x, f_y = _compute_forces(path, force_plan)
        c_d, c_l = _compute_coefficients(f_x, f_y, u_ref, char_length)
        f_x_arr[k] = f_x
        f_y_arr[k] = f_y
        c_d_arr[k] = c_d
        c_l_arr[k] = c_l

    return t, u_probe, v_probe, p_probe, f_x_arr, f_y_arr, c_d_arr, c_l_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive aerodynamic analysis: Strouhal and drag/lift coefficients.",
    )
    parser.add_argument("--indir", type=str, default="output",
                        help="Snapshot directory (default: output)")
    parser.add_argument("--pattern", type=str, default="snap_*.npz",
                        help="Snapshot filename pattern (default: snap_*.npz)")
    parser.add_argument("--config", type=str, default="config.txt",
                        help="Configuration file (default: config.txt)")

    parser.add_argument("--probe-x", type=float, default=None,
                        help="Optional probe x coordinate in physical units for CSV export")
    parser.add_argument("--probe-y", type=float, default=None,
                        help="Optional probe y coordinate in physical units for CSV export")

    parser.add_argument("--u-ref", type=float, default=1.0,
                        help="Reference velocity U (default: 1.0)")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="Characteristic length L (overrides config; default: read from config or 1.0)")
    parser.add_argument("--use-cylinder-diameter", action="store_true",
                        help="Use L = 2*cylinder_radius from config (or ly/4 if radius is default)")
    parser.add_argument("--cylinder-radius", type=float, default=None,
                        help="Cylinder radius (default: read from config or ly/8)")

    parser.add_argument("--t-min", type=float, default=1.0,
                        help="Ignore data before this time for frequency fit (default: 1.0)")
    parser.add_argument("--f-min", type=float, default=0.05,
                        help="Min search frequency (default: 0.05)")
    parser.add_argument("--f-max", type=float, default=2.0,
                        help="Max search frequency (default: 2.0)")

    parser.add_argument("--save-series", type=str, default=None,
                        help="Optional CSV path for combined time series")
    parser.add_argument("--save-report", type=str,
                        default=os.path.join(
                            DEFAULT_RESULTS_DIR, "aero_report.txt"),
                        help="TXT path for comprehensive report (default: results/aero_report.txt)")

    return parser.parse_args()


def run_analysis(
    indir: str = "output",
    pattern: str = "snap_*.npz",
    config: str = "config.txt",
    probe_x: Optional[float] = None,
    probe_y: Optional[float] = None,
    u_ref: float = 1.0,
    length_scale: Optional[float] = None,
    use_cylinder_diameter: bool = False,
    cylinder_radius: Optional[float] = None,
    t_min: float = 1.0,
    f_min: float = 0.05,
    f_max: float = 2.0,
    save_series: Optional[str] = None,
    save_report: Optional[str] = os.path.join(DEFAULT_RESULTS_DIR, "aero_report.txt"),
) -> int:
    """Run aerodynamic post-processing programmatically."""
    snapshots = _collect_snapshots(indir, pattern)
    if not snapshots:
        print(f"No snapshots found in {indir!r} with pattern {pattern!r}.")
        return 1

    snapshots, skipped = _filter_valid_snapshots(snapshots)
    if skipped:
        preview = ", ".join(skipped[:5])
        suffix = "" if len(skipped) <= 5 else ", ..."
        print(
            f"Skipped {len(skipped)} snapshot(s) missing required fields: "
            f"{preview}{suffix}"
        )
    if not snapshots:
        print("No valid snapshots remain after filtering incomplete files.")
        return 1

    l_char, u_ref, nx, ny, lx, ly = _estimate_scales(
        snapshots[0][1],
        length_scale=length_scale,
        use_cylinder_diameter=use_cylinder_diameter,
        u_ref=u_ref,
        config_path=config,
    )

    geom = _estimate_cylinder_geometry(
        snapshots[0][1],
        config_path=config,
        cylinder_radius=cylinder_radius,
    )

    t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l = _extract_combined_series(
        snapshots,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        probe_x=probe_x,
        probe_y=probe_y,
        geom=geom,
        u_ref=u_ref,
        char_length=l_char,
    )

    if save_series:
        out = np.column_stack((t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l))
        header = "t,u_probe,v_probe,p_probe,f_x,f_y,c_d,c_l"
        np.savetxt(save_series, out, delimiter=",", header=header, comments="")
        print(f"Saved combined series: {save_series}")

    dt = np.diff(t)
    dt_median = float(np.median(dt)) if dt.size else np.nan
    nyquist_est = 0.5 / dt_median if np.isfinite(dt_median) and dt_median > 0 else np.nan

    lift_strouhal: Optional[SpectralResult] = None
    out = _dominant_frequency(
        t,
        c_l,
        t_min=t_min,
        f_min=f_min,
        f_max=f_max,
    )
    if out is not None:
        f_peak, peak_power = out
        lift_strouhal = SpectralResult(
            freq=f_peak,
            peak_power=peak_power,
            st=f_peak * l_char / u_ref,
        )

    c_d_mean = float(np.mean(c_d))
    c_d_std = float(np.std(c_d))
    c_l_mean = float(np.mean(c_l))
    c_l_std = float(np.std(c_l))
    c_l_rms = float(np.sqrt(np.mean(c_l**2)))

    print("=" * 70)
    print("COMPREHENSIVE AERODYNAMIC ANALYSIS")
    print("=" * 70)
    print(f"Snapshots         : {len(snapshots)}")
    print(f"Time span         : [{t.min():.4f}, {t.max():.4f}]")
    if probe_x is not None and probe_y is not None:
        print(f"Probe location    : ({probe_x:.6g}, {probe_y:.6g})")
    print(f"Cylinder center   : ({geom.center_x:.6g}, {geom.center_y:.6g})")
    print(f"Cylinder radius   : {geom.radius:.6g}")
    print(f"Char. length (L)  : {l_char:.6g}")
    print(f"Ref. velocity (U) : {u_ref:.6g}")
    print()
    print("-" * 70)
    print("STROUHAL NUMBER ANALYSIS")
    print("-" * 70)
    print("Signal used       : C_l")
    print(f"Frequency window  : [{f_min:.4f}, {f_max:.4f}]")
    if np.isfinite(nyquist_est):
        print(f"Median dt         : {dt_median:.6g} (Nyquist approx {nyquist_est:.6g})")

    if lift_strouhal is None:
        print("C_l peak         : unavailable (insufficient variation/samples)")
    else:
        edge_note = ""
        if _is_edge_frequency(lift_strouhal.freq, f_min, f_max):
            edge_note = " [edge]"
        print(
            f"C_l peak         : f={lift_strouhal.freq:.6g}, "
            f"St={lift_strouhal.st:.6g}, "
            f"power={lift_strouhal.peak_power:.6g}{edge_note}"
        )
        print("-" * 70)
        print(f"Lift f0           : {lift_strouhal.freq:.6g}")
        print(f"Lift Strouhal     : {lift_strouhal.st:.6g}")
        if np.isfinite(nyquist_est) and lift_strouhal.freq > 0.8 * nyquist_est:
            print("WARNING: Estimated f0 is close to Nyquist limit.")
            print("         Use smaller save_dt for confidence.")

    print()
    print("-" * 70)
    print("DRAG AND LIFT COEFFICIENT ANALYSIS")
    print("-" * 70)
    print(f"C_d mean          : {c_d_mean:.6g}")
    print(f"C_d std           : {c_d_std:.6g}")
    print(f"C_l mean          : {c_l_mean:.6g}")
    print(f"C_l std           : {c_l_std:.6g}")
    print(f"C_l rms           : {c_l_rms:.6g}")

    if save_report:
        report_dir = os.path.dirname(save_report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        lines: List[str] = [
            "COMPREHENSIVE AERODYNAMIC ANALYSIS",
            "=" * 70,
            f"Snapshots         : {len(snapshots)}",
            f"Time span         : [{t.min():.4f}, {t.max():.4f}]",
            f"Cylinder center   : ({geom.center_x:.6g}, {geom.center_y:.6g})",
            f"Cylinder radius   : {geom.radius:.6g}",
            f"Char. length (L)  : {l_char:.6g}",
            f"Ref. velocity (U) : {u_ref:.6g}",
            "",
            "-" * 70,
            "STROUHAL NUMBER ANALYSIS",
            "-" * 70,
            "Signal used       : C_l",
            f"Frequency window  : [{f_min:.4f}, {f_max:.4f}]",
        ]

        if probe_x is not None and probe_y is not None:
            lines.insert(4, f"Probe location    : ({probe_x:.6g}, {probe_y:.6g})")

        if np.isfinite(nyquist_est):
            lines.append(
                f"Median dt         : {dt_median:.6g} "
                f"(Nyquist approx {nyquist_est:.6g})"
            )

        if lift_strouhal is None:
            lines.append("C_l peak         : unavailable (insufficient variation/samples)")
        else:
            edge_note = ""
            if _is_edge_frequency(lift_strouhal.freq, f_min, f_max):
                edge_note = " [edge]"
            lines.append(
                f"C_l peak         : f={lift_strouhal.freq:.6g}, "
                f"St={lift_strouhal.st:.6g}, "
                f"power={lift_strouhal.peak_power:.6g}{edge_note}"
            )
            lines.append("")
            lines.append(f"Lift f0           : {lift_strouhal.freq:.6g}")
            lines.append(f"Lift Strouhal     : {lift_strouhal.st:.6g}")

        lines.extend([
            "",
            "-" * 70,
            "DRAG AND LIFT COEFFICIENT ANALYSIS",
            "-" * 70,
            f"C_d mean          : {c_d_mean:.6g}",
            f"C_d std           : {c_d_std:.6g}",
            f"C_l mean          : {c_l_mean:.6g}",
            f"C_l std           : {c_l_std:.6g}",
            f"C_l rms           : {c_l_rms:.6g}",
        ])

        with open(save_report, "w", encoding="utf-8") as fout:
            fout.write("\n".join(lines) + "\n")

        print(f"\nSaved comprehensive report: {save_report}")

    return 0


def main() -> int:
    args = parse_args()
    return run_analysis(
        indir=args.indir,
        pattern=args.pattern,
        config=args.config,
        probe_x=args.probe_x,
        probe_y=args.probe_y,
        u_ref=args.u_ref,
        length_scale=args.length_scale,
        use_cylinder_diameter=args.use_cylinder_diameter,
        cylinder_radius=args.cylinder_radius,
        t_min=args.t_min,
        f_min=args.f_min,
        f_max=args.f_max,
        save_series=args.save_series,
        save_report=args.save_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
