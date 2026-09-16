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


@dataclass(frozen=True)
class SurfaceForceDiagnostics:
    pressure_cp_min: float
    pressure_cp_max: float
    pressure_cd: float
    pressure_cl: float
    sample_spacing: float
    sample_offset: float


@dataclass(frozen=True)
class SurfaceForceComponents:
    pressure_fx: float
    pressure_fy: float
    viscous_fx: float
    viscous_fy: float
    sample_spacing: float
    sample_offset: float


_LAST_SURFACE_DIAGNOSTICS: Optional[SurfaceForceDiagnostics] = None


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
    with np.load(first_path, allow_pickle=False) as data:
        radius_meta = _safe_scalar(data, "meta_cylinder_radius")
    cylinder_radius = (
        float(radius_meta) if radius_meta is not None
        else _resolve_cylinder_radius(config, ly)
    )

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
    with np.load(first_path, allow_pickle=False) as data:
        center_x_meta = _safe_scalar(data, "meta_cylinder_center_x")
        center_y_meta = _safe_scalar(data, "meta_cylinder_center_y")
        radius_meta = _safe_scalar(data, "meta_cylinder_radius")

    if cylinder_radius is not None:
        r = float(cylinder_radius)
    elif radius_meta is not None:
        r = float(radius_meta)
    else:
        r = _resolve_cylinder_radius(config, ly)

    center_x_cfg, center_y_cfg = _resolve_cylinder_center(config, lx, ly)
    center_x = center_x_cfg if center_x_meta is None else float(center_x_meta)
    center_y = center_y_cfg if center_y_meta is None else float(center_y_meta)

    return CylinderGeometry(
        center_x=center_x,
        center_y=center_y,
        radius=r,
    )


def _estimate_kinematic_viscosity(
    first_path: str,
    config_path: Optional[str],
    u_ref: float,
    geom: CylinderGeometry,
    char_length: float,
) -> float:
    with np.load(first_path, allow_pickle=False) as data:
        re_meta = _safe_scalar(data, "meta_re")
        re_is_d_meta = _safe_scalar(data, "meta_re_is_cylinder_based")

    config = _read_config(config_path) if config_path else {}
    # Snapshot metadata describes the data being post-processed; prefer it over
    # the current config file so archived Re sweeps are not reinterpreted after
    # config.txt changes.
    re_value = float(re_meta if re_meta is not None else config.get("re", 100.0))
    if re_value <= 0.0:
        raise ValueError("Reynolds number must be positive for surface forces.")

    re_is_d = (
        bool(re_is_d_meta)
        if re_is_d_meta is not None
        else bool(config.get("re_is_cylinder_based", 0.0))
    )
    length = 2.0 * geom.radius if re_is_d else char_length
    return float(u_ref) * float(length) / re_value


def _snapshot_cylinder_geometry(
    data: np.lib.npyio.NpzFile,
    fallback: CylinderGeometry,
) -> CylinderGeometry:
    """Return per-snapshot cylinder geometry when moving-body metadata exists."""
    center_x = _safe_scalar(data, "meta_cylinder_center_x")
    center_y = _safe_scalar(data, "meta_cylinder_center_y")
    radius = _safe_scalar(data, "meta_cylinder_radius")
    return CylinderGeometry(
        center_x=fallback.center_x if center_x is None else float(center_x),
        center_y=fallback.center_y if center_y is None else float(center_y),
        radius=fallback.radius if radius is None else float(radius),
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


def _sample_bilinear(
    values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
) -> float:
    return _apply_bilinear_plan(values, _build_bilinear_plan(x_grid, y_grid, x, y))


def _sample_bilinear_points(
    values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear interpolation for many points on one grid."""
    i = np.searchsorted(x_grid, x) - 1
    j = np.searchsorted(y_grid, y) - 1
    i = np.clip(i, 0, len(x_grid) - 2).astype(int)
    j = np.clip(j, 0, len(y_grid) - 2).astype(int)

    x0 = x_grid[i]
    x1 = x_grid[i + 1]
    y0 = y_grid[j]
    y1 = y_grid[j + 1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x, dtype=float), where=x1 != x0)
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y, dtype=float), where=y1 != y0)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    return (
        (1.0 - tx) * (1.0 - ty) * values[i, j]
        + tx * (1.0 - ty) * values[i + 1, j]
        + (1.0 - tx) * ty * values[i, j + 1]
        + tx * ty * values[i + 1, j + 1]
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


def _cell_center_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (u[:-1, :] + u[1:, :]), 0.5 * (v[:, :-1] + v[:, 1:])


def _local_spacing_near_body(
    coords: np.ndarray,
    center: float,
    radius: float,
) -> float:
    """Estimate representative spacing near the immersed-body surface."""
    if len(coords) <= 1:
        return 1.0
    diffs = np.abs(np.diff(coords))
    median_spacing = float(np.median(diffs))
    band_half_width = float(radius) + 2.0 * median_spacing
    segment_midpoints = 0.5 * (coords[:-1] + coords[1:])
    near_body = np.abs(segment_midpoints - float(center)) <= band_half_width
    if np.any(near_body):
        return float(np.median(diffs[near_body]))
    return median_spacing


def _compute_surface_force_components(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    geom: CylinderGeometry,
    nu: float,
    n_samples: int = 720,
    sample_offset_factor: float = 0.5,
) -> SurfaceForceComponents:
    """Integrate pressure and viscous traction components on a circle.

    This is an experimental Cartesian-grid postprocessor for IBM snapshots.
    It samples fields slightly outside the immersed surface to avoid using
    solid-interior values in the finite-difference velocity gradients.
    """
    global _LAST_SURFACE_DIAGNOSTICS
    if nu <= 0.0:
        raise ValueError("Kinematic viscosity must be positive for surface forces.")

    u_c, v_c = _cell_center_velocity(u, v)
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    n_x = np.cos(theta)
    n_y = np.sin(theta)

    dx_local = _local_spacing_near_body(xc, geom.center_x, geom.radius)
    dy_local = _local_spacing_near_body(yc, geom.center_y, geom.radius)
    h = max(min(abs(dx_local), abs(dy_local)), np.finfo(float).eps)
    sample_offset = max(float(sample_offset_factor), 0.0) * h
    arc_length = 2.0 * np.pi * geom.radius / float(n_samples)

    x0 = geom.center_x + (geom.radius + sample_offset) * n_x
    y0 = geom.center_y + (geom.radius + sample_offset) * n_y

    pressure_samples = _sample_bilinear_points(p, xc, yc, x0, y0)
    du_dx = (
        _sample_bilinear_points(u_c, xc, yc, x0 + h, y0)
        - _sample_bilinear_points(u_c, xc, yc, x0 - h, y0)
    ) / (2.0 * h)
    du_dy = (
        _sample_bilinear_points(u_c, xc, yc, x0, y0 + h)
        - _sample_bilinear_points(u_c, xc, yc, x0, y0 - h)
    ) / (2.0 * h)
    dv_dx = (
        _sample_bilinear_points(v_c, xc, yc, x0 + h, y0)
        - _sample_bilinear_points(v_c, xc, yc, x0 - h, y0)
    ) / (2.0 * h)
    dv_dy = (
        _sample_bilinear_points(v_c, xc, yc, x0, y0 + h)
        - _sample_bilinear_points(v_c, xc, yc, x0, y0 - h)
    ) / (2.0 * h)

    tau_xx = 2.0 * nu * du_dx
    tau_yy = 2.0 * nu * dv_dy
    tau_xy = nu * (du_dy + dv_dx)

    pressure_centered = pressure_samples - np.mean(pressure_samples)
    pressure_traction_x = -pressure_centered * n_x
    pressure_traction_y = -pressure_centered * n_y
    pressure_fx = float(arc_length * np.sum(pressure_traction_x))
    pressure_fy = float(arc_length * np.sum(pressure_traction_y))
    _LAST_SURFACE_DIAGNOSTICS = SurfaceForceDiagnostics(
        pressure_cp_min=float(2.0 * np.min(pressure_centered)),
        pressure_cp_max=float(2.0 * np.max(pressure_centered)),
        pressure_cd=float(pressure_fx / 0.5),
        pressure_cl=float(pressure_fy / 0.5),
        sample_spacing=float(h),
        sample_offset=float(sample_offset),
    )

    traction_x = tau_xx * n_x + tau_xy * n_y
    traction_y = tau_xy * n_x + tau_yy * n_y
    viscous_fx = float(arc_length * np.sum(traction_x))
    viscous_fy = float(arc_length * np.sum(traction_y))

    return SurfaceForceComponents(
        pressure_fx=pressure_fx,
        pressure_fy=pressure_fy,
        viscous_fx=viscous_fx,
        viscous_fy=viscous_fy,
        sample_spacing=float(h),
        sample_offset=float(sample_offset),
    )


def _compute_surface_stress_forces(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    geom: CylinderGeometry,
    nu: float,
    n_samples: int = 720,
    include_pressure: bool = False,
    sample_offset_factor: float = 0.5,
) -> Tuple[float, float]:
    """Integrate viscous, or pressure plus viscous, traction on a circle.

    Use force_source="surface-full" for the physical pressure plus viscous
    estimate; force_source="surface" is a viscous-only diagnostic.
    """
    components = _compute_surface_force_components(
        u,
        v,
        p,
        xc,
        yc,
        geom,
        nu=nu,
        n_samples=n_samples,
        sample_offset_factor=sample_offset_factor,
    )
    traction_x = components.viscous_fx
    traction_y = components.viscous_fy
    if include_pressure:
        traction_x += components.pressure_fx
        traction_y += components.pressure_fy

    return traction_x, traction_y


def _compute_forces(
    snapshot_path: str,
    xc: np.ndarray,
    yc: np.ndarray,
    geom: CylinderGeometry,
    force_source: str = "ibm",
    nu: Optional[float] = None,
    surface_sample_offset_factor: float = 0.5,
) -> Tuple[float, float]:
    """Compute x and y forces from a single snapshot."""
    with np.load(snapshot_path, allow_pickle=False) as data:
        fx_meta = _safe_scalar(data, "meta_ibm_force_x")
        fy_meta = _safe_scalar(data, "meta_ibm_force_y")
        if force_source == "ibm" and fx_meta is not None and fy_meta is not None:
            return float(fx_meta), float(fy_meta)
        required = {"p"} if force_source == "pressure" else {"u", "v", "p"}
        missing = sorted(required.difference(data.files))
        if missing:
            available = ", ".join(sorted(data.files))
            raise KeyError(
                f"Snapshot {os.path.basename(snapshot_path)!r} is missing field(s) "
                f"{missing}. Available fields: {available}"
            )
        p = np.array(data["p"], copy=True)
        u = np.array(data["u"], copy=True) if "u" in required else None
        v = np.array(data["v"], copy=True) if "v" in required else None
        snapshot_geom = _snapshot_cylinder_geometry(data, geom)

    force_plan = _build_surface_force_plan(xc, yc, snapshot_geom)
    if force_source == "pressure":
        return _compute_pressure_forces(p, force_plan)
    if force_source in {"surface", "surface-full"}:
        if nu is None:
            raise ValueError(
                f"force_source={force_source!r} requires a kinematic viscosity."
            )
        return _compute_surface_stress_forces(
            u,
            v,
            p,
            xc,
            yc,
            snapshot_geom,
            nu=nu,
            include_pressure=(force_source == "surface-full"),
            sample_offset_factor=surface_sample_offset_factor,
        )
    return _compute_pressure_forces(p, force_plan)


def _snapshot_has_ibm_force_metadata(snapshot_path: str) -> bool:
    """Return whether a snapshot carries direct IBM force diagnostics."""
    try:
        with np.load(snapshot_path, allow_pickle=False) as data:
            return (
                _safe_scalar(data, "meta_ibm_force_x") is not None
                and _safe_scalar(data, "meta_ibm_force_y") is not None
            )
    except Exception:
        return False


def _snapshot_has_fields(snapshot_path: str, required: set[str]) -> bool:
    try:
        with np.load(snapshot_path, allow_pickle=False) as data:
            return required.issubset(set(data.files))
    except Exception:
        return False


def _resolve_force_source(force_source: str, snapshots: List[Tuple[float, str]]) -> str:
    """Resolve automatic force source for available snapshot diagnostics."""
    source = str(force_source).strip().lower()
    if source not in {"auto", "ibm", "pressure", "surface", "surface-full"}:
        raise ValueError(
            "force_source must be 'auto', 'ibm', 'pressure', 'surface', or 'surface-full', "
            f"got {force_source!r}"
        )
    if source in {"pressure", "surface", "surface-full"}:
        return source
    if not snapshots:
        return "pressure"

    first_snapshot = snapshots[0][1]
    if source == "ibm":
        if _snapshot_has_ibm_force_metadata(first_snapshot):
            return "ibm"
        return "pressure"

    if _snapshot_has_fields(first_snapshot, {"u", "v", "p"}):
        return "surface-full"
    if _snapshot_has_ibm_force_metadata(first_snapshot):
        return "ibm"
    return "pressure"


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
    mark_peaks: bool = False,
    show_title: bool = True,
) -> None:
    """Plot the Fourier energy spectrum of C_l."""
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
        raise ValueError(
            "Insufficient oscillatory data to compute a Fourier spectrum.")

    _, _, freq_band, power_band = out
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(freq_band, power_band, color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Fourier energy")
    if show_title:
        ax.set_title(
            f"Lift Spectrum / Shedding Frequencies ({os.path.basename(csv_path)})",
            fontsize=12,
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3)

    ymax = float(np.max(power_band)) if power_band.size else 1.0
    if mark_peaks:
        peak_indices = _find_top_spectral_peaks(freq_band, power_band, max_peaks=3)
        for rank, idx in enumerate(peak_indices, start=1):
            f_peak = float(freq_band[idx])
            p_peak = float(power_band[idx])
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
    ax.set_ylim(
        bottom=max(float(np.min(power_band)), np.finfo(float).eps),
        top=max(1.05 * ymax, np.finfo(float).eps),
    )
    fig.tight_layout()

    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(DEFAULT_RESULTS_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def plot_drag_decomposition(
    csv_path: str,
    save_name: str = "drag_decomposition.png",
    t_min: Optional[float] = None,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> str:
    """Plot pressure and viscous drag/lift coefficient components."""
    plt.switch_backend("Agg")
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    if arr.size == 0:
        raise ValueError(f"No rows found in drag decomposition file: {csv_path}")

    names = arr.dtype.names or ()
    required = (
        "t",
        "pressure_c_d",
        "viscous_c_d",
        "total_c_d",
        "pressure_c_l",
        "viscous_c_l",
        "total_c_l",
    )
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            f"CSV missing required columns {missing}. Found: {list(names)}"
        )

    t = np.atleast_1d(arr["t"]).astype(float)
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None:
        mask = t >= float(t_min)
    if not np.any(mask):
        raise ValueError("No drag decomposition samples left after t_min trimming.")

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_name)

    fig, (ax_drag, ax_lift) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_drag.plot(t[mask], np.atleast_1d(arr["total_c_d"])[mask],
                 color="black", linewidth=1.8, label="total")
    ax_drag.plot(t[mask], np.atleast_1d(arr["pressure_c_d"])[mask],
                 color="tab:blue", linewidth=1.4, label="pressure")
    ax_drag.plot(t[mask], np.atleast_1d(arr["viscous_c_d"])[mask],
                 color="tab:red", linewidth=1.4, label="viscous")
    ax_drag.set_ylabel("C_d")
    ax_drag.set_title(
        f"Pressure/Viscous Force Decomposition ({os.path.basename(csv_path)})",
        fontsize=12,
        fontweight="bold",
    )
    ax_drag.grid(True, alpha=0.3)
    ax_drag.legend(loc="best")

    ax_lift.plot(t[mask], np.atleast_1d(arr["total_c_l"])[mask],
                 color="black", linewidth=1.8, label="total")
    ax_lift.plot(t[mask], np.atleast_1d(arr["pressure_c_l"])[mask],
                 color="tab:blue", linewidth=1.4, label="pressure")
    ax_lift.plot(t[mask], np.atleast_1d(arr["viscous_c_l"])[mask],
                 color="tab:red", linewidth=1.4, label="viscous")
    ax_lift.set_xlabel("time")
    ax_lift.set_ylabel("C_l")
    ax_lift.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")
    return save_path


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
        raise ValueError(
            "Reference velocity must be positive for pressure coefficient.")

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
    projected_area = char_length
    c_d = f_x / (q * projected_area) if projected_area > 0 else 0.0
    c_l = f_y / (q * projected_area) if projected_area > 0 else 0.0

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
    force_source: str = "ibm",
    nu: Optional[float] = None,
    surface_sample_offset_factor: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract optional probe series and force/coefficient histories.

    Returns:
        (t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l)
    """
    t, u_probe, v_probe, p_probe = _extract_probe_series(
        snapshots, nx, ny, lx, ly, probe_x, probe_y
    )
    xf, yf = _load_grid_faces_for_snapshot(snapshots[0][1], nx=nx, ny=ny)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])

    n = len(snapshots)
    f_x_arr = np.empty(n, dtype=float)
    f_y_arr = np.empty(n, dtype=float)
    c_d_arr = np.empty(n, dtype=float)
    c_l_arr = np.empty(n, dtype=float)

    for k, (_, path) in enumerate(snapshots):
        f_x, f_y = _compute_forces(
            path,
            xc=xc,
            yc=yc,
            geom=geom,
            force_source=force_source,
            nu=nu,
            surface_sample_offset_factor=surface_sample_offset_factor,
        )
        c_d, c_l = _compute_coefficients(f_x, f_y, u_ref, char_length)
        f_x_arr[k] = f_x
        f_y_arr[k] = f_y
        c_d_arr[k] = c_d
        c_l_arr[k] = c_l

    return t, u_probe, v_probe, p_probe, f_x_arr, f_y_arr, c_d_arr, c_l_arr


def _compute_force_decomposition(
    snapshot_path: str,
    xc: np.ndarray,
    yc: np.ndarray,
    geom: CylinderGeometry,
    nu: float,
    surface_sample_offset_factor: float = 0.5,
) -> SurfaceForceComponents:
    """Compute pressure and viscous surface-force components from one snapshot."""
    with np.load(snapshot_path, allow_pickle=False) as data:
        required = {"u", "v", "p"}
        missing = sorted(required.difference(data.files))
        if missing:
            available = ", ".join(sorted(data.files))
            raise KeyError(
                f"Snapshot {os.path.basename(snapshot_path)!r} is missing field(s) "
                f"{missing}. Available fields: {available}"
            )
        u = np.array(data["u"], copy=True)
        v = np.array(data["v"], copy=True)
        p = np.array(data["p"], copy=True)
        snapshot_geom = _snapshot_cylinder_geometry(data, geom)

    return _compute_surface_force_components(
        u,
        v,
        p,
        xc,
        yc,
        snapshot_geom,
        nu=nu,
        sample_offset_factor=surface_sample_offset_factor,
    )


def _extract_force_decomposition_series(
    snapshots: List[Tuple[float, str]],
    nx: int,
    ny: int,
    geom: CylinderGeometry,
    u_ref: float,
    char_length: float,
    nu: float,
    surface_sample_offset_factor: float = 0.5,
) -> tuple[np.ndarray, ...]:
    """Extract pressure/viscous force and coefficient histories."""
    xf, yf = _load_grid_faces_for_snapshot(snapshots[0][1], nx=nx, ny=ny)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])

    n = len(snapshots)
    t = np.array([time for time, _ in snapshots], dtype=float)
    pressure_fx = np.empty(n, dtype=float)
    pressure_fy = np.empty(n, dtype=float)
    viscous_fx = np.empty(n, dtype=float)
    viscous_fy = np.empty(n, dtype=float)
    sample_spacing = np.empty(n, dtype=float)
    sample_offset = np.empty(n, dtype=float)

    for k, (_, path) in enumerate(snapshots):
        components = _compute_force_decomposition(
            path,
            xc=xc,
            yc=yc,
            geom=geom,
            nu=nu,
            surface_sample_offset_factor=surface_sample_offset_factor,
        )
        pressure_fx[k] = components.pressure_fx
        pressure_fy[k] = components.pressure_fy
        viscous_fx[k] = components.viscous_fx
        viscous_fy[k] = components.viscous_fy
        sample_spacing[k] = components.sample_spacing
        sample_offset[k] = components.sample_offset

    total_fx = pressure_fx + viscous_fx
    total_fy = pressure_fy + viscous_fy
    pressure_cd, pressure_cl = _compute_coefficients(
        pressure_fx, pressure_fy, u_ref, char_length
    )
    viscous_cd, viscous_cl = _compute_coefficients(
        viscous_fx, viscous_fy, u_ref, char_length
    )
    total_cd, total_cl = _compute_coefficients(
        total_fx, total_fy, u_ref, char_length
    )

    return (
        t,
        pressure_fx,
        pressure_fy,
        viscous_fx,
        viscous_fy,
        total_fx,
        total_fy,
        pressure_cd,
        pressure_cl,
        viscous_cd,
        viscous_cl,
        total_cd,
        total_cl,
        sample_spacing,
        sample_offset,
    )


def _drop_final_endpoint_sample(
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Drop the final saved endpoint sample, which is often a restart artifact."""
    if not arrays or len(arrays[0]) <= 1:
        return arrays
    return tuple(arr[:-1] for arr in arrays)


def _coefficient_window_stats(
    t: np.ndarray,
    c_d: np.ndarray,
    c_l: np.ndarray,
    start: float,
) -> Optional[dict[str, float]]:
    mask = t >= start
    if not np.any(mask):
        return None
    t_window = t[mask]
    c_d_window = c_d[mask]
    c_l_window = c_l[mask]
    c_l_min = float(np.min(c_l_window))
    c_l_max = float(np.max(c_l_window))
    return {
        "start": float(start),
        "t_min": float(t_window.min()),
        "t_max": float(t_window.max()),
        "c_d_mean": float(np.mean(c_d_window)),
        "c_d_std": float(np.std(c_d_window)),
        "c_d_min": float(np.min(c_d_window)),
        "c_d_max": float(np.max(c_d_window)),
        "c_l_mean": float(np.mean(c_l_window)),
        "c_l_std": float(np.std(c_l_window)),
        "c_l_rms": float(np.sqrt(np.mean(c_l_window**2))),
        "c_l_min": c_l_min,
        "c_l_max": c_l_max,
        "c_l_abs_max": float(np.max(np.abs(c_l_window))),
        "c_l_amp_half_range": 0.5 * (c_l_max - c_l_min),
    }


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
    parser.add_argument("--force-source", choices=("auto", "ibm", "pressure", "surface", "surface-full"), default="auto",
                        help=(
                            "Force source: auto prefers physical surface-full "
                            "integration when velocity/pressure fields are available "
                            "(default); ibm uses saved direct-forcing metadata; "
                            "pressure integrates pressure only around "
                            "the per-snapshot cylinder surface; surface integrates "
                            "experimental viscous surface stress only; surface-full "
                            "uses pressure plus viscous surface stress"
                        ))
    parser.add_argument("--surface-sample-offset-factor", type=float, default=0.5,
                        help=(
                            "Surface-force contour offset in local grid spacings "
                            "(default: 0.5)"
                        ))

    parser.add_argument("--t-min", type=float, default=1.0,
                        help="Ignore data before this time for frequency fit (default: 1.0)")
    parser.add_argument("--stats-t-min", type=float, default=None,
                        help=(
                            "Ignore data before this time for settled drag/lift "
                            "statistics. Values between 0 and 1 are interpreted "
                            "as a fraction of the saved time span; 0.70 means "
                            "use the last 30%%. Default: no extra settled section."
                        ))
    parser.add_argument("--keep-final-sample", action="store_true",
                        help="Keep the final endpoint sample in CSV/statistics")
    parser.add_argument("--f-min", type=float, default=0.05,
                        help="Min search frequency (default: 0.05)")
    parser.add_argument("--f-max", type=float, default=2.0,
                        help="Max search frequency (default: 2.0)")

    parser.add_argument("--save-series", type=str, default=None,
                        help="Optional CSV path for combined time series")
    parser.add_argument("--save-drag-decomposition", type=str, default=None,
                        help=(
                            "Optional CSV path for pressure/viscous force and "
                            "coefficient components"
                        ))
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
    save_drag_decomposition: Optional[str] = None,
    save_report: Optional[str] = os.path.join(
        DEFAULT_RESULTS_DIR, "aero_report.txt"),
    force_source: str = "auto",
    drop_final_sample: bool = True,
    surface_sample_offset_factor: float = 0.5,
    stats_t_min: Optional[float] = None,
) -> int:
    """Run aerodynamic post-processing programmatically."""
    global _LAST_SURFACE_DIAGNOSTICS
    _LAST_SURFACE_DIAGNOSTICS = None

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

    force_source = _resolve_force_source(force_source, snapshots)

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
    nu = None
    if force_source in {"surface", "surface-full"}:
        nu = _estimate_kinematic_viscosity(
            snapshots[0][1],
            config,
            u_ref=u_ref,
            geom=geom,
            char_length=l_char,
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
        force_source=force_source,
        nu=nu,
        surface_sample_offset_factor=surface_sample_offset_factor,
    )
    raw_sample_count = len(t)
    if drop_final_sample:
        t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l = _drop_final_endpoint_sample(
            t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l
        )
        if len(t) == 0:
            print("No samples remain after dropping the final endpoint sample.")
            return 1

    if save_series:
        out = np.column_stack(
            (t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l))
        header = "t,u_probe,v_probe,p_probe,f_x,f_y,c_d,c_l"
        np.savetxt(save_series, out, delimiter=",", header=header, comments="")
        print(f"Saved combined series: {save_series}")

    drag_decomposition_stats = None
    if save_drag_decomposition:
        if force_source not in {"surface", "surface-full"}:
            raise ValueError(
                "Drag decomposition requires surface-force post-processing "
                "with velocity and pressure snapshots."
            )
        if nu is None:
            raise ValueError("Drag decomposition requires a kinematic viscosity.")
        decomposition = _extract_force_decomposition_series(
            snapshots,
            nx=nx,
            ny=ny,
            geom=geom,
            u_ref=u_ref,
            char_length=l_char,
            nu=nu,
            surface_sample_offset_factor=surface_sample_offset_factor,
        )
        if drop_final_sample:
            decomposition = _drop_final_endpoint_sample(*decomposition)

        (
            t_decomp,
            pressure_fx,
            pressure_fy,
            viscous_fx,
            viscous_fy,
            total_fx,
            total_fy,
            pressure_cd,
            pressure_cl,
            viscous_cd,
            viscous_cl,
            total_cd,
            total_cl,
            sample_spacing,
            sample_offset,
        ) = decomposition
        out = np.column_stack(
            (
                t_decomp,
                pressure_fx,
                pressure_fy,
                viscous_fx,
                viscous_fy,
                total_fx,
                total_fy,
                pressure_cd,
                pressure_cl,
                viscous_cd,
                viscous_cl,
                total_cd,
                total_cl,
                sample_spacing,
                sample_offset,
            )
        )
        header = (
            "t,pressure_fx,pressure_fy,viscous_fx,viscous_fy,"
            "total_fx,total_fy,pressure_c_d,pressure_c_l,"
            "viscous_c_d,viscous_c_l,total_c_d,total_c_l,"
            "sample_spacing,sample_offset"
        )
        decomp_dir = os.path.dirname(save_drag_decomposition)
        if decomp_dir:
            os.makedirs(decomp_dir, exist_ok=True)
        np.savetxt(
            save_drag_decomposition,
            out,
            delimiter=",",
            header=header,
            comments="",
        )
        drag_decomposition_stats = {
            "pressure_c_d_mean": float(np.mean(pressure_cd)),
            "viscous_c_d_mean": float(np.mean(viscous_cd)),
            "total_c_d_mean": float(np.mean(total_cd)),
            "pressure_c_d_min": float(np.min(pressure_cd)),
            "pressure_c_d_max": float(np.max(pressure_cd)),
            "viscous_c_d_min": float(np.min(viscous_cd)),
            "viscous_c_d_max": float(np.max(viscous_cd)),
        }
        print(f"Saved drag decomposition: {save_drag_decomposition}")

    dt = np.diff(t)
    dt_median = float(np.median(dt)) if dt.size else np.nan
    nyquist_est = 0.5 / \
        dt_median if np.isfinite(dt_median) and dt_median > 0 else np.nan

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

    primary_stats = _coefficient_window_stats(t, c_d, c_l, t_min)
    if primary_stats is None:
        print(
            f"No samples at or after t_min={t_min:.6g}; "
            "cannot compute coefficient statistics."
        )
        return 1
    def _resolve_stats_start(raw_start: float) -> float:
        value = float(raw_start)
        if 0.0 < value < 1.0:
            return float(t.min() + value * (t.max() - t.min()))
        return value

    settled_stats = None
    settled_stats_start = None
    if stats_t_min is not None:
        settled_stats_start = _resolve_stats_start(float(stats_t_min))
        if not np.isclose(settled_stats_start, float(t_min)):
            settled_stats = _coefficient_window_stats(t, c_d, c_l, settled_stats_start)

    def _print_stats(stats: dict[str, float]) -> None:
        print(f"Stats window      : [{stats['t_min']:.4f}, {stats['t_max']:.4f}]")
        print(f"C_d mean          : {stats['c_d_mean']:.6g}")
        print(f"C_d range         : [{stats['c_d_min']:.6g}, {stats['c_d_max']:.6g}]")
        print(f"C_l mean          : {stats['c_l_mean']:.6g}")
        print(f"C_l rms           : {stats['c_l_rms']:.6g}")
        print(f"C_l range         : [{stats['c_l_min']:.6g}, {stats['c_l_max']:.6g}]")

    def _stats_lines(stats: dict[str, float]) -> List[str]:
        return [
            f"Stats window      : [{stats['t_min']:.4f}, {stats['t_max']:.4f}]",
            f"C_d mean          : {stats['c_d_mean']:.6g}",
            f"C_d range         : [{stats['c_d_min']:.6g}, {stats['c_d_max']:.6g}]",
            f"C_l mean          : {stats['c_l_mean']:.6g}",
            f"C_l rms           : {stats['c_l_rms']:.6g}",
            f"C_l range         : [{stats['c_l_min']:.6g}, {stats['c_l_max']:.6g}]",
        ]

    print("=" * 70)
    print("COMPREHENSIVE AERODYNAMIC ANALYSIS")
    print("=" * 70)
    print(f"Snapshots         : {len(snapshots)}")
    if drop_final_sample and raw_sample_count != len(t):
        print("Endpoint trim     : dropped final sample")
    print(f"Time span         : [{t.min():.4f}, {t.max():.4f}]")
    if probe_x is not None and probe_y is not None:
        print(f"Probe location    : ({probe_x:.6g}, {probe_y:.6g})")
    print(f"Cylinder center   : ({geom.center_x:.6g}, {geom.center_y:.6g})")
    print(f"Cylinder radius   : {geom.radius:.6g}")
    print(f"Force source      : {force_source}")
    if nu is not None:
        print(f"Kinematic visc.   : {nu:.6g}")
        if _LAST_SURFACE_DIAGNOSTICS is not None:
            print(
                "Pressure sample   : "
                f"Cp=[{_LAST_SURFACE_DIAGNOSTICS.pressure_cp_min:.6g}, "
                f"{_LAST_SURFACE_DIAGNOSTICS.pressure_cp_max:.6g}], "
                f"pressure-only Cd={_LAST_SURFACE_DIAGNOSTICS.pressure_cd:.6g}"
            )
            print(
                "Surface contour   : "
                f"h={_LAST_SURFACE_DIAGNOSTICS.sample_spacing:.6g}, "
                f"offset={_LAST_SURFACE_DIAGNOSTICS.sample_offset:.6g}"
            )
            if max(
                abs(_LAST_SURFACE_DIAGNOSTICS.pressure_cp_min),
                abs(_LAST_SURFACE_DIAGNOSTICS.pressure_cp_max),
            ) > 100.0:
                print("WARNING          : surface pressure samples are not physical.")
    if drag_decomposition_stats is not None:
        print(
            "Drag decomposition: "
            f"C_d,p={drag_decomposition_stats['pressure_c_d_mean']:.6g}, "
            f"C_d,v={drag_decomposition_stats['viscous_c_d_mean']:.6g}, "
            f"C_d,total={drag_decomposition_stats['total_c_d_mean']:.6g}"
        )
    print(f"Char. length (L)  : {l_char:.6g}")
    print(f"Ref. velocity (U) : {u_ref:.6g}")
    print()
    print("-" * 70)
    print("STROUHAL NUMBER ANALYSIS")
    print("-" * 70)
    print("Signal used       : C_l")
    print(f"Frequency window  : [{f_min:.4f}, {f_max:.4f}]")
    if np.isfinite(nyquist_est):
        print(
            f"Median dt         : {dt_median:.6g} (Nyquist approx {nyquist_est:.6g})")

    if lift_strouhal is None:
        print("C_l spectral peak: unavailable (insufficient variation/samples)")
    else:
        edge_note = ""
        if _is_edge_frequency(lift_strouhal.freq, f_min, f_max):
            edge_note = " [edge]"
        print(
            f"C_l spectral peak: f={lift_strouhal.freq:.6g}, "
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
    _print_stats(primary_stats)
    if settled_stats is not None:
        print()
        print("-" * 70)
        print("SETTLED COEFFICIENT STATISTICS")
        print("-" * 70)
        _print_stats(settled_stats)
    elif settled_stats_start is not None:
        print(f"Settled stats     : unavailable for t >= {settled_stats_start:.6g}")

    if save_report:
        report_dir = os.path.dirname(save_report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        lines: List[str] = [
            "COMPREHENSIVE AERODYNAMIC ANALYSIS",
            "=" * 70,
            f"Snapshots         : {len(snapshots)}",
            (
                "Endpoint trim     : dropped final sample"
                if drop_final_sample and raw_sample_count != len(t)
                else "Endpoint trim     : none"
            ),
            f"Time span         : [{t.min():.4f}, {t.max():.4f}]",
            f"Cylinder center   : ({geom.center_x:.6g}, {geom.center_y:.6g})",
            f"Cylinder radius   : {geom.radius:.6g}",
            f"Force source      : {force_source}",
            *( [f"Kinematic visc.   : {nu:.6g}"] if nu is not None else [] ),
            *(
                [
                    "Pressure sample   : "
                    f"Cp=[{_LAST_SURFACE_DIAGNOSTICS.pressure_cp_min:.6g}, "
                    f"{_LAST_SURFACE_DIAGNOSTICS.pressure_cp_max:.6g}], "
                    f"pressure-only Cd={_LAST_SURFACE_DIAGNOSTICS.pressure_cd:.6g}",
                    "Surface contour   : "
                    f"h={_LAST_SURFACE_DIAGNOSTICS.sample_spacing:.6g}, "
                    f"offset={_LAST_SURFACE_DIAGNOSTICS.sample_offset:.6g}",
                    (
                        "WARNING          : surface pressure samples are not physical."
                        if max(
                            abs(_LAST_SURFACE_DIAGNOSTICS.pressure_cp_min),
                            abs(_LAST_SURFACE_DIAGNOSTICS.pressure_cp_max),
                        )
                        > 100.0
                        else "WARNING          : none"
                    ),
                ]
                if nu is not None and _LAST_SURFACE_DIAGNOSTICS is not None
                else []
            ),
            *(
                [
                    "Drag decomposition: "
                    f"C_d,p={drag_decomposition_stats['pressure_c_d_mean']:.6g}, "
                    f"C_d,v={drag_decomposition_stats['viscous_c_d_mean']:.6g}, "
                    f"C_d,total={drag_decomposition_stats['total_c_d_mean']:.6g}",
                    "Drag decomp. range: "
                    f"C_d,p=[{drag_decomposition_stats['pressure_c_d_min']:.6g}, "
                    f"{drag_decomposition_stats['pressure_c_d_max']:.6g}], "
                    f"C_d,v=[{drag_decomposition_stats['viscous_c_d_min']:.6g}, "
                    f"{drag_decomposition_stats['viscous_c_d_max']:.6g}]",
                ]
                if drag_decomposition_stats is not None
                else []
            ),
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
            lines.insert(
                4, f"Probe location    : ({probe_x:.6g}, {probe_y:.6g})")

        if np.isfinite(nyquist_est):
            lines.append(
                f"Median dt         : {dt_median:.6g} "
                f"(Nyquist approx {nyquist_est:.6g})"
            )

        if lift_strouhal is None:
            lines.append(
                "C_l spectral peak: unavailable (insufficient variation/samples)")
        else:
            edge_note = ""
            if _is_edge_frequency(lift_strouhal.freq, f_min, f_max):
                edge_note = " [edge]"
            lines.append(
                f"C_l spectral peak: f={lift_strouhal.freq:.6g}, "
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
            *_stats_lines(primary_stats),
        ])
        if settled_stats is not None:
            lines.extend([
                "",
                "-" * 70,
                "SETTLED COEFFICIENT STATISTICS",
                "-" * 70,
                *_stats_lines(settled_stats),
            ])
        elif settled_stats_start is not None:
            lines.append(
                f"Settled stats     : unavailable for t >= {settled_stats_start:.6g}"
            )

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
        save_drag_decomposition=args.save_drag_decomposition,
        save_report=args.save_report,
        force_source=args.force_source,
        drop_final_sample=not args.keep_final_sample,
        surface_sample_offset_factor=args.surface_sample_offset_factor,
        stats_t_min=args.stats_t_min,
    )


if __name__ == "__main__":
    raise SystemExit(main())
