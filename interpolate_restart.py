"""Interpolate a saved MAC-grid snapshot onto a restart grid.

The solver stores a staggered 2-D MAC layout:

    p : cell centers     (nx, ny)
    u : x-faces/y-centers (nx + 1, ny)
    v : x-centers/y-faces (nx, ny + 1)

This tool preserves that staggering while moving a coarse snapshot onto a
finer grid.  It writes a normal ``snap_*.npz``-style restart file that can be
passed to ``main.py --resume-from``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from src.grid import CartesianGrid
from src.io_utils import (
    load_grid_metadata_dict,
    load_prepared_grid,
    load_snapshot,
    save_grid_metadata,
    save_grid_metadata_dict,
    save_snapshot,
)


@dataclass(frozen=True)
class InterpolatedRestart:
    """Container returned by restart interpolation helpers."""

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    t: float
    meta: dict
    grid: CartesianGrid


def _grid_path_for_directory(directory: str) -> str | None:
    """Return a prepared-grid metadata path from a snapshot directory."""
    for name in ("nonuniform_grid.npz", "uniform_grid.npz"):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def infer_snapshot_grid(snapshot_path: str, meta: dict) -> CartesianGrid:
    """Infer the source grid for a snapshot.

    A prepared grid file next to the snapshot is preferred so nonuniform
    coordinates are preserved.  If none exists, uniform-grid metadata stored in
    the snapshot is used.
    """
    grid_path = _grid_path_for_directory(os.path.dirname(os.path.abspath(snapshot_path)))
    if grid_path is not None:
        return load_prepared_grid(grid_path)

    nx = int(meta["nx"])
    ny = int(meta["ny"])
    x_min = float(meta.get("x_min", 0.0))
    y_min = float(meta.get("y_min", 0.0))
    x_max = float(meta.get("x_max", x_min + float(meta["lx"])))
    y_max = float(meta.get("y_max", y_min + float(meta["ly"])))
    return CartesianGrid(
        nx=nx,
        ny=ny,
        lx=x_max - x_min,
        ly=y_max - y_min,
        x_min=x_min,
        y_min=y_min,
    )


def build_uniform_target_grid(
    nx: int,
    ny: int,
    source_grid: CartesianGrid,
) -> CartesianGrid:
    """Build a uniform target grid over the source physical domain."""
    return CartesianGrid(
        nx=int(nx),
        ny=int(ny),
        lx=source_grid.lx,
        ly=source_grid.ly,
        x_min=source_grid.x_min,
        y_min=source_grid.y_min,
    )


def _linear_interpolate(
    values: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (x_old, y_old),
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    x_mesh, y_mesh = np.meshgrid(x_new, y_new, indexing="ij")
    points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    return interpolator(points).reshape((x_new.size, y_new.size))


def _cubic_interpolate(
    values: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    if x_old.size < 4 or y_old.size < 4:
        raise ValueError(
            "cubic interpolation requires at least four points in each direction"
        )
    spline = RectBivariateSpline(x_old, y_old, values, kx=3, ky=3)
    return spline(x_new, y_new)


def interpolate_field(
    values: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpolate a 2-D field from old tensor-product points to new ones."""
    method = str(method).strip().lower()
    if method == "linear":
        return _linear_interpolate(values, x_old, y_old, x_new, y_new)
    if method in {"cubic", "cubic-spline", "spline"}:
        return _cubic_interpolate(values, x_old, y_old, x_new, y_new)
    raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_restart_fields(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    t: float,
    meta: dict,
    source_grid: CartesianGrid,
    target_grid: CartesianGrid,
    method: str = "linear",
) -> InterpolatedRestart:
    """Interpolate snapshot fields from ``source_grid`` to ``target_grid``."""
    expected_source_shapes = {
        "u": source_grid.u_shape,
        "v": source_grid.v_shape,
        "p": source_grid.p_shape,
    }
    found_source_shapes = {"u": u.shape, "v": v.shape, "p": p.shape}
    if found_source_shapes != expected_source_shapes:
        raise ValueError(
            "Snapshot shapes do not match the source grid: "
            f"expected={expected_source_shapes}, found={found_source_shapes}"
        )

    u_fine = interpolate_field(
        u,
        source_grid.xf,
        source_grid.yc,
        target_grid.xf,
        target_grid.yc,
        method,
    )
    v_fine = interpolate_field(
        v,
        source_grid.xc,
        source_grid.yf,
        target_grid.xc,
        target_grid.yf,
        method,
    )
    p_fine = interpolate_field(
        p,
        source_grid.xc,
        source_grid.yc,
        target_grid.xc,
        target_grid.yc,
        method,
    )

    restart_meta = dict(meta)
    restart_meta.update(
        {
            "nx": target_grid.nx,
            "ny": target_grid.ny,
            "nz": target_grid.nz,
            "lx": target_grid.lx,
            "ly": target_grid.ly,
            "lz": target_grid.lz,
            "x_min": target_grid.x_min,
            "x_max": target_grid.x_max,
            "y_min": target_grid.y_min,
            "y_max": target_grid.y_max,
            "restart_interpolation_method": str(method).lower(),
            "restart_source_nx": source_grid.nx,
            "restart_source_ny": source_grid.ny,
            "restart_source_grid_type": (
                "uniform" if source_grid.is_uniform else "nonuniform"
            ),
            "restart_target_grid_type": (
                "uniform" if target_grid.is_uniform else "nonuniform"
            ),
        }
    )
    return InterpolatedRestart(
        u=u_fine,
        v=v_fine,
        p=p_fine,
        t=float(t),
        meta=restart_meta,
        grid=target_grid,
    )


def interpolate_restart_file(
    input_snapshot: str,
    output_snapshot: str,
    target_grid: CartesianGrid,
    method: str = "linear",
    source_grid: CartesianGrid | None = None,
    write_grid: bool = False,
) -> InterpolatedRestart:
    """Load, interpolate, and save a restart snapshot."""
    u, v, p, t, meta = load_snapshot(input_snapshot, fmt="auto")
    if source_grid is None:
        source_grid = infer_snapshot_grid(input_snapshot, meta)
    result = interpolate_restart_fields(
        u,
        v,
        p,
        t,
        meta,
        source_grid=source_grid,
        target_grid=target_grid,
        method=method,
    )
    save_snapshot(
        output_snapshot,
        result.u,
        result.v,
        result.p,
        result.t,
        meta=result.meta,
        fmt="auto",
    )
    if write_grid:
        write_target_grid_metadata(
            os.path.dirname(os.path.abspath(output_snapshot)),
            result.grid,
        )
    return result


def write_target_grid_metadata(outdir: str, grid: CartesianGrid) -> str:
    """Write prepared-grid metadata for the target restart directory."""
    os.makedirs(outdir, exist_ok=True)
    if grid.is_uniform:
        path = os.path.join(outdir, "uniform_grid.npz")
        save_grid_metadata(path, grid)
    else:
        path = os.path.join(outdir, "nonuniform_grid.npz")
        save_grid_metadata_dict(path, grid.to_metadata())
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate a coarse snapshot onto a finer restart grid."
    )
    parser.add_argument("input_snapshot", help="Source snap_*.npz or .h5 snapshot")
    parser.add_argument("output_snapshot", help="Interpolated restart snapshot path")
    parser.add_argument(
        "--method",
        choices=["linear", "cubic"],
        default="linear",
        help="Interpolation method",
    )
    parser.add_argument("--nx", type=int, help="Target number of x cells")
    parser.add_argument("--ny", type=int, help="Target number of y cells")
    parser.add_argument(
        "--source-grid",
        help="Optional source prepared-grid metadata file",
    )
    parser.add_argument(
        "--target-grid",
        help="Optional target prepared-grid metadata file. Overrides --nx/--ny.",
    )
    parser.add_argument(
        "--write-grid",
        action="store_true",
        help="Write target uniform_grid.npz or nonuniform_grid.npz next to output",
    )
    args = parser.parse_args(argv)
    if args.target_grid is None and (args.nx is None or args.ny is None):
        parser.error("Provide either --target-grid or both --nx and --ny")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, _, _, _, meta = load_snapshot(args.input_snapshot, fmt="auto")
    source_grid = (
        load_prepared_grid(args.source_grid)
        if args.source_grid
        else infer_snapshot_grid(args.input_snapshot, meta)
    )
    if args.target_grid:
        target_grid = load_prepared_grid(args.target_grid)
        if args.write_grid:
            target_meta = load_grid_metadata_dict(args.target_grid)
            outdir = os.path.dirname(os.path.abspath(args.output_snapshot))
            name = "uniform_grid.npz" if target_grid.is_uniform else "nonuniform_grid.npz"
            save_grid_metadata_dict(os.path.join(outdir, name), target_meta)
    else:
        target_grid = build_uniform_target_grid(args.nx, args.ny, source_grid)

    result = interpolate_restart_file(
        args.input_snapshot,
        args.output_snapshot,
        target_grid=target_grid,
        method=args.method,
        source_grid=source_grid,
        write_grid=(args.write_grid and args.target_grid is None),
    )
    print(
        "Wrote restart snapshot "
        f"{args.output_snapshot} at t={result.t:.6g} "
        f"with shapes u={result.u.shape}, v={result.v.shape}, p={result.p.shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
