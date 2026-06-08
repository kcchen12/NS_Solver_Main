"""Pre-generate and save prepared grid metadata (uniform or non-uniform)."""

from __future__ import annotations

import argparse
import os

from src.config import ConfigParser
from src.grid import CartesianGrid, build_nonuniform_grid_metadata
from src.io_utils import save_grid_metadata, save_grid_metadata_dict


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config.txt")
    default_outdir = os.path.join(script_dir, "output")

    p_pre = argparse.ArgumentParser(add_help=False)
    p_pre.add_argument("--config", type=str, default=default_config,
                       help="Path to configuration file")
    args_pre, remaining = p_pre.parse_known_args()

    cfg = ConfigParser(args_pre.config)

    # Unified grid controls from config.txt.
    uniform_grid = cfg.get("uniform_grid", None, bool)
    if uniform_grid is None:
        grid_type_default = cfg.get("grid_type", "uniform", str)
    else:
        grid_type_default = "uniform" if uniform_grid else "nonuniform"

    beta_x_default = cfg.get("grid_beta_x", 2.0, float)
    beta_y_default = cfg.get("grid_beta_y", 2.0, float)
    uniform_x_start_default = cfg.get("grid_uniform_x_start", None, float)
    uniform_x_end_default = cfg.get("grid_uniform_x_end", None, float)
    uniform_y_start_default = cfg.get("grid_uniform_y_start", None, float)
    uniform_y_end_default = cfg.get("grid_uniform_y_end", None, float)

    p = argparse.ArgumentParser(
        description="Pre-generate prepared grid metadata for post-processing and setup",
    )
    p.add_argument("--config", type=str, default=default_config,
                   help="Path to configuration file")
    p.add_argument("--nx", type=int, default=cfg.get("nx", 64, int))
    p.add_argument("--ny", type=int, default=cfg.get("ny", 32, int))
    p.add_argument("--lx", type=float, default=cfg.get("lx", 4.0, float))
    p.add_argument("--ly", type=float, default=cfg.get("ly", 2.0, float))
    p.add_argument("--x-min", type=float, default=cfg.get("x_min", None, float),
                   help="Domain lower bound in x (optional; defaults to 0)")
    p.add_argument("--x-max", type=float, default=cfg.get("x_max", None, float),
                   help="Domain upper bound in x (optional; inferred from x_min+lx)")
    p.add_argument("--y-min", type=float, default=cfg.get("y_min", None, float),
                   help="Domain lower bound in y (optional; defaults to 0)")
    p.add_argument("--y-max", type=float, default=cfg.get("y_max", None, float),
                   help="Domain upper bound in y (optional; inferred from y_min+ly)")
    p.add_argument("--outdir", type=str,
                   default=cfg.get("outdir", default_outdir, str))

    p.add_argument(
        "--grid-type",
        type=str,
        choices=["uniform", "nonuniform"],
        default=grid_type_default,
        help="Prepared-grid type to generate",
    )
    p.add_argument(
        "--beta-x",
        type=float,
        default=beta_x_default,
        help="x-direction center-density boost for nonuniform grid (<=0 gives uniform spacing)",
    )
    p.add_argument(
        "--beta-y",
        type=float,
        default=beta_y_default,
        help="y-direction center-density boost for nonuniform grid (<=0 gives uniform spacing)",
    )
    p.add_argument(
        "--uniform-x-start",
        type=float,
        default=uniform_x_start_default,
        help="Absolute x-start of the uniform core for nonuniform grids",
    )
    p.add_argument(
        "--uniform-x-end",
        type=float,
        default=uniform_x_end_default,
        help="Absolute x-end of the uniform core for nonuniform grids",
    )
    p.add_argument(
        "--uniform-y-start",
        type=float,
        default=uniform_y_start_default,
        help="Absolute y-start of the uniform core for nonuniform grids (use -a)",
    )
    p.add_argument(
        "--uniform-y-end",
        type=float,
        default=uniform_y_end_default,
        help="Absolute y-end of the uniform core for nonuniform grids (use +a)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename override (default depends on grid type)",
    )

    args = p.parse_args(remaining)
    args.x_min = 0.0 if args.x_min is None else float(args.x_min)
    args.y_min = 0.0 if args.y_min is None else float(args.y_min)

    if args.x_max is None:
        args.x_max = args.x_min + float(args.lx)
    else:
        args.x_max = float(args.x_max)
    if args.y_max is None:
        args.y_max = args.y_min + float(args.ly)
    else:
        args.y_max = float(args.y_max)

    if not args.x_max > args.x_min:
        p.error("Require x_max > x_min")
    if not args.y_max > args.y_min:
        p.error("Require y_max > y_min")
    if (args.uniform_x_start is None) != (args.uniform_x_end is None):
        p.error("Provide both --uniform-x-start and --uniform-x-end, or neither")
    if (args.uniform_y_start is None) != (args.uniform_y_end is None):
        p.error("Provide both --uniform-y-start and --uniform-y-end, or neither")
    if args.grid_type == "nonuniform":
        if args.uniform_x_start is None or args.uniform_y_start is None:
            p.error(
                "nonuniform grids require explicit --uniform-x-* and --uniform-y-* bounds")

    args.lx = float(args.x_max - args.x_min)
    args.ly = float(args.y_max - args.y_min)
    args.nonuniform_mode = "center-uniform"
    return args


def _default_output_name(grid_type: str) -> str:
    return "uniform_grid.npz" if grid_type == "uniform" else "nonuniform_grid.npz"


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    output_name = args.output_name or _default_output_name(args.grid_type)
    output_path = os.path.join(args.outdir, output_name)
    display_path = output_path.replace("\\", "/")

    if args.grid_type == "uniform":
        grid = CartesianGrid(nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
                             x_min=args.x_min, y_min=args.y_min)
        save_grid_metadata(output_path, grid)
        print(
            "Saved uniform prepared grid metadata to "
            f"{display_path} for nx={grid.nx}, ny={grid.ny}, lx={grid.lx}, ly={grid.ly}"
        )
    else:
        metadata = build_nonuniform_grid_metadata(
            nx=args.nx,
            ny=args.ny,
            lx=args.lx,
            ly=args.ly,
            beta_x=args.beta_x,
            beta_y=args.beta_y,
            x_min=args.x_min,
            y_min=args.y_min,
            uniform_x_start=args.uniform_x_start,
            uniform_x_end=args.uniform_x_end,
            uniform_y_start=args.uniform_y_start,
            uniform_y_end=args.uniform_y_end,
        )
        save_grid_metadata_dict(output_path, metadata)
        print(
            "Saved nonuniform prepared grid metadata to "
            f"{display_path} for nx={args.nx}, ny={args.ny}, lx={args.lx}, ly={args.ly}, "
            f"beta_x={args.beta_x}, beta_y={args.beta_y}, "
            "mode=center-uniform, "
            f"uniform_x=[{args.uniform_x_start}, {args.uniform_x_end}], "
            f"uniform_y=[{args.uniform_y_start}, {args.uniform_y_end}]"
        )
