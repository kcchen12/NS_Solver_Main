"""
main.py — 2-D incompressible Navier-Stokes solver driver.

Simulates **uniform flow in a 2-D box** using:
    - MAC staggered Cartesian grid
    - Finite-volume spatial discretisation
    - SSP-RK3 time integration
    - Fractional-step (projection) pressure-velocity coupling
    - Convective outflow at the right boundary
    - No-slip walls on top and bottom
    - Inflow at the left boundary
    - (Optional) Immersed-boundary cylinder demo

Usage
-----
Serial run (reads from config.txt, experimental_config.txt, and post_config.txt)::

    python main.py

Parallel run (4 MPI processes)::

    mpirun -n 4 python main.py

Command-line arguments override config file values::

    --config    Path to config file              [default: config.txt]
    --nx        Number of cells in x             [default: from config]
    --ny        Number of cells in y             [default: from config]
    --lx        Domain length in x               [default: from config]
    --ly        Domain length in y               [default: from config]
    --re        Reynolds number                  [default: from config]
    --t_end     End time                         [default: from config]
    --cfl       Target CFL number                [default: from config]
    --save_dt   Interval between snapshots       [default: from config]
    --outdir    Output directory                 [default: from config]
    --cylinder  Add an immersed-boundary cylinder [flag]
    --experiment-config  Path to experimental cylinder config [default: experimental_config.txt]
    --plot      Save matplotlib plots at the end  [default: from post config]
"""

import argparse
import os
import numpy as np

from src.grid import CartesianGrid, build_nonuniform_grid_metadata
from src.boundary import BoundaryConfig, BCType, FarfieldMode
from src.solver import FractionalStepSolver
from src.ibm import ImmersedBoundary
from src.io_utils import (
    save_snapshot,
    load_snapshot,
    save_grid_metadata,
    save_grid_metadata_dict,
    load_prepared_grid,
    load_grid_metadata_dict,
)
from src.parallel import ParallelDecomposition
from src.config import ConfigParser
from analyze_aerodynamics import (
    run_analysis as run_aero_analysis,
    plot_drag_decomposition,
    plot_shedding_spectrum,
    save_pressure_coefficient_report,
)
from time_average_snapshots import plot_time_averaged_fields, save_time_averaged_fields
from view_snapshot_viewer import find_latest_snapshot, plot_coeff_history
from view_snapshot_viewer import plot_ibm_forcing, plot_vorticity_video


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _normalize_bc_type(raw_value: str, default: str) -> str:
    value = str(raw_value).strip().lower()
    valid = {
        BCType.INFLOW,
        BCType.FARFIELD,
        BCType.OUTFLOW,
        BCType.WALL,
        BCType.PERIODIC,
    }
    return value if value in valid else default


def _normalize_farfield_mode(raw_value: str | None) -> str:
    value = FarfieldMode.DIRICHLET if raw_value is None else str(raw_value).strip().lower()
    return value if value in {FarfieldMode.DIRICHLET, FarfieldMode.NEUMANN} else FarfieldMode.DIRICHLET


def _normalize_cylinder_rotation_mode(raw_value: str | None) -> str:
    value = "stationary" if raw_value is None else str(
        raw_value).strip().lower()
    aliases = {
        "fixed_cylinder": "stationary",
        "fixed-cylinder": "stationary",
    }
    value = aliases.get(value, value)
    return value if value in {"stationary", "oscillatory", "constant"} else "stationary"


def _normalize_cylinder_translation_mode(raw_value: str | None) -> str:
    value = "stationary" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "none": "stationary",
        "off": "stationary",
        "fixed": "stationary",
        "left-right": "oscillatory-x",
        "left_right": "oscillatory-x",
        "horizontal": "oscillatory-x",
        "up-down": "oscillatory-y",
        "up_down": "oscillatory-y",
        "vertical": "oscillatory-y",
        "xy": "oscillatory-xy",
        "both": "oscillatory-xy",
        "oscillatory": "oscillatory-xy",
    }
    value = aliases.get(value, value)
    valid = {"stationary", "oscillatory-x", "oscillatory-y", "oscillatory-xy"}
    return value if value in valid else "stationary"


def _normalize_ibm_shape(raw_value: str | None) -> str:
    value = "circle" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "cylinder": "circle",
        "square-cylinder": "square",
        "square_cylinder": "square",
        "square-bluff-body": "square",
        "square_bluff_body": "square",
        "airfoil": "airfoil",
        "naca": "airfoil",
        "naca0012": "airfoil",
        "naca-0012": "airfoil",
        "circle_top_indent": "circle-with-top-indent",
        "circle-with-indent": "circle-with-top-indent",
        "indented-circle": "circle-with-top-indent",
    }
    value = aliases.get(value, value)
    return value if value in {"circle", "circle-with-top-indent", "square", "airfoil"} else "circle"


def _normalize_cylinder_geometry_mode(raw_value: str | None) -> str:
    value = "circle" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "ibm-circle": "circle",
        "square-cylinder": "square",
        "square_cylinder": "square",
        "square-bluff-body": "square",
        "square_bluff_body": "square",
        "naca": "airfoil",
        "naca0012": "airfoil",
        "naca-0012": "airfoil",
        "indented-circle": "circle-with-top-indent",
        "rectangular-top-indent": "circle-with-top-indent",
    }
    return _normalize_ibm_shape(aliases.get(value, value))


def _normalize_cylinder_experiment_mode(raw_value: str | None) -> str:
    value = "circle" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "none": "circle",
        "off": "circle",
        "cylinder": "circle",
        "indent": "top-indent",
        "rectangular-indent": "top-indent",
        "square-cylinder": "square",
        "square_cylinder": "square",
        "square-bluff-body": "square",
        "square_bluff_body": "square",
        "naca": "airfoil",
        "naca0012": "airfoil",
        "naca-0012": "airfoil",
    }
    value = aliases.get(value, value)
    return value if value in {"circle", "top-indent", "square", "airfoil"} else "circle"


def _experimental_default(exp_cfg: ConfigParser, enabled: bool, key: str, default, dtype):
    if not enabled:
        return default
    return exp_cfg.get(key, default, dtype)


def _truncate_grid_metadata_y(metadata: dict, trim_cells: int) -> dict:
    """Return grid metadata with bottom/top y cells removed."""
    trim_cells = int(trim_cells)
    if trim_cells <= 0:
        return metadata

    yf = np.asarray(metadata["yf"], dtype=float)
    ny = int(metadata["ny"])
    if ny <= 2 * trim_cells:
        raise ValueError(
            f"Cannot truncate {trim_cells} y cells from both ends of ny={ny}"
        )

    trimmed_yf = yf[trim_cells:-trim_cells]
    grid = CartesianGrid(
        nx=int(metadata["nx"]),
        ny=int(trimmed_yf.size - 1),
        nz=int(metadata.get("nz", 1)),
        lx=float(metadata["lx"]),
        ly=float(trimmed_yf[-1] - trimmed_yf[0]),
        lz=float(metadata.get("lz", 1.0)),
        x_min=float(metadata.get("x_min", 0.0)),
        y_min=float(trimmed_yf[0]),
        z_min=float(metadata.get("z_min", 0.0)),
        xf=np.asarray(metadata["xf"], dtype=float),
        yf=trimmed_yf,
        zf=np.asarray(metadata["zf"], dtype=float) if "zf" in metadata else None,
    )
    truncated = grid.to_metadata()
    for key in (
        "grid_type",
        "beta_x",
        "beta_y",
        "nonuniform_mode",
        "uniform_x_start",
        "uniform_x_end",
        "uniform_y_start",
        "uniform_y_end",
        "band_start_x",
        "band_end_x",
        "band_start_y",
        "band_end_y",
    ):
        if key in metadata:
            truncated[key] = metadata[key]
    truncated["y_truncation_cells_each_end"] = trim_cells
    truncated["source_ny_before_y_truncation"] = ny
    truncated["source_y_min_before_y_truncation"] = float(metadata["y_min"])
    truncated["source_y_max_before_y_truncation"] = float(metadata["y_max"])
    truncated["dx"] = grid.dx_cells.copy()
    truncated["dy"] = grid.dy_cells.copy()
    return truncated


def _apply_y_truncation_to_args(args, parser) -> None:
    trim_cells = int(getattr(args, "truncate_y_cells", 0))
    args.y_truncation_enabled = False
    args.y_truncation_cells_each_end = 0
    args.source_ny_before_y_truncation = args.ny
    args.source_y_min_before_y_truncation = args.y_min
    args.source_y_max_before_y_truncation = args.y_max
    args.source_ly_before_y_truncation = args.ly

    if trim_cells < 0:
        parser.error("--truncate-y-cells must be non-negative")
    if trim_cells == 0:
        return
    if args.ny <= 2 * trim_cells:
        parser.error(
            f"--truncate-y-cells={trim_cells} requires ny > {2 * trim_cells}"
        )

    args.y_truncation_enabled = True
    args.y_truncation_cells_each_end = trim_cells
    args.source_ny_before_y_truncation = args.ny
    args.source_y_min_before_y_truncation = args.y_min
    args.source_y_max_before_y_truncation = args.y_max
    args.source_ly_before_y_truncation = args.ly

    if args.grid_type == "nonuniform":
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
        truncated = _truncate_grid_metadata_y(metadata, trim_cells)
        args._truncated_yf = np.asarray(truncated["yf"], dtype=float)
    else:
        yf = np.linspace(args.y_min, args.y_max, args.ny + 1)
        args._truncated_yf = yf[trim_cells:-trim_cells]

    args.ny = int(args._truncated_yf.size - 1)
    args.y_min = float(args._truncated_yf[0])
    args.y_max = float(args._truncated_yf[-1])
    args.ly = float(args.y_max - args.y_min)


def parse_args():
    """
    Parse command-line arguments and merge with config file.

    Command-line arguments take precedence over config file values.
    All paths default to the script directory to ensure consistent output location.
    """
    def str_to_bool(v):
        """Convert string to boolean."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Get the directory where this script is located
    default_config = os.path.join(SCRIPT_DIR, "config.txt")
    default_outdir = os.path.join(SCRIPT_DIR, "output")

    # First parse just the config file paths
    p_pre = argparse.ArgumentParser(add_help=False)
    p_pre.add_argument("--config", type=str, default=default_config,
                       help="Path to configuration file")
    p_pre.add_argument("--experiment-config", type=str,
                       default=os.path.join(SCRIPT_DIR, "experimental_config.txt"),
                       help="Path to experimental cylinder configuration file")
    p_pre.add_argument("--post-config", type=str,
                       default=os.path.join(SCRIPT_DIR, "post_config.txt"),
                       help="Path to post-processing configuration file")
    args_pre, remaining = p_pre.parse_known_args()

    # Read config files
    cfg = ConfigParser(args_pre.config)
    exp_cfg = ConfigParser(args_pre.experiment_config)
    post_cfg = ConfigParser(args_pre.post_config)
    experimental_config_enabled = exp_cfg.get(
        "enable_experimental_config",
        False,
        bool,
    )

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

    # Now parse all arguments with defaults from config file
    p = argparse.ArgumentParser(description="2-D Navier-Stokes solver")
    p.add_argument("--config", type=str, default=default_config,
                   help="Path to configuration file")
    p.add_argument("--experiment-config", type=str, default=args_pre.experiment_config,
                   help="Path to experimental cylinder configuration file")
    p.add_argument("--post-config", type=str, default=args_pre.post_config,
                   help="Path to post-processing configuration file")
    p.add_argument("--nx",       type=int,   default=cfg.get("nx", 64, int))
    p.add_argument("--ny",       type=int,   default=cfg.get("ny", 32, int))
    p.add_argument("--lx",       type=float, default=cfg.get("lx", 4.0, float))
    p.add_argument("--ly",       type=float, default=cfg.get("ly", 2.0, float))
    p.add_argument("--x-min",    type=float, default=cfg.get("x_min", None, float),
                   help="Domain lower bound in x (optional; defaults to 0)")
    p.add_argument("--x-max",    type=float, default=cfg.get("x_max", None, float),
                   help="Domain upper bound in x (optional; inferred from x_min+lx)")
    p.add_argument("--y-min",    type=float, default=cfg.get("y_min", None, float),
                   help="Domain lower bound in y (optional; defaults to 0)")
    p.add_argument("--y-max",    type=float, default=cfg.get("y_max", None, float),
                   help="Domain upper bound in y (optional; inferred from y_min+ly)")
    p.add_argument("--re",       type=float, default=cfg.get("re", 100.0, float),
                   help="Reynolds number (Re = U_inf * L / nu)")
    p.add_argument("--t_end",    type=float,
                   default=cfg.get("t_end", 5.0, float))
    p.add_argument("--cfl",      type=float,
                   default=cfg.get("cfl", 0.4, float))
    p.add_argument("--save_dt",  type=float,
                   default=cfg.get("save_dt", 0.5, float))
    p.add_argument("--outdir",   type=str,
                   default=cfg.get("outdir", default_outdir, str))
    p.add_argument("--resume-from", type=str, default=cfg.get("resume_from", "", str),
                   help="Resume from a saved snapshot path instead of starting from t=0")
    p.add_argument("--resume-latest", type=str_to_bool,
                   default=cfg.get("resume_latest", False, bool),
                   help="Resume from the latest snap_*.npz in --outdir")
    p.add_argument("--grid-type", type=str,
                   choices=["uniform", "nonuniform"],
                   default=grid_type_default,
                   help="Runtime grid type")
    p.add_argument("--beta-x", type=float,
                   default=beta_x_default,
                   help="x-direction center-density boost for nonuniform grid")
    p.add_argument("--beta-y", type=float,
                   default=beta_y_default,
                   help="y-direction center-density boost for nonuniform grid")
    p.add_argument("--uniform-x-start", type=float,
                   default=uniform_x_start_default,
                   help="Absolute x-start of the uniform core for nonuniform grids")
    p.add_argument("--uniform-x-end", type=float,
                   default=uniform_x_end_default,
                   help="Absolute x-end of the uniform core for nonuniform grids")
    p.add_argument("--uniform-y-start", type=float,
                   default=uniform_y_start_default,
                   help="Absolute y-start of the uniform core for nonuniform grids (use -a)")
    p.add_argument("--uniform-y-end", type=float,
                   default=uniform_y_end_default,
                   help="Absolute y-end of the uniform core for nonuniform grids (use +a)")
    p.add_argument("--cylinder", type=str_to_bool, default=cfg.get("cylinder", False, bool),
                   help="Add an immersed-boundary cylinder at the domain centre")
    p.add_argument("--cylinder-radius", type=float,
                   default=cfg.get("cylinder_radius", -1.0, float),
                   help="Cylinder radius in physical units (<=0 uses default ly/8)")
    p.add_argument("--cylinder-center-x", type=float,
                   default=cfg.get("cylinder_center_x", -1.0, float),
                   help="Cylinder center x-coordinate in physical units (<0 uses default lx/4)")
    p.add_argument("--cylinder-center-y", type=float,
                   default=cfg.get("cylinder_center_y", -1.0, float),
                   help="Cylinder center y-coordinate in physical units (<0 uses default ly/2)")
    p.add_argument("--cylinder-free-y-dof", type=str_to_bool,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_dof",
                       False,
                       bool,
                   ),
                   help=(
                       "Experimental option: request one-degree-of-freedom "
                       "transverse cylinder motion for VIV studies"
                   ))
    p.add_argument("--cylinder-free-x-dof", type=str_to_bool,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_dof",
                       False,
                       bool,
                   ),
                   help=(
                       "Experimental option: request free streamwise cylinder "
                       "motion with the spring-mass-damper model"
                   ))
    p.add_argument("--cylinder-free-x-mass", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_mass",
                       100.0,
                       float,
                   ),
                   help="Mass for the experimental free-x cylinder oscillator")
    p.add_argument("--cylinder-free-x-damping", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_damping",
                       5.0,
                       float,
                   ),
                   help="Damping coefficient for the experimental free-x cylinder oscillator")
    p.add_argument("--cylinder-free-x-stiffness", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_stiffness",
                       20.0,
                       float,
                   ),
                   help="Spring stiffness for the experimental free-x cylinder oscillator")
    p.add_argument("--cylinder-free-x-initial-velocity", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_initial_velocity",
                       0.0,
                       float,
                   ),
                   help="Initial streamwise velocity for the experimental free-x cylinder oscillator")
    p.add_argument("--cylinder-free-x-force-relaxation", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_force_relaxation",
                       0.05,
                       float,
                   ),
                   help="Exponential relaxation factor applied to the IBM drag force")
    p.add_argument("--cylinder-free-x-max-displacement-percent", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_max_displacement_percent",
                       25.0,
                       float,
                   ),
                   help="Maximum free-x displacement as percent of cylinder diameter")
    p.add_argument("--cylinder-free-x-max-speed", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_x_max_speed",
                       0.25,
                       float,
                   ),
                   help="Maximum absolute streamwise speed for the free-x cylinder")
    p.add_argument("--cylinder-free-y-mass", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_mass",
                       100.0,
                       float,
                   ),
                   help="Mass for the experimental free-y cylinder oscillator")
    p.add_argument("--cylinder-free-y-damping", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_damping",
                       5.0,
                       float,
                   ),
                   help="Damping coefficient for the experimental free-y cylinder oscillator")
    p.add_argument("--cylinder-free-y-stiffness", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_stiffness",
                       20.0,
                       float,
                   ),
                   help="Spring stiffness for the experimental free-y cylinder oscillator")
    p.add_argument("--cylinder-free-y-initial-velocity", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_initial_velocity",
                       0.0,
                       float,
                   ),
                   help="Initial transverse velocity for the experimental free-y cylinder oscillator")
    p.add_argument("--cylinder-free-y-force-relaxation", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_force_relaxation",
                       0.05,
                       float,
                   ),
                   help="Exponential relaxation factor applied to the IBM lift force")
    p.add_argument("--cylinder-free-y-max-displacement-percent", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_max_displacement_percent",
                       25.0,
                       float,
                   ),
                   help="Maximum free-y displacement as percent of cylinder diameter")
    p.add_argument("--cylinder-free-y-max-speed", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_free_y_max_speed",
                       0.25,
                       float,
                   ),
                   help="Maximum absolute transverse speed for the free-y cylinder")
    p.add_argument("--cylinder-experiment", type=str,
                   choices=[
                       "circle",
                       "top-indent",
                       "square",
                       "airfoil",
                   ],
                   default=_normalize_cylinder_experiment_mode(
                       _experimental_default(
                           exp_cfg,
                           experimental_config_enabled,
                           "cylinder_experiment",
                           "circle",
                           str,
                       )),
                   help="High-level experimental cylinder mode")
    p.add_argument("--cylinder-geometry-mode", type=str,
                   choices=["circle", "circle-with-top-indent", "square", "airfoil"],
                   default=_normalize_cylinder_geometry_mode(
                       (
                           _experimental_default(
                               exp_cfg,
                               experimental_config_enabled,
                               "cylinder_geometry_mode",
                               None,
                               str,
                           )
                           or _experimental_default(
                               exp_cfg,
                               experimental_config_enabled,
                               "ibm_shape",
                               "circle",
                               str,
                           )
                       )),
                   help="Cylinder geometry mode")
    p.add_argument("--ibm-shape", type=str,
                   choices=["circle", "circle-with-top-indent", "square", "airfoil"],
                   default=_normalize_cylinder_geometry_mode(
                       (
                           _experimental_default(
                               exp_cfg,
                               experimental_config_enabled,
                               "cylinder_geometry_mode",
                               None,
                               str,
                           )
                           or _experimental_default(
                               exp_cfg,
                               experimental_config_enabled,
                               "ibm_shape",
                               "circle",
                               str,
                           )
                       )),
                   help="Immersed-body shape")
    p.add_argument("--cylinder-indent-width", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_indent_width",
                       0.0,
                       float,
                   ),
                   help="Width of the rectangular top indent for circle-with-top-indent")
    p.add_argument("--cylinder-indent-depth", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "cylinder_indent_depth",
                       0.0,
                       float,
                   ),
                   help="Depth of the rectangular top indent for circle-with-top-indent")
    p.add_argument("--airfoil-chord", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "airfoil_chord",
                       -1.0,
                       float,
                   ),
                   help="NACA 00xx airfoil chord length (<=0 uses cylinder diameter)")
    p.add_argument("--airfoil-thickness-percent", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "airfoil_thickness_percent",
                       12.0,
                       float,
                   ),
                   help="NACA 00xx maximum thickness as percent of chord")
    p.add_argument("--airfoil-angle-deg", type=float,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "airfoil_angle_deg",
                       0.0,
                       float,
                   ),
                   help="Airfoil angle of attack in degrees")
    p.add_argument("--surface-sample-offset-factor", type=float,
                   default=post_cfg.get(
                       "surface_force_sample_offset_factor", 0.5, float),
                   help=(
                       "Surface-force contour offset in local grid spacings "
                       "for pressure/viscous post-processing"
                   ))
    p.add_argument("--truncate-y-cells", type=int,
                   default=_experimental_default(
                       exp_cfg,
                       experimental_config_enabled,
                       "truncate_y_cells",
                       0,
                       int,
                   ),
                   help=(
                       "Experimental option: drop this many cells from both "
                       "the bottom and top of the y-domain before running"
                   ))
    p.add_argument("--re-is-cylinder-based", type=str_to_bool,
                   default=cfg.get("re_is_cylinder_based", True, bool),
                   help="Interpret --re as Re_D based on cylinder diameter when cylinder is enabled")
    p.add_argument("--cylinder-rotation-mode", type=str,
                   choices=["stationary", "oscillatory", "constant"],
                   default=_normalize_cylinder_rotation_mode(
                       cfg.get("cylinder_rotation_mode", "stationary", str)),
                   help="Cylinder wall-motion mode")
    p.add_argument("--cylinder-rotation-amplitude", type=float,
                   default=cfg.get("cylinder_rotation_amplitude", 0.0, float),
                   help="Angular velocity for constant rotation, or amplitude for oscillatory rotation")
    p.add_argument("--cylinder-rotation-frequency", type=float,
                   default=cfg.get("cylinder_rotation_frequency", 0.0, float),
                   help="Oscillation frequency for cylinder rotation")
    p.add_argument("--cylinder-rotation-phase-deg", type=float,
                   default=cfg.get("cylinder_rotation_phase_deg", 0.0, float),
                   help="Phase offset in degrees for oscillatory cylinder rotation")
    translation_amplitude_percent_default = cfg.get(
        "cylinder_translation_amplitude_percent", 10.0, float
    )
    p.add_argument("--cylinder-translation-mode", type=str,
                   choices=[
                       "stationary",
                       "oscillatory-x",
                       "oscillatory-y",
                       "oscillatory-xy",
                   ],
                   default=_normalize_cylinder_translation_mode(
                       cfg.get("cylinder_translation_mode", "stationary", str)),
                   help="Non-rotating cylinder translation mode")
    p.add_argument("--cylinder-translation-amplitude-percent", type=float,
                   default=translation_amplitude_percent_default,
                   help="Default translation amplitude as percent of cylinder diameter")
    p.add_argument("--cylinder-translation-x-percent", type=float,
                   default=cfg.get(
                       "cylinder_translation_x_percent",
                       translation_amplitude_percent_default,
                       float,
                   ),
                   help="Left/right translation amplitude as percent of cylinder diameter")
    p.add_argument("--cylinder-translation-y-percent", type=float,
                   default=cfg.get(
                       "cylinder_translation_y_percent",
                       translation_amplitude_percent_default,
                       float,
                   ),
                   help="Up/down translation amplitude as percent of cylinder diameter")
    p.add_argument("--cylinder-translation-frequency", type=float,
                   default=cfg.get("cylinder_translation_frequency", 0.0, float),
                   help="Oscillation frequency for cylinder translation")
    p.add_argument("--cylinder-translation-phase-deg", type=float,
                   default=cfg.get("cylinder_translation_phase_deg", 0.0, float),
                   help="Phase offset in degrees for oscillatory cylinder translation")
    p.add_argument("--plot",     type=str_to_bool, default=post_cfg.get("plot", False, bool),
                   help="Save the standard end-of-run result figure")
    p.add_argument("--auto-generate-grid-spacing", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_grid_spacing", False, bool),
                   help="Automatically save the grid spacing/concentration figure after the run")
    p.add_argument("--auto-generate-coeff-history", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_coeff_history", False, bool),
                   help="Automatically save the drag/lift coefficient history figure after the run")
    p.add_argument("--auto-generate-aero-report", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_aero_report", False, bool),
                   help="Automatically save the aerodynamic report after the run")
    p.add_argument("--auto-generate-shedding-spectrum", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_shedding_spectrum", False, bool),
                   help="Automatically save the Fourier energy spectrum of C_l")
    p.add_argument("--auto-generate-drag-decomposition", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_drag_decomposition", False, bool),
                   help=(
                       "Automatically save pressure and viscous drag/lift "
                       "component histories"
                   ))
    p.add_argument("--auto-generate-drag-decomposition-plot", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_drag_decomposition_plot", False, bool),
                   help=(
                       "Automatically save a pressure/viscous force "
                       "decomposition figure"
                   ))
    p.add_argument("--auto-generate-pressure-coefficient-theta", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_pressure_coefficient_theta", False, bool),
                   help="Automatically save C_p(theta) CSV and PNG from the latest snapshot")
    p.add_argument("--auto-generate-time-averaged-fields", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_time_averaged_fields", False, bool),
                   help="Automatically save time-averaged mean/RMS fields from output snapshots")
    p.add_argument("--auto-generate-time-averaged-plots", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_time_averaged_plots", False, bool),
                   help="Automatically save a readable PNG summary of the time-averaged fields")
    p.add_argument("--auto-generate-ibm-forcing", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_ibm_forcing", False, bool),
                   help="Automatically save a plot of IBM forcing components/magnitude")
    p.add_argument("--auto-generate-vorticity-video", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_vorticity_video", False, bool),
                   help="Automatically save an animated vorticity GIF from output snapshots")
    p.add_argument("--auto-vorticity-video-frame-stride", type=int,
                   default=post_cfg.get(
                       "auto_vorticity_video_frame_stride", 1, int),
                   help="Use every nth snapshot when auto-generating the vorticity GIF")
    p.add_argument("--draw-cylinder-overlay", type=str_to_bool,
                   default=post_cfg.get("draw_cylinder_overlay", True, bool),
                   help="Draw the cylinder/body outline on generated flow plots and GIFs")
    p.add_argument("--verbose",  type=str_to_bool, default=cfg.get("verbose", True, bool),
                   help="Print periodic diagnostics during the time loop")

    # Boundary condition configuration
    p.add_argument("--bc-left", type=str,
                   default=cfg.get("bc_left", BCType.INFLOW, str),
                   help="Left boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--bc-right", type=str,
                   default=cfg.get("bc_right", BCType.OUTFLOW, str),
                   help="Right boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--bc-bottom", type=str,
                   default=cfg.get("bc_bottom", BCType.WALL, str),
                   help="Bottom boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--bc-top", type=str,
                   default=cfg.get("bc_top", BCType.WALL, str),
                   help="Top boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--inflow-u", type=float,
                   default=cfg.get("inflow_u", 1.0, float),
                   help="Inflow/farfield x-velocity component")
    p.add_argument("--inflow-v", type=float,
                   default=cfg.get("inflow_v", 0.0, float),
                   help="Inflow/farfield y-velocity component")
    p.add_argument("--initial-v-perturbation-percent", type=float,
                   default=cfg.get("initial_v_perturbation_percent", 0.0, float),
                   help="One-time initial y-velocity perturbation as a percent of inflow_u")
    p.add_argument("--inflow-w", type=float,
                   default=cfg.get("inflow_w", 0.0, float),
                   help="Inflow/farfield z-velocity component for 3-D")
    p.add_argument("--farfield-mode", type=str,
                   default=_normalize_farfield_mode(
                       cfg.get("farfield_mode", FarfieldMode.DIRICHLET, str)),
                   help="Farfield enforcement: dirichlet or neumann")
    p.add_argument("--wall-slip-mode", type=str,
                   default=cfg.get("wall_slip_mode", "no-slip", str),
                   help="Wall tangential model: no-slip or free-slip")
    p.add_argument("--wall-penetration", type=str_to_bool,
                   default=cfg.get("wall_penetration", False, bool),
                   help="Allow non-zero wall-normal velocity on wall boundaries")
    p.add_argument("--wall-normal-velocity", type=float,
                   default=cfg.get("wall_normal_velocity", 0.0, float),
                   help="Wall-normal velocity used when wall_penetration=true")
    p.add_argument("--outflow-mode", type=str,
                   default=cfg.get("outflow_mode", "convective", str),
                   help="Outflow update mode: convective or zero-gradient")
    p.add_argument("--outflow-speed", type=float,
                   default=cfg.get("outflow_speed", 1.0, float),
                   help="Convective outflow wave speed")
    p.add_argument("--auto-coeff-t-min", type=float,
                   default=post_cfg.get("auto_coeff_t_min", 0.5, float),
                   help="Minimum time used when auto-generating coefficient history")
    p.add_argument("--auto-aero-t-min", type=float,
                   default=post_cfg.get("auto_aero_t_min", 1.0, float),
                   help="Minimum time used for auto-generated aerodynamic frequency analysis")
    p.add_argument("--auto-aero-stats-t-min", type=float,
                   default=post_cfg.get(
                       "auto_aero_stats_t_min",
                       post_cfg.get("auto_aero_t_min", 1.0, float),
                       float,
                   ),
                   help="Minimum time used for auto-generated drag/lift statistics")
    args = p.parse_args()

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
    args.experimental_config_enabled = bool(experimental_config_enabled)
    _apply_y_truncation_to_args(args, p)
    return args


def _grid_metadata_path(args) -> str:
    name = "uniform_grid.npz" if args.grid_type == "uniform" else "nonuniform_grid.npz"
    return os.path.join(args.outdir, name)


def _expected_nonuniform_band(args) -> tuple[float, float, float, float]:
    start_x = float(np.clip(args.uniform_x_start, args.x_min, args.x_max))
    end_x = float(np.clip(args.uniform_x_end, args.x_min, args.x_max))
    if end_x <= start_x:
        raise ValueError("uniform_x_end must be greater than uniform_x_start")

    start_y = float(args.uniform_y_start)
    end_y = float(args.uniform_y_end)
    return start_x, end_x, start_y, end_y


def _ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _latest_snapshot_path(outdir: str) -> str | None:
    if not os.path.isdir(outdir):
        return None
    snapshots = [
        os.path.join(outdir, name)
        for name in os.listdir(outdir)
        if (
            name.startswith("snap_")
            and name.endswith(".npz")
            and os.path.getsize(os.path.join(outdir, name)) > 0
        )
    ]
    best_path = None
    best_time = -np.inf
    for path in snapshots:
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in ("u", "v", "p", "t"):
                    if key not in data:
                        raise KeyError(key)
                snapshot_time = float(data["t"])
            if snapshot_time > best_time:
                best_time = snapshot_time
                best_path = path
        except (OSError, ValueError, KeyError):
            continue
    return best_path


def _resolve_resume_snapshot(args) -> str | None:
    resume_from = str(args.resume_from).strip()
    if resume_from:
        path = resume_from
        if not os.path.isabs(path):
            path = os.path.join(SCRIPT_DIR, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Resume snapshot not found: {path}")
        return path
    if args.resume_latest:
        path = _latest_snapshot_path(args.outdir)
        if path is None:
            raise FileNotFoundError(
                f"No snap_*.npz files found in resume outdir: {args.outdir}"
            )
        return path
    return None


def _resolve_cylinder_geometry(args) -> tuple[float, float, float]:
    cx = args.cylinder_center_x if args.cylinder_center_x >= 0.0 else args.x_min + 0.25 * args.lx
    cy = args.cylinder_center_y if args.cylinder_center_y >= 0.0 else args.y_min + 0.5 * args.ly
    radius = args.cylinder_radius if args.cylinder_radius > 0.0 else args.ly / 8.0
    return cx, cy, radius


def _resolve_cylinder_translation(args, cx: float, cy: float, radius: float) -> dict:
    mode = _normalize_cylinder_translation_mode(args.cylinder_translation_mode)
    diameter = 2.0 * float(radius)
    x_percent = max(float(args.cylinder_translation_x_percent), 0.0)
    y_percent = max(float(args.cylinder_translation_y_percent), 0.0)

    amplitude_x = 0.01 * x_percent * diameter if mode in {"oscillatory-x", "oscillatory-xy"} else 0.0
    amplitude_y = 0.01 * y_percent * diameter if mode in {"oscillatory-y", "oscillatory-xy"} else 0.0
    frequency = max(float(args.cylinder_translation_frequency), 0.0)

    if mode != "stationary" and frequency <= 0.0:
        raise ValueError("cylinder translation requires a positive frequency")

    if (
        mode != "stationary"
        and args.grid_type == "nonuniform"
        and args.uniform_x_start is not None
        and args.uniform_y_start is not None
    ):
        x0, x1, y0, y1 = _expected_nonuniform_band(args)
        swept_x0 = cx - radius - amplitude_x
        swept_x1 = cx + radius + amplitude_x
        swept_y0 = cy - radius - amplitude_y
        swept_y1 = cy + radius + amplitude_y
        if swept_x0 < x0 or swept_x1 > x1 or swept_y0 < y0 or swept_y1 > y1:
            raise ValueError(
                "moving cylinder sweep must stay inside the concentrated mesh "
                f"core: sweep=({swept_x0:.4g},{swept_x1:.4g}) x "
                f"({swept_y0:.4g},{swept_y1:.4g}), core=({x0:.4g},{x1:.4g}) x "
                f"({y0:.4g},{y1:.4g})"
            )

    return {
        "mode": mode,
        "amplitude_x": amplitude_x,
        "amplitude_y": amplitude_y,
        "frequency": frequency,
        "phase_rad": np.deg2rad(float(args.cylinder_translation_phase_deg)),
        "x_percent": x_percent,
        "y_percent": y_percent,
    }


def _cylinder_center_at_time(args, time: float) -> tuple[float, float, float]:
    cx, cy, radius = _resolve_cylinder_geometry(args)
    cfg = _resolve_cylinder_translation(args, cx, cy, radius)
    if cfg["mode"] == "stationary":
        return cx, cy, radius
    theta = 2.0 * np.pi * cfg["frequency"] * float(time) + cfg["phase_rad"]
    displacement = np.sin(theta)
    return (
        float(cx + cfg["amplitude_x"] * displacement),
        float(cy + cfg["amplitude_y"] * displacement),
        radius,
    )


def _resolve_indent_geometry(args, radius: float) -> tuple[float, float]:
    indent_width = float(args.cylinder_indent_width)
    indent_depth = float(args.cylinder_indent_depth)
    if indent_width <= 0.0:
        indent_width = 0.6 * radius
    if indent_depth <= 0.0:
        indent_depth = 0.35 * radius
    return indent_width, indent_depth


def _resolve_experiment_overrides(args) -> str:
    experiment = _normalize_cylinder_experiment_mode(
        getattr(args, "cylinder_experiment", "circle")
    )
    if experiment == "circle":
        return "circle"
    if experiment == "top-indent":
        return "circle-with-top-indent"
    if experiment == "square":
        return "square"
    if experiment == "airfoil":
        return "airfoil"
    return _normalize_cylinder_geometry_mode(
        getattr(args, "cylinder_geometry_mode", getattr(args, "ibm_shape", "circle"))
    )


def _resolve_airfoil_geometry(args, radius: float) -> tuple[float, float, float]:
    chord = float(getattr(args, "airfoil_chord", -1.0))
    if chord <= 0.0:
        chord = 2.0 * float(radius)
    thickness_ratio = 0.01 * float(getattr(args, "airfoil_thickness_percent", 12.0))
    angle_deg = float(getattr(args, "airfoil_angle_deg", 0.0))
    if thickness_ratio <= 0.0:
        raise ValueError("airfoil_thickness_percent must be positive")
    return chord, thickness_ratio, angle_deg


def _airfoil_outline_points(
    cx: float,
    cy: float,
    chord: float,
    thickness_ratio: float,
    angle_deg: float,
    n_points: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, int(n_points))
    yt = 5.0 * thickness_ratio * chord * (
        0.2969 * np.sqrt(s)
        - 0.1260 * s
        - 0.3516 * s ** 2
        + 0.2843 * s ** 3
        - 0.1015 * s ** 4
    )
    x_upper = s * chord - 0.5 * chord
    x_lower = x_upper[::-1]
    y_upper = yt
    y_lower = -yt[::-1]
    x_local = np.concatenate([x_upper, x_lower])
    y_local = np.concatenate([y_upper, y_lower])
    angle = np.deg2rad(float(angle_deg))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = float(cx) + cos_a * x_local - sin_a * y_local
    y = float(cy) + sin_a * x_local + cos_a * y_local
    return x, y


def _plot_ibm_outline(
    ax,
    args,
    color: str = "white",
    linewidth: float = 1.6,
    time: float = 0.0,
    center_override: tuple[float, float] | None = None,
) -> None:
    cx, cy, radius = _cylinder_center_at_time(args, time)
    if center_override is not None:
        cx, cy = center_override
    shape = _resolve_experiment_overrides(args)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    if shape == "circle":
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=6)
        return
    if shape == "square":
        half_side = radius
        left = cx - half_side
        right = cx + half_side
        bottom = cy - half_side
        top = cy + half_side
        ax.plot(
            [left, right, right, left, left],
            [bottom, bottom, top, top, bottom],
            color=color,
            linewidth=linewidth,
            zorder=6,
        )
        return
    if shape == "airfoil":
        chord, thickness_ratio, angle_deg = _resolve_airfoil_geometry(args, radius)
        x_airfoil, y_airfoil = _airfoil_outline_points(
            cx,
            cy,
            chord,
            thickness_ratio,
            angle_deg,
        )
        ax.plot(x_airfoil, y_airfoil, color=color, linewidth=linewidth, zorder=6)
        return

    indent_width, indent_depth = _resolve_indent_geometry(args, radius)
    notch_left = cx - 0.5 * indent_width
    notch_right = cx + 0.5 * indent_width
    notch_bottom = cy + radius - indent_depth
    wall_top = cy + np.sqrt(max(radius ** 2 - (0.5 * indent_width) ** 2, 0.0))
    notch_mask = (
        (x >= notch_left)
        & (x <= notch_right)
        & (y >= notch_bottom)
    )
    if np.all(notch_mask):
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=6)
        return

    x_visible = x.copy()
    y_visible = y.copy()
    x_visible[notch_mask] = np.nan
    y_visible[notch_mask] = np.nan
    ax.plot(x_visible, y_visible, color=color, linewidth=linewidth, zorder=6)

    ax.plot(
        [notch_left, notch_left, notch_right, notch_right],
        [wall_top, notch_bottom, notch_bottom, wall_top],
        color=color,
        linewidth=linewidth,
        zorder=6,
    )


def _cylinder_angular_velocity(args, time: float) -> float:
    mode = _normalize_cylinder_rotation_mode(args.cylinder_rotation_mode)
    if mode == "constant":
        return float(args.cylinder_rotation_amplitude)
    if mode != "oscillatory":
        return 0.0
    phase = np.deg2rad(float(args.cylinder_rotation_phase_deg))
    return float(
        args.cylinder_rotation_amplitude *
        np.sin(2.0 * np.pi * args.cylinder_rotation_frequency * time + phase)
    )


def _kinematic_viscosity(args, inflow_u: float, cylinder_radius: float | None) -> float:
    if args.cylinder and args.re_is_cylinder_based and cylinder_radius is not None:
        return inflow_u * (2.0 * cylinder_radius) / args.re
    return 1.0 / args.re


def _snapshot_metadata(args, solver) -> dict:
    metadata = {
        "t": solver.t,
        "nx": args.nx,
        "ny": args.ny,
        "lx": args.lx,
        "ly": args.ly,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "re": args.re,
        "re_is_cylinder_based": bool(args.re_is_cylinder_based),
        "experimental_config_enabled": bool(args.experimental_config_enabled),
        "cylinder_free_x_dof": bool(args.cylinder_free_x_dof),
        "cylinder_free_x_mass": float(args.cylinder_free_x_mass),
        "cylinder_free_x_damping": float(args.cylinder_free_x_damping),
        "cylinder_free_x_stiffness": float(args.cylinder_free_x_stiffness),
        "cylinder_free_x_force_relaxation": float(args.cylinder_free_x_force_relaxation),
        "cylinder_free_x_max_displacement_percent": float(args.cylinder_free_x_max_displacement_percent),
        "cylinder_free_x_max_speed": float(args.cylinder_free_x_max_speed),
        "cylinder_free_y_dof": bool(args.cylinder_free_y_dof),
        "cylinder_free_y_mass": float(args.cylinder_free_y_mass),
        "cylinder_free_y_damping": float(args.cylinder_free_y_damping),
        "cylinder_free_y_stiffness": float(args.cylinder_free_y_stiffness),
        "cylinder_free_y_force_relaxation": float(args.cylinder_free_y_force_relaxation),
        "cylinder_free_y_max_displacement_percent": float(args.cylinder_free_y_max_displacement_percent),
        "cylinder_free_y_max_speed": float(args.cylinder_free_y_max_speed),
        "ibm_force_x": solver.last_ibm_force_x,
        "ibm_force_y": solver.last_ibm_force_y,
        "y_truncation_enabled": bool(args.y_truncation_enabled),
        "y_truncation_cells_each_end": int(args.y_truncation_cells_each_end),
        "source_ny_before_y_truncation": int(args.source_ny_before_y_truncation),
        "source_y_min_before_y_truncation": float(args.source_y_min_before_y_truncation),
        "source_y_max_before_y_truncation": float(args.source_y_max_before_y_truncation),
        "cylinder_enabled": bool(args.cylinder),
        "cylinder_omega": _cylinder_angular_velocity(args, solver.t),
    }
    if args.cylinder:
        cx, cy, radius = _cylinder_center_at_time(args, solver.t)
        if (args.cylinder_free_x_dof or args.cylinder_free_y_dof) and solver.ibm is not None:
            free_state = solver.ibm.first_free_y_circle_state()
            if free_state is not None:
                cx = free_state["center_x"]
                cy = free_state["center_y"]
                metadata["cylinder_free_x_velocity"] = free_state["velocity_x"]
                metadata["cylinder_free_x_displacement"] = free_state["displacement_x"]
                metadata["cylinder_free_y_velocity"] = free_state["velocity_y"]
                metadata["cylinder_free_y_displacement"] = free_state["displacement_y"]
        metadata["cylinder_center_x"] = cx
        metadata["cylinder_center_y"] = cy
        metadata["cylinder_radius"] = radius
    return metadata


def _snapshot_extra_fields(solver) -> dict:
    return {
        "ibm_forcing_u_face": solver.last_ibm_forcing_u,
        "ibm_forcing_v_face": solver.last_ibm_forcing_v,
        "ibm_forcing_x_cell": solver.last_ibm_forcing_xc,
        "ibm_forcing_y_cell": solver.last_ibm_forcing_yc,
    }


def _grid_matches_args(metadata: dict, grid, args) -> bool:
    metadata_type = str(metadata.get("grid_type", "uniform")).strip().lower()
    meta_lx = float(metadata.get("lx", grid.lx))
    meta_ly = float(metadata.get("ly", grid.ly))
    meta_x_min = float(metadata.get("x_min", 0.0))
    meta_y_min = float(metadata.get("y_min", 0.0))
    meta_x_max = float(metadata.get("x_max", meta_x_min + meta_lx))
    meta_y_max = float(metadata.get("y_max", meta_y_min + meta_ly))

    return (
        grid.nx == args.nx and
        grid.ny == args.ny and
        np.isclose(grid.lx, args.lx) and
        np.isclose(grid.ly, args.ly) and
        np.isclose(meta_x_min, args.x_min) and
        np.isclose(meta_x_max, args.x_max) and
        np.isclose(meta_y_min, args.y_min) and
        np.isclose(meta_y_max, args.y_max) and
        ((args.grid_type == "uniform" and grid.is_uniform and metadata_type == "uniform") or (
            args.grid_type == "nonuniform" and not grid.is_uniform and metadata_type == "nonuniform"))
    )


def _nonuniform_metadata_matches_args(metadata: dict, args) -> bool:
    # Only accept nonuniform files built with the current center-uniform scheme.
    mode = str(metadata.get("nonuniform_mode", "")).strip().lower()
    if mode != "center-uniform":
        return False

    band_start_x, band_end_x, band_start_y, band_end_y = _expected_nonuniform_band(
        args)
    common_matches = (
        np.isclose(float(metadata.get("beta_x", np.nan)), float(args.beta_x)) and
        np.isclose(float(metadata.get("beta_y", np.nan)), float(args.beta_y)) and
        np.isclose(float(metadata.get("band_start_x", np.nan)), float(band_start_x)) and
        np.isclose(float(metadata.get("band_end_x", np.nan)), float(band_end_x)) and
        np.isclose(float(metadata.get("band_start_y", np.nan)), float(band_start_y)) and
        np.isclose(float(metadata.get("band_end_y", np.nan)),
                   float(band_end_y))
    )
    if not common_matches:
        return False

    def _both_nan_or_close(a: float, b: float) -> bool:
        return (np.isnan(a) and np.isnan(b)) or np.isclose(a, b)

    meta_uniform_x_start = metadata.get("uniform_x_start", np.nan)
    meta_uniform_x_end = metadata.get("uniform_x_end", np.nan)
    meta_uniform_y_start = metadata.get("uniform_y_start", np.nan)
    meta_uniform_y_end = metadata.get("uniform_y_end", np.nan)

    expected_uniform_x_start = np.nan if args.uniform_x_start is None else float(
        args.uniform_x_start)
    expected_uniform_x_end = np.nan if args.uniform_x_end is None else float(
        args.uniform_x_end)
    expected_uniform_y_start = np.nan if args.uniform_y_start is None else float(
        args.uniform_y_start)
    expected_uniform_y_end = np.nan if args.uniform_y_end is None else float(
        args.uniform_y_end)

    return (
        _both_nan_or_close(float(meta_uniform_x_start), expected_uniform_x_start) and
        _both_nan_or_close(float(meta_uniform_x_end), expected_uniform_x_end) and
        _both_nan_or_close(float(meta_uniform_y_start), expected_uniform_y_start) and
        _both_nan_or_close(float(meta_uniform_y_end), expected_uniform_y_end)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_uniform_grid(args):
    """Build the runtime grid and write its metadata before the solver starts."""
    yf = getattr(args, "_truncated_yf", None)
    grid = CartesianGrid(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        x_min=args.x_min,
        y_min=args.y_min,
        yf=yf,
    )
    os.makedirs(args.outdir, exist_ok=True)
    if getattr(args, "y_truncation_enabled", False):
        metadata = grid.to_metadata()
        metadata["y_truncation_cells_each_end"] = int(args.y_truncation_cells_each_end)
        metadata["source_ny_before_y_truncation"] = int(
            args.source_ny_before_y_truncation
        )
        metadata["source_y_min_before_y_truncation"] = float(
            args.source_y_min_before_y_truncation
        )
        metadata["source_y_max_before_y_truncation"] = float(
            args.source_y_max_before_y_truncation
        )
        save_grid_metadata_dict(_grid_metadata_path(args), metadata)
    else:
        save_grid_metadata(_grid_metadata_path(args), grid)
    return grid


def prepare_nonuniform_grid(args):
    """Build the runtime non-uniform grid and write its metadata before startup."""
    source_ny = int(getattr(args, "source_ny_before_y_truncation", args.ny))
    source_y_min = float(getattr(args, "source_y_min_before_y_truncation", args.y_min))
    source_ly = float(getattr(args, "source_ly_before_y_truncation", args.ly))
    metadata = build_nonuniform_grid_metadata(
        nx=args.nx,
        ny=source_ny,
        lx=args.lx,
        ly=source_ly,
        beta_x=args.beta_x,
        beta_y=args.beta_y,
        x_min=args.x_min,
        y_min=source_y_min,
        uniform_x_start=args.uniform_x_start,
        uniform_x_end=args.uniform_x_end,
        uniform_y_start=args.uniform_y_start,
        uniform_y_end=args.uniform_y_end,
    )
    if getattr(args, "y_truncation_enabled", False):
        metadata = _truncate_grid_metadata_y(
            metadata,
            int(args.y_truncation_cells_each_end),
        )
    os.makedirs(args.outdir, exist_ok=True)
    save_grid_metadata_dict(_grid_metadata_path(args), metadata)
    return CartesianGrid.from_metadata(metadata)


def get_runtime_grid(args):
    """Load a pre-generated runtime grid when available, otherwise create one."""
    grid_path = _grid_metadata_path(args)
    if os.path.exists(grid_path):
        try:
            metadata = load_grid_metadata_dict(grid_path)
            grid = load_prepared_grid(grid_path)
            if _grid_matches_args(metadata, grid, args):
                if args.grid_type == "nonuniform" and not _nonuniform_metadata_matches_args(metadata, args):
                    print(
                        "Warning: Prepared nonuniform grid metadata does not match "
                        "requested beta/core settings; regenerating grid."
                    )
                else:
                    return grid, True
        except (ValueError, OSError, KeyError) as exc:
            print(
                f"Warning: Ignoring incompatible prepared grid at {grid_path}: {exc}"
            )
    if args.grid_type == "nonuniform":
        return prepare_nonuniform_grid(args), False
    return prepare_uniform_grid(args), False


def run(args, grid=None, grid_loaded_from_file=False):
    # ------------------------------------------------------------------
    # MPI setup
    # ------------------------------------------------------------------
    decomp = ParallelDecomposition(args.ny)
    rank = decomp.rank
    is_root = rank == 0

    if is_root and args.verbose:
        print("=" * 60)
        print("  2-D Incompressible Navier-Stokes Solver")
        print("=" * 60)
        print(f"  Config file   : {args.config}")
        print(
            "  Experiment cfg: "
            f"{'enabled' if args.experimental_config_enabled else 'ignored'} "
            f"({args.experiment_config})"
        )
        print(f"  Grid          : {args.nx} x {args.ny}")
        print(f"  Grid type     : {args.grid_type}")
        if args.grid_type == "nonuniform":
            print("  Grid mode     : center-uniform")
        print(
            "  Domain        : "
            f"x=[{args.x_min}, {args.x_max}] (Lx={args.lx}), "
            f"y=[{args.y_min}, {args.y_max}] (Ly={args.ly})"
        )
        if args.y_truncation_enabled:
            print(
                "  Y truncation  : "
                f"dropped {args.y_truncation_cells_each_end} cells from "
                "bottom and top "
                f"(source ny={args.source_ny_before_y_truncation}, "
                f"source y=[{args.source_y_min_before_y_truncation}, "
                f"{args.source_y_max_before_y_truncation}])"
            )
        print(f"  Reynolds no.  : {args.re}")
        print(f"  End time      : {args.t_end}")
        print(f"  MPI ranks     : {decomp.size}")
        print(f"  IBM cylinder  : {args.cylinder}")
        print(
            f"  Grid source   : {'pre-generated file' if grid_loaded_from_file else 'generated at startup'}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    if grid is None:
        grid, grid_loaded_from_file = get_runtime_grid(args)

    # ------------------------------------------------------------------
    # Boundary conditions (fully configurable from config/CLI)
    # ------------------------------------------------------------------
    bc_left = _normalize_bc_type(args.bc_left, BCType.INFLOW)
    bc_right = _normalize_bc_type(args.bc_right, BCType.OUTFLOW)
    bc_bottom = _normalize_bc_type(args.bc_bottom, BCType.WALL)
    bc_top = _normalize_bc_type(args.bc_top, BCType.WALL)

    bc = BoundaryConfig(
        left=bc_left,
        right=bc_right,
        bottom=bc_bottom,
        top=bc_top,
        u_inf=args.inflow_u,
        v_inf=args.inflow_v,
        w_inf=args.inflow_w,
        wall_slip_mode=str(args.wall_slip_mode).strip().lower(),
        wall_penetration=bool(args.wall_penetration),
        wall_normal_velocity=args.wall_normal_velocity,
        farfield_mode=_normalize_farfield_mode(args.farfield_mode),
        outflow_mode=str(args.outflow_mode).strip().lower(),
        outflow_speed=args.outflow_speed,
    )

    # ------------------------------------------------------------------
    # Immersed boundary (optional cylinder)
    # ------------------------------------------------------------------
    ibm = ImmersedBoundary(grid)
    r = None
    if args.cylinder:
        cx, cy, r = _resolve_cylinder_geometry(args)
        ibm_shape = _resolve_experiment_overrides(args)
        rotation_mode = _normalize_cylinder_rotation_mode(args.cylinder_rotation_mode)
        translation_cfg = _resolve_cylinder_translation(args, cx, cy, r)
        translation_mode = translation_cfg["mode"]
        free_cylinder_dof = args.cylinder_free_x_dof or args.cylinder_free_y_dof
        if free_cylinder_dof:
            if ibm_shape != "circle":
                raise ValueError("free cylinder DoF currently supports circle bodies only")
            if rotation_mode != "stationary" or translation_mode != "stationary":
                raise ValueError(
                    "free cylinder DoF cannot be combined with prescribed "
                    "rotation or translation"
                )
        if ibm_shape != "circle" and rotation_mode != "stationary":
            raise ValueError(
                f"{ibm_shape} currently supports stationary IBM bodies only"
            )
        if ibm_shape != "circle" and translation_mode != "stationary":
            raise ValueError(
                f"{ibm_shape} currently supports stationary IBM bodies only"
            )
        if rotation_mode != "stationary" and translation_mode != "stationary":
            raise ValueError(
                "cylinder rotation and cylinder translation are mutually exclusive"
            )
        if free_cylinder_dof:
            if args.cylinder_free_x_dof and args.cylinder_free_y_dof:
                if not np.isclose(args.cylinder_free_x_mass, args.cylinder_free_y_mass):
                    raise ValueError("free x/y DoF currently require matching mass")
                if not np.isclose(args.cylinder_free_x_damping, args.cylinder_free_y_damping):
                    raise ValueError("free x/y DoF currently require matching damping")
                if not np.isclose(args.cylinder_free_x_stiffness, args.cylinder_free_y_stiffness):
                    raise ValueError("free x/y DoF currently require matching stiffness")
                if not np.isclose(
                    args.cylinder_free_x_force_relaxation,
                    args.cylinder_free_y_force_relaxation,
                ):
                    raise ValueError("free x/y DoF currently require matching force relaxation")
                if not np.isclose(
                    args.cylinder_free_x_max_displacement_percent,
                    args.cylinder_free_y_max_displacement_percent,
                ):
                    raise ValueError("free x/y DoF currently require matching max displacement")
                if not np.isclose(args.cylinder_free_x_max_speed, args.cylinder_free_y_max_speed):
                    raise ValueError("free x/y DoF currently require matching max speed")
            free_mass = (
                args.cylinder_free_x_mass
                if args.cylinder_free_x_dof
                else args.cylinder_free_y_mass
            )
            free_damping = (
                args.cylinder_free_x_damping
                if args.cylinder_free_x_dof
                else args.cylinder_free_y_damping
            )
            free_stiffness = (
                args.cylinder_free_x_stiffness
                if args.cylinder_free_x_dof
                else args.cylinder_free_y_stiffness
            )
            free_force_relaxation = (
                args.cylinder_free_x_force_relaxation
                if args.cylinder_free_x_dof
                else args.cylinder_free_y_force_relaxation
            )
            free_max_displacement_percent = (
                args.cylinder_free_x_max_displacement_percent
                if args.cylinder_free_x_dof
                else args.cylinder_free_y_max_displacement_percent
            )
            free_max_speed = (
                args.cylinder_free_x_max_speed
                if args.cylinder_free_x_dof
                else args.cylinder_free_y_max_speed
            )
            ibm.add_free_y_circle(
                cx,
                cy,
                r,
                mass=free_mass,
                damping=free_damping,
                stiffness=free_stiffness,
                initial_velocity_x=args.cylinder_free_x_initial_velocity,
                initial_velocity_y=args.cylinder_free_y_initial_velocity,
                force_relaxation=free_force_relaxation,
                max_displacement=(
                    0.01
                    * free_max_displacement_percent
                    * 2.0
                    * r
                ),
                max_speed=free_max_speed,
                free_x=args.cylinder_free_x_dof,
                free_y=args.cylinder_free_y_dof,
            )
        elif translation_mode != "stationary":
            ibm.add_translating_circle(
                cx,
                cy,
                r,
                amplitude_x=translation_cfg["amplitude_x"],
                amplitude_y=translation_cfg["amplitude_y"],
                frequency=translation_cfg["frequency"],
                phase=translation_cfg["phase_rad"],
            )
        elif rotation_mode == "oscillatory":
            ibm.add_rotating_circle(
                cx,
                cy,
                r,
                omega_amplitude=args.cylinder_rotation_amplitude,
                frequency=args.cylinder_rotation_frequency,
                phase=np.deg2rad(args.cylinder_rotation_phase_deg),
            )
        elif rotation_mode == "constant":
            ibm.add_constant_rotating_circle(
                cx,
                cy,
                r,
                omega=args.cylinder_rotation_amplitude,
            )
        else:
            if ibm_shape == "circle-with-top-indent":
                indent_width, indent_depth = _resolve_indent_geometry(args, r)
                ibm.add_circle_with_top_indent(
                    cx,
                    cy,
                    r,
                    indent_width=indent_width,
                    indent_depth=indent_depth,
                )
            elif ibm_shape == "square":
                ibm.add_rectangle(cx - r, cx + r, cy - r, cy + r)
            elif ibm_shape == "airfoil":
                chord, thickness_ratio, angle_deg = _resolve_airfoil_geometry(args, r)
                ibm.add_naca_00xx_airfoil(
                    cx,
                    cy,
                    chord=chord,
                    thickness_ratio=thickness_ratio,
                    angle_deg=angle_deg,
                )
            else:
                ibm.add_circle(cx, cy, r)
        if is_root and args.verbose:
            print(
                f"  IBM cylinder: centre=({cx:.2f},{cy:.2f}), r={r:.4f}, "
                f"shape={ibm_shape}, "
                f"experiment={_normalize_cylinder_experiment_mode(args.cylinder_experiment)}"
            )
            if ibm_shape == "circle-with-top-indent":
                indent_width, indent_depth = _resolve_indent_geometry(args, r)
                print(
                    "  Top indent   : "
                    f"width={indent_width:.4f}, depth={indent_depth:.4f}"
                )
            elif ibm_shape == "square":
                print(f"  Square body  : side={2.0 * r:.4f}")
            elif ibm_shape == "airfoil":
                chord, thickness_ratio, angle_deg = _resolve_airfoil_geometry(args, r)
                print(
                    "  Airfoil body : "
                    f"NACA 00{100.0 * thickness_ratio:.0f}, "
                    f"chord={chord:.4f}, alpha={angle_deg:.4g} deg"
                )
            if free_cylinder_dof:
                free_axes = "".join(
                    axis
                    for axis, enabled in (
                        ("x", args.cylinder_free_x_dof),
                        ("y", args.cylinder_free_y_dof),
                    )
                    if enabled
                )
                print(
                    "  Cylinder DOF : "
                    f"free-{free_axes}, "
                    f"m={free_mass:.4g}, "
                    f"c={free_damping:.4g}, "
                    f"k={free_stiffness:.4g}, "
                    f"vx0={args.cylinder_free_x_initial_velocity:.4g}, "
                    f"vy0={args.cylinder_free_y_initial_velocity:.4g}, "
                    f"relax={free_force_relaxation:.4g}, "
                    f"max displacement={free_max_displacement_percent:.4g}% D, "
                    f"max speed={free_max_speed:.4g}"
                )
            if rotation_mode == "oscillatory":
                print(
                    "  Cylinder rot.: "
                    f"omega(t)={args.cylinder_rotation_amplitude:.4g}"
                    f"*sin(2*pi*{args.cylinder_rotation_frequency:.4g}*t + "
                    f"{args.cylinder_rotation_phase_deg:.4g} deg)"
                )
            elif rotation_mode == "constant":
                print(
                    "  Cylinder rot.: "
                    f"omega(t)={args.cylinder_rotation_amplitude:.4g}"
                )
            if translation_mode != "stationary":
                print(
                    "  Cylinder move: "
                    f"mode={translation_mode}, "
                    f"Ax={translation_cfg['amplitude_x']:.4g} "
                    f"({translation_cfg['x_percent']:.4g}% D), "
                    f"Ay={translation_cfg['amplitude_y']:.4g} "
                    f"({translation_cfg['y_percent']:.4g}% D), "
                    f"f={translation_cfg['frequency']:.4g}, "
                    f"phase={args.cylinder_translation_phase_deg:.4g} deg"
                )

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    nu = _kinematic_viscosity(args, bc.u_inf, r)

    if is_root and args.verbose and args.cylinder and args.re_is_cylinder_based and r is not None:
        d_cyl = 2.0 * r
        print(
            f"  Re interpretation: Re_D={args.re} with D={d_cyl:.4f} -> nu={nu:.6g}")

    initial_v_perturbation = 0.01 * \
        args.initial_v_perturbation_percent * bc.u_inf
    solver = FractionalStepSolver(
        grid,
        bc,
        nu,
        ibm=ibm,
    )
    solver.init_fields(
        u0=bc.u_inf,
        v0=bc.v_inf,
        initial_v_perturbation=initial_v_perturbation,
    )
    if is_root and args.verbose and args.initial_v_perturbation_percent != 0.0:
        print(
            "  Initial v perturbation: "
            f"{args.initial_v_perturbation_percent:.3g}% of inflow_u "
            f"-> dv={initial_v_perturbation:.6g}"
        )

    resume_snapshot = _resolve_resume_snapshot(args)
    if resume_snapshot is not None:
        u_restart, v_restart, p_restart, t_restart, _ = load_snapshot(
            resume_snapshot, fmt="numpy"
        )
        expected_shapes = {
            "u": solver.u.shape,
            "v": solver.v.shape,
            "p": solver.p.shape,
        }
        found_shapes = {
            "u": u_restart.shape,
            "v": v_restart.shape,
            "p": p_restart.shape,
        }
        if found_shapes != expected_shapes:
            raise ValueError(
                "Resume snapshot field shapes do not match this run: "
                f"expected={expected_shapes}, found={found_shapes}"
            )
        solver.u[:, :] = u_restart
        solver.v[:, :] = v_restart
        solver.p[:, :] = p_restart
        solver.t = float(t_restart)
        if solver.t >= args.t_end - 1e-12:
            raise ValueError(
                f"Resume snapshot t={solver.t:.4f} is already at/after "
                f"t_end={args.t_end:.4f}"
            )
        if is_root and args.verbose:
            print(
                f"  Resuming from : {resume_snapshot} "
                f"(t={solver.t:.4f})"
            )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if is_root:
        os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    t_save_next = 0.0
    if resume_snapshot is not None:
        t_save_next = (np.floor(solver.t / args.save_dt) + 1.0) * args.save_dt
    step_count = 0

    if is_root and args.verbose:
        print(f"\n  Starting time loop …")

    while solver.t < args.t_end - 1e-12:
        dt = solver.suggest_dt(cfl_target=args.cfl)
        dt = min(dt, args.t_end - solver.t)

        solver.step(dt)
        step_count += 1

        # ---- diagnostics ----
        if is_root and args.verbose and step_count % 50 == 0:
            div_max = np.max(np.abs(solver.divergence()))
            cfl_val = solver.cfl(dt)
            print(f"  t={solver.t:8.4f}  dt={dt:.2e}  "
                  f"|div u|_max={div_max:.2e}  CFL={cfl_val:.3f}")

        # ---- save snapshot ----
        if solver.t >= t_save_next - 1e-12:
            if is_root:
                snap_path = os.path.join(
                    args.outdir, f"snap_{solver.t:08.4f}.npz")
                save_snapshot(snap_path, solver.u, solver.v, solver.p,
                              solver.t,
                              meta=_snapshot_metadata(args, solver),
                              extra=_snapshot_extra_fields(solver),
                              fmt="numpy")
            t_save_next += args.save_dt

    if is_root and args.verbose:
        print(f"\n  Done.  t_final={solver.t:.4f},  steps={step_count}")

    if is_root:
        _run_auto_outputs(grid, args)

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if args.plot and is_root:
        _plot_results(solver, grid, args)

    return solver


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_results(solver, grid, args):
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend for CI
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    nx, ny = grid.nx, grid.ny

    # Interpolate to cell centres
    u_c = 0.5 * (solver.u[:-1, :] + solver.u[1:, :])
    v_c = 0.5 * (solver.v[:, :-1] + solver.v[:, 1:])
    # 2-D scalar vorticity: omega_z = dv/dx - du/dy
    edge_x = 2 if grid.nx >= 3 else 1
    edge_y = 2 if grid.ny >= 3 else 1
    dv_dx = np.gradient(v_c, grid.xc, axis=0, edge_order=edge_x)
    du_dy = np.gradient(u_c, grid.yc, axis=1, edge_order=edge_y)
    omega = dv_dx - du_dy

    X, Y = np.meshgrid(grid.xc, grid.yc, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if args.cylinder and args.draw_cylinder_overlay:
        # Draw the immersed cylinder on every panel so geometry alignment
        # is visible in vorticity, pressure, and velocity plots.
        center_override = None
        if (args.cylinder_free_x_dof or args.cylinder_free_y_dof) and solver.ibm is not None:
            free_state = solver.ibm.first_free_y_circle_state()
            if free_state is not None:
                center_override = (
                    free_state["center_x"],
                    free_state["center_y"],
                )
        for ax in axes:
            _plot_ibm_outline(
                ax,
                args,
                color="black",
                linewidth=1.6,
                time=solver.t,
                center_override=center_override,
            )

    # Vorticity: use robust clipping + high-contrast diverging map
    # so coherent structures are easier to read.
    wmax = float(np.percentile(np.abs(omega), 99.0))
    wmax = max(wmax, 1e-8)
    levels = np.linspace(-wmax, wmax, 81)
    im0 = axes[0].contourf(X, Y, omega, levels=levels,
                           cmap="seismic", extend="both")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(r"Vorticity $\omega_z$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # Pressure
    im1 = axes[1].contourf(X, Y, solver.p, 50, cmap="RdBu_r")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Pressure p")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    fig.suptitle(f"Re={args.re:.0f},  t={solver.t:.3f}")
    fig.tight_layout()

    plot_path = os.path.join(_ensure_results_dir(), "result.png")
    fig.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close(fig)


def _plot_grid(grid, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib not available - skipping grid plot.")
        return

    x_edges = np.asarray(grid.xf, dtype=float)
    y_edges = np.asarray(grid.yf, dtype=float)
    Xf, Yf = np.meshgrid(x_edges, y_edges, indexing="ij")

    dx = np.asarray(grid.dx_cells, dtype=float)
    dy = np.asarray(grid.dy_cells, dtype=float)
    cell_area = dx[:, np.newaxis] * dy[np.newaxis, :]
    density = 1.0 / np.maximum(cell_area, 1e-30)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    mesh_ax = axes[0]
    vertical_segments = [
        [(float(x), float(grid.y_min)), (float(x), float(grid.y_max))]
        for x in x_edges
    ]
    horizontal_segments = [
        [(float(grid.x_min), float(y)), (float(grid.x_max), float(y))]
        for y in y_edges
    ]
    mesh_ax.add_collection(LineCollection(
        vertical_segments, colors="0.15", linewidths=0.6))
    mesh_ax.add_collection(LineCollection(
        horizontal_segments, colors="0.15", linewidths=0.6))
    mesh_ax.set_xlim(grid.x_min, grid.x_max)
    mesh_ax.set_ylim(grid.y_min, grid.y_max)
    mesh_ax.set_aspect("equal")
    mesh_ax.set_title("Physical Grid")
    mesh_ax.set_xlabel("x")
    mesh_ax.set_ylabel("y")

    density_ax = axes[1]
    density_im = density_ax.pcolormesh(
        Xf, Yf, density, shading="flat", cmap="viridis")
    fig.colorbar(density_im, ax=density_ax,
                 label=r"Cell density $1/(\Delta x \Delta y)$")
    density_ax.set_xlim(grid.x_min, grid.x_max)
    density_ax.set_ylim(grid.y_min, grid.y_max)
    density_ax.set_aspect("equal")
    density_ax.set_title("Point Concentration")
    density_ax.set_xlabel("x")
    density_ax.set_ylabel("y")

    spacing_ax = axes[2]
    spacing_ax.plot(grid.xc, grid.dx_cells,
                    label=r"$\Delta x$ at $x_c$", color="#d95f02", lw=2.0)
    spacing_ax.plot(grid.yc, grid.dy_cells,
                    label=r"$\Delta y$ at $y_c$", color="#1b9e77", lw=2.0)
    spacing_ax.set_title("Cell Spacing")
    spacing_ax.set_xlabel("Physical coordinate")
    spacing_ax.set_ylabel("Cell width")
    spacing_ax.grid(True, alpha=0.25)
    spacing_ax.legend()

    if args.grid_type == "nonuniform":
        from matplotlib.patches import Rectangle
        band_start_x, band_end_x, band_start_y, band_end_y = _expected_nonuniform_band(
            args)
        rect_width = band_end_x - band_start_x
        rect_height = band_end_y - band_start_y
        mesh_ax.add_patch(
            Rectangle(
                (band_start_x, band_start_y),
                rect_width,
                rect_height,
                fill=False,
                ec="#c1121f",
                ls="--",
                lw=1.4,
                alpha=0.9,
            )
        )
        density_ax.add_patch(
            Rectangle(
                (band_start_x, band_start_y),
                rect_width,
                rect_height,
                fill=False,
                ec="white",
                ls="--",
                lw=1.2,
                alpha=0.9,
            )
        )

    if args.cylinder and args.draw_cylinder_overlay:
        _plot_ibm_outline(mesh_ax, args, color="#001219", linewidth=1.8)
        _plot_ibm_outline(density_ax, args, color="white", linewidth=1.6)

    fig.suptitle(
        f"Grid type={args.grid_type}"
        f"{', mode=center-uniform' if args.grid_type == 'nonuniform' else ''}, "
        f"nx={grid.nx}, ny={grid.ny}, "
        f"dx_min={grid.dx_min:.4g}, dy_min={grid.dy_min:.4g}"
    )
    fig.tight_layout()

    plot_path = os.path.join(_ensure_results_dir(), "grid.png")
    fig.savefig(plot_path, dpi=180)
    print(f"  Grid plot saved to {plot_path}")
    plt.close(fig)


def _run_auto_outputs(grid, args):
    results_dir = _ensure_results_dir()

    if args.auto_generate_grid_spacing:
        _plot_grid(grid, args)

    need_aero_series = (
        args.auto_generate_coeff_history
        or args.auto_generate_aero_report
        or args.auto_generate_shedding_spectrum
        or args.auto_generate_drag_decomposition
        or args.auto_generate_drag_decomposition_plot
    )
    aero_series_path = os.path.join(results_dir, "aero.csv")
    aero_report_path = os.path.join(results_dir, "aero_report.txt")
    drag_decomposition_path = os.path.join(results_dir, "drag_decomposition.csv")
    aero_ready = False

    if need_aero_series:
        try:
            status = run_aero_analysis(
                indir=args.outdir,
                pattern="snap_*.npz",
                config=args.config,
                u_ref=args.inflow_u,
                use_cylinder_diameter=bool(
                    args.re_is_cylinder_based and args.cylinder),
                t_min=args.auto_aero_t_min,
                save_series=aero_series_path,
                save_drag_decomposition=(
                    drag_decomposition_path
                    if (
                        args.auto_generate_drag_decomposition
                        or args.auto_generate_drag_decomposition_plot
                    )
                    else None
                ),
                save_report=aero_report_path if args.auto_generate_aero_report else None,
                force_source="surface-full",
                surface_sample_offset_factor=args.surface_sample_offset_factor,
                stats_t_min=args.auto_aero_stats_t_min,
            )
            aero_ready = status == 0 and os.path.exists(aero_series_path)
            if status != 0:
                print("  Warning: automatic aerodynamic post-processing failed.")
        except Exception as exc:
            print(
                f"  Warning: automatic aerodynamic post-processing failed: {exc}"
            )

    if args.auto_generate_drag_decomposition_plot:
        if aero_ready and os.path.exists(drag_decomposition_path):
            try:
                plot_drag_decomposition(
                    drag_decomposition_path,
                    save_name="drag_decomposition.png",
                    t_min=args.auto_coeff_t_min,
                    results_dir=results_dir,
                )
            except Exception as exc:
                print(
                    "  Warning: automatic drag-decomposition plot failed: "
                    f"{exc}"
                )
        else:
            print(
                "  Warning: automatic drag-decomposition plot skipped because "
                "the decomposition series was not generated."
            )

    if args.auto_generate_coeff_history:
        if aero_ready:
            coeff_history_name = "coeff_history.png"
            try:
                plot_coeff_history(
                    aero_series_path,
                    save_name=coeff_history_name,
                    coeff_t_min=args.auto_coeff_t_min,
                    settled_coeff_t_min=args.auto_aero_stats_t_min,
                )
            except Exception as exc:
                print(
                    f"  Warning: automatic coefficient-history plot failed: {exc}")
        else:
            print(
                "  Warning: automatic coefficient-history plot skipped because "
                "the aerodynamic series was not generated."
            )

    if args.auto_generate_shedding_spectrum:
        if aero_ready:
            try:
                _, _, r = _resolve_cylinder_geometry(args)
                char_length = 2.0 * r if args.cylinder else 1.0
                plot_shedding_spectrum(
                    aero_series_path,
                    save_name="shedding_spectrum.png",
                    t_min=args.auto_aero_t_min,
                    f_min=0.05,
                    f_max=2.0,
                    char_length=char_length,
                    u_ref=args.inflow_u,
                )
            except Exception as exc:
                print(
                    f"  Warning: automatic shedding-spectrum plot failed: {exc}"
                )
        else:
            print(
                "  Warning: automatic shedding-spectrum plot skipped because "
                "the aerodynamic series was not generated."
            )

    latest_snapshot = None
    if (
        args.auto_generate_ibm_forcing
        or args.auto_generate_vorticity_video
        or args.auto_generate_pressure_coefficient_theta
    ):
        latest_snapshot = find_latest_snapshot(dirpath=args.outdir)
        if latest_snapshot is None:
            print(
                "  Warning: automatic snapshot plots skipped because no snapshots were found.")
            return

    if args.auto_generate_ibm_forcing:
        try:
            plot_ibm_forcing(latest_snapshot, save_name="ibm_forcing.png")
        except Exception as exc:
            print(f"  Warning: automatic IBM-forcing plot failed: {exc}")

    if args.auto_generate_pressure_coefficient_theta:
        try:
            cx, cy, r = _resolve_cylinder_geometry(args)
            save_pressure_coefficient_report(
                latest_snapshot,
                u_ref=args.inflow_u,
                save_csv="pressure_coefficient_theta.csv",
                save_plot="pressure_coefficient_theta.png",
                config_path=args.config,
                cylinder_center=(cx, cy),
                cylinder_radius=r,
            )
        except Exception as exc:
            print(
                f"  Warning: automatic pressure-coefficient report failed: {exc}"
            )

    if args.auto_generate_vorticity_video:
        try:
            plot_vorticity_video(
                snapshot_dir=args.outdir,
                save_name="vorticity.gif",
                frame_stride=max(int(args.auto_vorticity_video_frame_stride), 1),
                verbose=bool(args.verbose),
                config_path=args.config,
                draw_cylinder_overlay=bool(args.draw_cylinder_overlay),
            )
        except Exception as exc:
            print(f"  Warning: automatic vorticity video failed: {exc}")

    if args.auto_generate_time_averaged_fields or args.auto_generate_time_averaged_plots:
        try:
            averaged_path = save_time_averaged_fields(
                indir=args.outdir,
                t_min=args.auto_aero_t_min,
                results_dir=results_dir,
                save_name="time_averaged_fields.npz",
            )
            print(
                f"Saved time-averaged fields: "
                f"{os.path.join(results_dir, 'time_averaged_fields.npz')}"
            )
            if args.auto_generate_time_averaged_plots:
                plot_path = plot_time_averaged_fields(
                    averaged_path,
                    save_name="time_averaged_fields.png",
                    results_dir=results_dir,
                )
                print(f"Saved time-averaged field plot: {plot_path}")
        except Exception as exc:
            print(f"  Warning: automatic time-averaged fields failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    runtime_grid, loaded_from_file = get_runtime_grid(args)
    run(args, grid=runtime_grid, grid_loaded_from_file=loaded_from_file)
