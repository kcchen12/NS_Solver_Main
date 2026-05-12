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
    save_grid_metadata,
    save_grid_metadata_dict,
    load_prepared_grid,
    load_grid_metadata_dict,
)
from src.parallel import ParallelDecomposition
from src.config import ConfigParser
from analyze_aerodynamics import (
    run_analysis as run_aero_analysis,
    plot_shedding_spectrum,
    save_pressure_coefficient_report,
)
from time_average_snapshots import save_time_averaged_fields
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


def _normalize_ibm_shape(raw_value: str | None) -> str:
    value = "circle" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "cylinder": "circle",
        "circle_top_indent": "circle-with-top-indent",
        "circle-with-indent": "circle-with-top-indent",
        "indented-circle": "circle-with-top-indent",
    }
    value = aliases.get(value, value)
    return value if value in {"circle", "circle-with-top-indent"} else "circle"


def _normalize_cylinder_geometry_mode(raw_value: str | None) -> str:
    value = "circle" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "ibm-circle": "circle",
        "indented-circle": "circle-with-top-indent",
        "rectangular-top-indent": "circle-with-top-indent",
    }
    return _normalize_ibm_shape(aliases.get(value, value))


def _normalize_cylinder_actuation_mode(raw_value: str | None) -> str:
    value = "none" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "off": "none",
        "sweeping_jet": "sweeping-jet",
        "jet": "sweeping-jet",
        "geometry_resolved_sweeping_jet": "geometry-resolved-sweeping-jet",
        "resolved-jet": "geometry-resolved-sweeping-jet",
    }
    value = aliases.get(value, value)
    return value if value in {"none", "sweeping-jet", "geometry-resolved-sweeping-jet"} else "none"


def _normalize_cylinder_experiment_mode(raw_value: str | None) -> str:
    value = "none" if raw_value is None else str(raw_value).strip().lower()
    aliases = {
        "off": "none",
        "indent": "top-indent",
        "rectangular-indent": "top-indent",
        "simulated-jet": "simulated-sweeping-jet",
        "simulated_sweeping_jet": "simulated-sweeping-jet",
        "geometry-resolved-jet": "geometry-resolved-sweeping-jet",
        "geometry_resolved_sweeping_jet": "geometry-resolved-sweeping-jet",
    }
    value = aliases.get(value, value)
    valid = {
        "none",
        "top-indent",
        "simulated-sweeping-jet",
        "geometry-resolved-sweeping-jet",
    }
    return value if value in valid else "none"


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
    p.add_argument("--cylinder-experiment", type=str,
                   choices=[
                       "none",
                       "top-indent",
                       "simulated-sweeping-jet",
                       "geometry-resolved-sweeping-jet",
                   ],
                   default=_normalize_cylinder_experiment_mode(
                       exp_cfg.get("cylinder_experiment", "none", str)),
                   help="High-level experimental cylinder mode")
    p.add_argument("--cylinder-geometry-mode", type=str,
                   choices=["circle", "circle-with-top-indent"],
                   default=_normalize_cylinder_geometry_mode(
                       exp_cfg.get(
                           "cylinder_geometry_mode",
                           exp_cfg.get("ibm_shape", "circle", str),
                           str,
                       )),
                   help="Cylinder geometry mode")
    p.add_argument("--ibm-shape", type=str,
                   choices=["circle", "circle-with-top-indent"],
                   default=_normalize_cylinder_geometry_mode(
                       exp_cfg.get(
                           "cylinder_geometry_mode",
                           exp_cfg.get("ibm_shape", "circle", str),
                           str,
                       )),
                   help="Immersed-body shape")
    p.add_argument("--cylinder-indent-width", type=float,
                   default=exp_cfg.get("cylinder_indent_width", 0.0, float),
                   help="Width of the rectangular top indent for circle-with-top-indent")
    p.add_argument("--cylinder-indent-depth", type=float,
                   default=exp_cfg.get("cylinder_indent_depth", 0.0, float),
                   help="Depth of the rectangular top indent for circle-with-top-indent")
    p.add_argument("--cylinder-actuation-mode", type=str,
                   choices=["none", "sweeping-jet", "geometry-resolved-sweeping-jet"],
                   default=_normalize_cylinder_actuation_mode(
                       exp_cfg.get(
                           "cylinder_actuation_mode",
                           exp_cfg.get("actuator_model", "none", str),
                           str,
                       )),
                   help="Optional finite jet actuation model on the cylinder surface")
    p.add_argument("--sweeping-jet-velocity-ratio", type=float,
                   default=exp_cfg.get("sweeping_jet_velocity_ratio", 0.25, float),
                   help="Jet speed magnitude relative to inflow_u")
    p.add_argument("--sweeping-jet-frequency", type=float,
                   default=exp_cfg.get("sweeping_jet_frequency", -1.0, float),
                   help="Jet sweep frequency; <=0 uses a shedding-scale default")
    p.add_argument("--sweeping-jet-center-deg", type=float,
                   default=exp_cfg.get("sweeping_jet_center_deg", 90.0, float),
                   help="Angular location of the jet outlet on the cylinder surface")
    p.add_argument("--sweeping-jet-slot-width-deg", type=float,
                   default=exp_cfg.get("sweeping_jet_slot_width_deg", 18.0, float),
                   help="Angular width of the finite jet outlet")
    p.add_argument("--sweeping-jet-slot-depth", type=float,
                   default=exp_cfg.get("sweeping_jet_slot_depth", 0.0, float),
                   help="Radial depth of the finite jet outlet band inside the IBM body")
    p.add_argument("--sweeping-jet-angle-deg", type=float,
                   default=exp_cfg.get("sweeping_jet_angle_deg", 25.0, float),
                   help="Sweep amplitude of the jet direction relative to the local normal")
    p.add_argument("--sweeping-jet-phase-deg", type=float,
                   default=exp_cfg.get("sweeping_jet_phase_deg", 0.0, float),
                   help="Phase offset for the sweeping jet direction oscillation")
    p.add_argument("--resolved-jet-cavity-width", type=float,
                   default=exp_cfg.get("resolved_jet_cavity_width", 0.0, float),
                   help="Width of the internal plenum for geometry-resolved jet mode")
    p.add_argument("--resolved-jet-cavity-height", type=float,
                   default=exp_cfg.get("resolved_jet_cavity_height", 0.0, float),
                   help="Height of the internal plenum for geometry-resolved jet mode")
    p.add_argument("--resolved-jet-slot-width", type=float,
                   default=exp_cfg.get("resolved_jet_slot_width", 0.0, float),
                   help="Width of the exit slot for geometry-resolved jet mode")
    p.add_argument("--resolved-jet-slot-height", type=float,
                   default=exp_cfg.get("resolved_jet_slot_height", 0.0, float),
                   help="Height of the exit slot for geometry-resolved jet mode")
    p.add_argument("--resolved-jet-feed-width", type=float,
                   default=exp_cfg.get("resolved_jet_feed_width", 0.0, float),
                   help="Width of the internal forcing/feed patch for geometry-resolved jet mode")
    p.add_argument("--resolved-jet-feed-height", type=float,
                   default=exp_cfg.get("resolved_jet_feed_height", 0.0, float),
                   help="Height of the internal forcing/feed patch for geometry-resolved jet mode")
    p.add_argument("--resolved-jet-nozzle-length", type=float,
                   default=exp_cfg.get("resolved_jet_nozzle_length", 0.0, float),
                   help="Length of the tapered nozzle section between plenum and slot")
    p.add_argument("--resolved-jet-slot-exit-width", type=float,
                   default=exp_cfg.get("resolved_jet_slot_exit_width", 0.0, float),
                   help="Width of the internal exit wedge where the slot meets the cylinder wall")
    p.add_argument("--resolved-jet-island-wall-gap", type=float,
                   default=exp_cfg.get("resolved_jet_island_wall_gap", 0.0, float),
                   help="Gap between each floating island and the outer plenum wall")
    p.add_argument("--resolved-jet-island-center-gap", type=float,
                   default=exp_cfg.get("resolved_jet_island_center_gap", 0.0, float),
                   help="Gap between the two floating islands across the center channel")
    p.add_argument("--resolved-jet-island-leading-gap", type=float,
                   default=exp_cfg.get("resolved_jet_island_leading_gap", 0.0, float),
                   help="Gap from the back of the plenum to the start of each floating island")
    p.add_argument("--resolved-jet-island-trailing-gap", type=float,
                   default=exp_cfg.get("resolved_jet_island_trailing_gap", 0.0, float),
                   help="Gap from the front of each floating island to the nozzle throat")
    p.add_argument("--resolved-jet-island-taper", type=float,
                   default=exp_cfg.get("resolved_jet_island_taper", 0.0, float),
                   help="Taper amount applied to the front face of each floating island")
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
    p.add_argument("--plot",     type=str_to_bool, default=post_cfg.get("plot", False, bool),
                   help="Save the standard end-of-run result figure")
    p.add_argument("--plot-grid", type=str_to_bool,
                   default=post_cfg.get("plot_grid", False, bool),
                   help="Save a physical grid plot showing mesh concentration")
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
    p.add_argument("--auto-generate-pressure-coefficient-theta", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_pressure_coefficient_theta", False, bool),
                   help="Automatically save C_p(theta) CSV and PNG from the latest snapshot")
    p.add_argument("--auto-generate-time-averaged-fields", type=str_to_bool,
                   default=post_cfg.get(
                       "auto_generate_time_averaged_fields", False, bool),
                   help="Automatically save time-averaged mean/RMS fields from output snapshots")
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
    p.add_argument("--initial-v-perturbation-pct", type=float,
                   default=cfg.get("initial_v_perturbation_pct", 0.0, float),
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
                   help="Minimum time used when auto-generating aerodynamic analysis")
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


def _resolve_cylinder_geometry(args) -> tuple[float, float, float]:
    cx = args.cylinder_center_x if args.cylinder_center_x >= 0.0 else args.x_min + 0.25 * args.lx
    cy = args.cylinder_center_y if args.cylinder_center_y >= 0.0 else args.y_min + 0.5 * args.ly
    radius = args.cylinder_radius if args.cylinder_radius > 0.0 else args.ly / 8.0
    return cx, cy, radius


def _resolve_indent_geometry(args, radius: float) -> tuple[float, float]:
    indent_width = float(args.cylinder_indent_width)
    indent_depth = float(args.cylinder_indent_depth)
    if indent_width <= 0.0:
        indent_width = 0.6 * radius
    if indent_depth <= 0.0:
        indent_depth = 0.35 * radius
    return indent_width, indent_depth


def _resolve_sweeping_jet_geometry(args, radius: float, inflow_u: float) -> dict:
    velocity_ratio = max(float(args.sweeping_jet_velocity_ratio), 0.0)
    jet_speed = velocity_ratio * abs(float(inflow_u))
    slot_depth = float(args.sweeping_jet_slot_depth)
    if slot_depth <= 0.0:
        slot_depth = 0.15 * radius

    frequency = float(args.sweeping_jet_frequency)
    if frequency <= 0.0:
        diameter = 2.0 * radius
        frequency = 0.16 * abs(float(inflow_u)) / diameter if diameter > 0.0 else 0.0

    return {
        "jet_speed": jet_speed,
        "frequency": frequency,
        "center_deg": float(args.sweeping_jet_center_deg),
        "slot_width_deg": float(args.sweeping_jet_slot_width_deg),
        "slot_depth": slot_depth,
        "angle_deg": float(args.sweeping_jet_angle_deg),
        "phase_rad": np.deg2rad(float(args.sweeping_jet_phase_deg)),
    }


def _resolve_geometry_resolved_jet_geometry(args, radius: float, inflow_u: float) -> dict:
    jet_cfg = _resolve_sweeping_jet_geometry(args, radius, inflow_u)

    cavity_width = float(args.resolved_jet_cavity_width)
    cavity_height = float(args.resolved_jet_cavity_height)
    slot_width = float(args.resolved_jet_slot_width)
    slot_height = float(args.resolved_jet_slot_height)
    feed_width = float(args.resolved_jet_feed_width)
    feed_height = float(args.resolved_jet_feed_height)
    nozzle_length = float(args.resolved_jet_nozzle_length)
    slot_exit_width = float(args.resolved_jet_slot_exit_width)
    island_wall_gap = float(args.resolved_jet_island_wall_gap)
    island_center_gap = float(args.resolved_jet_island_center_gap)
    island_leading_gap = float(args.resolved_jet_island_leading_gap)
    island_trailing_gap = float(args.resolved_jet_island_trailing_gap)
    island_taper = float(args.resolved_jet_island_taper)

    if cavity_width <= 0.0:
        cavity_width = 0.9 * radius
    if cavity_height <= 0.0:
        cavity_height = 0.75 * radius
    if slot_width <= 0.0:
        slot_width = 0.24 * radius
    if slot_height <= 0.0:
        slot_height = 0.18 * radius
    if feed_width <= 0.0:
        feed_width = 0.45 * radius
    if feed_height <= 0.0:
        feed_height = 0.22 * radius
    if nozzle_length <= 0.0:
        nozzle_length = min(
            max(1.5 * slot_height, 0.35 * cavity_height),
            0.65 * cavity_height,
        )
    plenum_depth = max(cavity_height - nozzle_length, 1e-12)
    if slot_exit_width <= 0.0:
        slot_exit_width = min(cavity_width, slot_width + 2.0 * slot_height)
    if island_wall_gap <= 0.0:
        island_wall_gap = 0.12 * cavity_width
    if island_center_gap <= 0.0:
        island_center_gap = max(0.22 * cavity_width, feed_width + 0.08 * cavity_width)
    if island_leading_gap <= 0.0:
        island_leading_gap = 0.18 * plenum_depth
    if island_trailing_gap <= 0.0:
        island_trailing_gap = 0.18 * plenum_depth

    island_height = 0.5 * max(
        cavity_width - 2.0 * island_wall_gap - island_center_gap,
        0.18 * cavity_width,
    )
    if island_taper <= 0.0:
        island_taper = 0.18 * island_height

    jet_cfg.update({
        "cavity_width": cavity_width,
        "cavity_height": cavity_height,
        "slot_width": slot_width,
        "slot_height": slot_height,
        "feed_width": feed_width,
        "feed_height": feed_height,
        "nozzle_length": nozzle_length,
        "slot_exit_width": slot_exit_width,
        "island_wall_gap": island_wall_gap,
        "island_center_gap": island_center_gap,
        "island_leading_gap": island_leading_gap,
        "island_trailing_gap": island_trailing_gap,
        "island_taper": island_taper,
    })
    return jet_cfg


def _resolve_experiment_overrides(args) -> tuple[str, str]:
    experiment = _normalize_cylinder_experiment_mode(
        getattr(args, "cylinder_experiment", "none")
    )
    if experiment == "top-indent":
        return "circle-with-top-indent", "none"
    if experiment == "simulated-sweeping-jet":
        return "circle", "sweeping-jet"
    if experiment == "geometry-resolved-sweeping-jet":
        return "circle", "geometry-resolved-sweeping-jet"

    shape = _normalize_cylinder_geometry_mode(
        getattr(args, "cylinder_geometry_mode", getattr(args, "ibm_shape", "circle"))
    )
    actuation = _normalize_cylinder_actuation_mode(
        getattr(args, "cylinder_actuation_mode", "none")
    )
    return shape, actuation


def _plot_ibm_outline(ax, args, color: str = "white", linewidth: float = 1.6) -> None:
    def local_box_points(
        center_x: float,
        center_y: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        s0: float,
        s1: float,
        n0: float,
        n1: float,
    ) -> tuple[list[float], list[float]]:
        corners = [
            (s0, n0),
            (s1, n0),
            (s1, n1),
            (s0, n1),
            (s0, n0),
        ]
        x_pts = [
            center_x + s * tangent_x + n * normal_x
            for s, n in corners
        ]
        y_pts = [
            center_y + s * tangent_y + n * normal_y
            for s, n in corners
        ]
        return x_pts, y_pts

    def local_poly_points(
        center_x: float,
        center_y: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        corners: list[tuple[float, float]],
    ) -> tuple[list[float], list[float]]:
        x_pts = [
            center_x + s * tangent_x + n * normal_x
            for s, n in corners
        ]
        y_pts = [
            center_y + s * tangent_y + n * normal_y
            for s, n in corners
        ]
        return x_pts, y_pts

    def local_curve_points(
        center_x: float,
        center_y: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        s_vals: np.ndarray,
        n_vals: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        x_pts = [
            center_x + s * tangent_x + n * normal_x
            for s, n in zip(s_vals, n_vals)
        ]
        y_pts = [
            center_y + s * tangent_y + n * normal_y
            for s, n in zip(s_vals, n_vals)
        ]
        return x_pts, y_pts

    cx, cy, radius = _resolve_cylinder_geometry(args)
    shape, actuation_mode = _resolve_experiment_overrides(args)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    if shape == "circle":
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=6)
        if actuation_mode == "geometry-resolved-sweeping-jet":
            jet_cfg = _resolve_geometry_resolved_jet_geometry(args, radius, 1.0)
            slot_center_angle = np.deg2rad(jet_cfg["center_deg"])
            normal_x = float(np.cos(slot_center_angle))
            normal_y = float(np.sin(slot_center_angle))
            tangent_x = float(-np.sin(slot_center_angle))
            tangent_y = float(np.cos(slot_center_angle))
            nozzle_length = jet_cfg["nozzle_length"]
            plenum_n1 = radius - jet_cfg["slot_height"] - nozzle_length
            plenum_depth = plenum_n1 - (radius - jet_cfg["slot_height"] - jet_cfg["cavity_height"])
            cavity_x, cavity_y = local_poly_points(
                cx, cy,
                tangent_x, tangent_y,
                normal_x, normal_y,
                [
                    (-0.5 * jet_cfg["cavity_width"], radius - jet_cfg["slot_height"] - jet_cfg["cavity_height"]),
                    (0.5 * jet_cfg["cavity_width"], radius - jet_cfg["slot_height"] - jet_cfg["cavity_height"]),
                    (0.5 * jet_cfg["cavity_width"], plenum_n1),
                    (0.5 * jet_cfg["slot_width"], radius - jet_cfg["slot_height"]),
                    (-0.5 * jet_cfg["slot_width"], radius - jet_cfg["slot_height"]),
                    (-0.5 * jet_cfg["cavity_width"], plenum_n1),
                    (-0.5 * jet_cfg["cavity_width"], radius - jet_cfg["slot_height"] - jet_cfg["cavity_height"]),
                ],
            )
            center_gap_s = jet_cfg["island_center_gap"]
            island_height = 0.5 * max(
                jet_cfg["cavity_width"] - 2.0 * jet_cfg["island_wall_gap"] - center_gap_s,
                0.18 * jet_cfg["cavity_width"],
            )
            island_n0 = (
                radius - jet_cfg["slot_height"] - jet_cfg["cavity_height"]
                + jet_cfg["island_leading_gap"]
            )
            island_n1 = plenum_n1 - jet_cfg["island_trailing_gap"]
            island_taper = jet_cfg["island_taper"] if jet_cfg["island_taper"] > 0.0 else 0.18 * island_height
            upper_s0 = 0.5 * center_gap_s
            upper_s1 = upper_s0 + island_height
            lower_s1 = -0.5 * center_gap_s
            lower_s0 = lower_s1 - island_height
            upper_splitter_x, upper_splitter_y = local_poly_points(
                cx, cy,
                tangent_x, tangent_y,
                normal_x, normal_y,
                [
                    (upper_s0, island_n0),
                    (upper_s1, island_n0),
                    (upper_s1 - island_taper, island_n1),
                    (upper_s0 + island_taper, island_n1),
                    (upper_s0, island_n0),
                ],
            )
            lower_splitter_x, lower_splitter_y = local_poly_points(
                cx, cy,
                tangent_x, tangent_y,
                normal_x, normal_y,
                [
                    (lower_s0, island_n0),
                    (lower_s1, island_n0),
                    (lower_s1 - island_taper, island_n1),
                    (lower_s0 + island_taper, island_n1),
                    (lower_s0, island_n0),
                ],
            )
            slot_exit_width = jet_cfg["slot_exit_width"]
            slot_n0 = radius - jet_cfg["slot_height"]
            slot_half_width0 = 0.5 * jet_cfg["slot_width"]
            slot_half_width1 = 0.5 * slot_exit_width

            def slot_minus_circle(n_val: float) -> float:
                alpha = (n_val - slot_n0) / max(jet_cfg["slot_height"], 1e-12)
                half_width = (1.0 - alpha) * slot_half_width0 + alpha * slot_half_width1
                return half_width - np.sqrt(max(radius ** 2 - n_val ** 2, 0.0))

            n_lo = slot_n0
            n_hi = radius
            for _ in range(50):
                n_mid = 0.5 * (n_lo + n_hi)
                if slot_minus_circle(n_mid) > 0.0:
                    n_hi = n_mid
                else:
                    n_lo = n_mid
            n_int = 0.5 * (n_lo + n_hi)
            s_int = np.sqrt(max(radius ** 2 - n_int ** 2, 0.0))

            theta_arc = np.linspace(
                np.arctan2(-s_int, n_int),
                np.arctan2(s_int, n_int),
                80,
            )
            arc_s = radius * np.sin(theta_arc)
            arc_n = radius * np.cos(theta_arc)
            slot_s = np.concatenate((
                np.array([-slot_half_width0, slot_half_width0, s_int]),
                arc_s[::-1],
                np.array([-s_int, -slot_half_width0]),
            ))
            slot_n = np.concatenate((
                np.array([slot_n0, slot_n0, n_int]),
                arc_n[::-1],
                np.array([n_int, slot_n0]),
            ))
            slot_x, slot_y = local_curve_points(
                cx, cy,
                tangent_x, tangent_y,
                normal_x, normal_y,
                slot_s,
                slot_n,
            )
            ax.plot(
                cavity_x,
                cavity_y,
                color=color,
                linewidth=linewidth * 0.9,
                zorder=6,
            )
            ax.plot(
                upper_splitter_x,
                upper_splitter_y,
                color=color,
                linewidth=linewidth * 0.9,
                zorder=6,
            )
            ax.plot(
                lower_splitter_x,
                lower_splitter_y,
                color=color,
                linewidth=linewidth * 0.9,
                zorder=6,
            )
            ax.plot(
                slot_x,
                slot_y,
                color=color,
                linewidth=linewidth * 0.9,
                zorder=6,
            )
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
    return {
        "t": solver.t,
        "nx": args.nx,
        "ny": args.ny,
        "lx": args.lx,
        "ly": args.ly,
        "re": args.re,
        "ibm_force_x": solver.last_ibm_force_x,
        "ibm_force_y": solver.last_ibm_force_y,
        "cylinder_omega": _cylinder_angular_velocity(args, solver.t),
    }


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
    grid = CartesianGrid(nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
                         x_min=args.x_min, y_min=args.y_min)
    os.makedirs(args.outdir, exist_ok=True)
    save_grid_metadata(_grid_metadata_path(args), grid)
    return grid


def prepare_nonuniform_grid(args):
    """Build the runtime non-uniform grid and write its metadata before startup."""
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
        print(f"  Grid          : {args.nx} x {args.ny}")
        print(f"  Grid type     : {args.grid_type}")
        if args.grid_type == "nonuniform":
            print("  Grid mode     : center-uniform")
        print(
            "  Domain        : "
            f"x=[{args.x_min}, {args.x_max}] (Lx={args.lx}), "
            f"y=[{args.y_min}, {args.y_max}] (Ly={args.ly})"
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
        ibm_shape, actuation_mode = _resolve_experiment_overrides(args)
        rotation_mode = _normalize_cylinder_rotation_mode(args.cylinder_rotation_mode)
        if ibm_shape != "circle" and rotation_mode != "stationary":
            raise ValueError(
                "circle-with-top-indent currently supports stationary IBM bodies only"
            )
        if actuation_mode != "none" and ibm_shape != "circle":
            raise ValueError(
                "jet actuation currently supports circle geometry only"
            )
        if actuation_mode != "none" and rotation_mode != "stationary":
            raise ValueError(
                "jet actuation currently supports stationary cylinders only"
            )
        if rotation_mode == "oscillatory":
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
            else:
                ibm.add_circle(cx, cy, r)
        if actuation_mode == "sweeping-jet":
            jet_cfg = _resolve_sweeping_jet_geometry(args, r, bc.u_inf)
            ibm.add_sweeping_jet_circle(
                cx=cx,
                cy=cy,
                radius=r,
                jet_speed=jet_cfg["jet_speed"],
                slot_center_angle_deg=jet_cfg["center_deg"],
                slot_width_angle_deg=jet_cfg["slot_width_deg"],
                slot_depth=jet_cfg["slot_depth"],
                sweep_amplitude_deg=jet_cfg["angle_deg"],
                frequency=jet_cfg["frequency"],
                phase=jet_cfg["phase_rad"],
            )
        elif actuation_mode == "geometry-resolved-sweeping-jet":
            jet_cfg = _resolve_geometry_resolved_jet_geometry(args, r, bc.u_inf)
            ibm.add_geometry_resolved_sweeping_jet_circle(
                cx=cx,
                cy=cy,
                radius=r,
                jet_speed=jet_cfg["jet_speed"],
                cavity_width=jet_cfg["cavity_width"],
                cavity_height=jet_cfg["cavity_height"],
                slot_width=jet_cfg["slot_width"],
                slot_height=jet_cfg["slot_height"],
                feed_width=jet_cfg["feed_width"],
                feed_height=jet_cfg["feed_height"],
                nozzle_length=jet_cfg["nozzle_length"],
                slot_exit_width=jet_cfg["slot_exit_width"],
                island_wall_gap=jet_cfg["island_wall_gap"],
                island_center_gap=jet_cfg["island_center_gap"],
                island_leading_gap=jet_cfg["island_leading_gap"],
                island_trailing_gap=jet_cfg["island_trailing_gap"],
                island_taper=jet_cfg["island_taper"],
                slot_center_angle_deg=jet_cfg["center_deg"],
                sweep_amplitude_deg=jet_cfg["angle_deg"],
                frequency=jet_cfg["frequency"],
                phase=jet_cfg["phase_rad"],
            )
        if is_root and args.verbose:
            print(
                f"  IBM cylinder: centre=({cx:.2f},{cy:.2f}), r={r:.4f}, "
                f"shape={ibm_shape}, experiment={_normalize_cylinder_experiment_mode(args.cylinder_experiment)}"
            )
            if ibm_shape == "circle-with-top-indent":
                indent_width, indent_depth = _resolve_indent_geometry(args, r)
                print(
                    "  Top indent   : "
                    f"width={indent_width:.4f}, depth={indent_depth:.4f}"
                )
            if actuation_mode == "sweeping-jet":
                jet_cfg = _resolve_sweeping_jet_geometry(args, r, bc.u_inf)
                print(
                    "  Jet actuator : "
                    f"speed={jet_cfg['jet_speed']:.4f}, "
                    f"f={jet_cfg['frequency']:.4f}, "
                    f"slot_center={jet_cfg['center_deg']:.2f} deg, "
                    f"slot_width={jet_cfg['slot_width_deg']:.2f} deg, "
                    f"slot_depth={jet_cfg['slot_depth']:.4f}, "
                    f"sweep={jet_cfg['angle_deg']:.2f} deg"
                )
            elif actuation_mode == "geometry-resolved-sweeping-jet":
                jet_cfg = _resolve_geometry_resolved_jet_geometry(args, r, bc.u_inf)
                print(
                    "  Jet actuator : "
                    f"mode=geometry-resolved, speed={jet_cfg['jet_speed']:.4f}, "
                    f"f={jet_cfg['frequency']:.4f}, "
                    f"cavity=({jet_cfg['cavity_width']:.4f} x {jet_cfg['cavity_height']:.4f}), "
                    f"slot=({jet_cfg['slot_width']:.4f} x {jet_cfg['slot_height']:.4f}), "
                    f"feed=({jet_cfg['feed_width']:.4f} x {jet_cfg['feed_height']:.4f}), "
                    f"sweep={jet_cfg['angle_deg']:.2f} deg"
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

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    nu = _kinematic_viscosity(args, bc.u_inf, r)

    if is_root and args.verbose and args.cylinder and args.re_is_cylinder_based and r is not None:
        d_cyl = 2.0 * r
        print(
            f"  Re interpretation: Re_D={args.re} with D={d_cyl:.4f} -> nu={nu:.6g}")

    initial_v_perturbation = 0.01 * \
        args.initial_v_perturbation_pct * bc.u_inf
    solver = FractionalStepSolver(grid, bc, nu, ibm=ibm)
    solver.init_fields(
        u0=bc.u_inf,
        v0=bc.v_inf,
        initial_v_perturbation=initial_v_perturbation,
    )
    if is_root and args.verbose and args.initial_v_perturbation_pct != 0.0:
        print(
            "  Initial v perturbation: "
            f"{args.initial_v_perturbation_pct:.3g}% of inflow_u "
            f"-> dv={initial_v_perturbation:.6g}"
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
                  f"|∇·u|_max={div_max:.2e}  CFL={cfl_val:.3f}")

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
    if args.plot_grid and is_root:
        _plot_grid(grid, args)
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

    if args.cylinder:
        # Draw the immersed cylinder on every panel so geometry alignment
        # is visible in vorticity, pressure, and velocity plots.
        for ax in axes:
            _plot_ibm_outline(ax, args, color="black", linewidth=1.6)

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

    if args.cylinder:
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
    )
    aero_series_path = os.path.join(results_dir, "aero.csv")
    aero_report_path = os.path.join(results_dir, "aero_report.txt")
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
                save_report=aero_report_path if args.auto_generate_aero_report else None,
            )
            aero_ready = status == 0 and os.path.exists(aero_series_path)
            if status != 0:
                print("  Warning: automatic aerodynamic post-processing failed.")
        except Exception as exc:
            print(
                f"  Warning: automatic aerodynamic post-processing failed: {exc}"
            )

    if args.auto_generate_coeff_history:
        if aero_ready:
            coeff_history_name = "coeff_history.png"
            try:
                plot_coeff_history(
                    aero_series_path,
                    save_name=coeff_history_name,
                    coeff_t_min=args.auto_coeff_t_min,
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
            )
        except Exception as exc:
            print(f"  Warning: automatic vorticity video failed: {exc}")

    if args.auto_generate_time_averaged_fields:
        try:
            save_time_averaged_fields(
                indir=args.outdir,
                t_min=args.auto_aero_t_min,
                results_dir=results_dir,
                save_name="time_averaged_fields.npz",
            )
            print(
                f"Saved time-averaged fields: "
                f"{os.path.join(results_dir, 'time_averaged_fields.npz')}"
            )
        except Exception as exc:
            print(f"  Warning: automatic time-averaged fields failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    runtime_grid, loaded_from_file = get_runtime_grid(args)
    run(args, grid=runtime_grid, grid_loaded_from_file=loaded_from_file)
