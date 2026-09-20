# NS_Solver

`NS_Solver` is a 2-D incompressible Navier-Stokes solver for box-domain flow. It uses a staggered MAC grid, configurable boundary conditions, optional immersed-boundary cylinder geometry, nonuniform center-focused meshes, MPI decomposition in `y`, and lightweight post-processing helpers.

## Overview

- Solves incompressible flow with a fractional-step / projection method
- Uses SSP-RK3 time integration
- Uses a staggered Cartesian finite-volume layout
- Supports configurable `inflow`, `farfield`, `outflow`, `wall`, and `periodic` boundaries
- Supports optional immersed-boundary cylinder cases
- Supports uniform and nonuniform 2-D grids
- Can generate grid-spacing plots, coefficient-history plots, and aerodynamic reports after a run

Additional notes live in [docs/README.md](docs/README.md).

## Repo Files

- [main.py](/Users/Carolyn/Desktop/NS_Solver_Claude/main.py): main solver entry point
- [config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/config.txt): run configuration
- [experimental_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/experimental_config.txt): experimental cylinder configuration
- [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt): post-processing configuration
- [pre_generate_grid.py](/Users/Carolyn/Desktop/NS_Solver_Claude/pre_generate_grid.py): standalone prepared-grid generator
- [analyze_aerodynamics.py](/Users/Carolyn/Desktop/NS_Solver_Claude/analyze_aerodynamics.py): aerodynamic coefficient and Strouhal-style analysis
- [time_average_snapshots.py](/Users/Carolyn/Desktop/NS_Solver_Claude/time_average_snapshots.py): time-averaged mean and RMS field extraction from snapshots
- [view_snapshot_viewer.py](/Users/Carolyn/Desktop/NS_Solver_Claude/view_snapshot_viewer.py): snapshot and coefficient-history plotting

## Quick Start

Typical setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Use explicit run, experimental, and post configs:

```bash
python main.py --config config.txt --experiment-config experimental_config.txt --post-config post_config.txt
```

Run in parallel:

```bash
mpirun -n 4 python main.py
```

## Setup

Requirements:

- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `pytest`
- `mpi4py` if you want MPI runs

## Config Files

The project now uses three config files on purpose:

- [config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/config.txt): how the simulation runs
- [experimental_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/experimental_config.txt): optional experimental cylinder controls
- [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt): which derived plots/reports get generated after the run

Examples are provided in [config_example.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/docs/examples/config_example.txt), [experimental_config_example.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/docs/examples/experimental_config_example.txt), and [post_config_example.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/docs/examples/post_config_example.txt).

### Run Config

Core grid and physics:

- `nx`, `ny`: cell counts
- `lx`, `ly`: domain size
- `re`: Reynolds number
- `t_end`: end time
- `cfl`: target CFL
- `save_dt`: snapshot interval
- `outdir`: snapshot/grid output directory

Grid control:

- `uniform_grid`: `true` for uniform, `false` for nonuniform
- `grid_beta_x`, `grid_beta_y`: tanh stretch strength in each direction for nonuniform grids
- `grid_uniform_x_start`, `grid_uniform_x_end`: explicit x-bounds of the uniform core for nonuniform grids
- `grid_uniform_y_start`, `grid_uniform_y_end`: explicit y-bounds of the uniform core for nonuniform grids

Boundary conditions:

- `bc_left`, `bc_right`, `bc_bottom`, `bc_top`
- `inflow_u`, `inflow_v`, `inflow_w`
- `wall_slip_mode`
- `wall_penetration`
- `wall_normal_velocity`
- `outflow_mode`
- `outflow_speed`

Cylinder / immersed boundary:

- `cylinder`
- `cylinder_center_x`, `cylinder_center_y`
- `cylinder_radius`
- `re_is_cylinder_based`
- `cylinder_rotation_mode`
- `cylinder_rotation_amplitude`
- `cylinder_rotation_frequency`
- `cylinder_rotation_phase_deg`
- `cylinder_translation_mode`
- `cylinder_translation_amplitude_percent`
- `cylinder_translation_x_percent`, `cylinder_translation_y_percent`
- `cylinder_translation_frequency`
- `cylinder_translation_phase_deg`

Initialization and runtime:

- `initial_v_perturbation_percent`: one-time startup perturbation applied to interior `v` as a percent of `inflow_u`
- `verbose`: print run diagnostics

### Experimental Config

- `enable_experimental_config`: master switch; when `false`, all other options in `experimental_config.txt` are ignored
- `truncate_y_cells`: drop this many cells from both the bottom and top y boundaries before running
- `cylinder_experiment`: body shape selector, one of `circle`, `top-indent`, `square`, `airfoil`, or `polygon`
- `polygon_points`, `polygon_angle_deg`: custom body-local vertices and initial orientation when `cylinder_experiment = polygon`
- `cylinder_free_x_dof`, `cylinder_free_y_dof`, `cylinder_free_theta_dof`: enable streamwise, transverse, and/or angular spring-mass-damper motion for the selected bluff body
- `cylinder_free_x_mass`, `cylinder_free_y_mass`, `cylinder_free_x_damping`, `cylinder_free_y_damping`, `cylinder_free_x_stiffness`, `cylinder_free_y_stiffness`: spring-mass-damper parameters for the free body
- `cylinder_free_x_initial_velocity`, `cylinder_free_y_initial_velocity`: initial streamwise/transverse cylinder velocity
- `cylinder_free_x_max_displacement_percent`, `cylinder_free_y_max_displacement_percent`: displacement clamp as percent of body diameter; set to `off` to disable the clamp
- `cylinder_free_x_release_time`, `cylinder_free_y_release_time`: hold the corresponding free axis fixed until this simulation time; `0` releases immediately
- `cylinder_free_theta_inertia`, `cylinder_free_theta_damping`, `cylinder_free_theta_stiffness`: torsional oscillator parameters for angular body motion
- `cylinder_indent_width`, `cylinder_indent_depth`

### Post Config

- `plot`: save the standard end-of-run result figure
- `auto_generate_grid_spacing`: automatically generate `results/grid.png`
- `auto_generate_coeff_history`: automatically generate `results/coeff_history.png`
- `auto_generate_aero_report`: automatically generate `results/aero_report.txt`
- `auto_generate_shedding_spectrum`: automatically generate `results/shedding_spectrum.png`
- `auto_generate_y_oscillation_strouhal`: automatically generate `results/y_oscillation_strouhal.png`
- `auto_generate_drag_decomposition`: automatically generate `results/drag_decomposition.csv`
- `auto_generate_drag_decomposition_plot`: automatically generate `results/drag_decomposition.png`
- `auto_generate_pressure_coefficient_theta`: automatically generate `results/pressure_coefficient_theta.csv` and `results/pressure_coefficient_theta.png`
- `auto_generate_time_averaged_fields`: automatically generate `results/time_averaged_fields.npz`
- `auto_generate_time_averaged_plots`: automatically generate `results/time_averaged_fields.png`
- `auto_generate_ibm_forcing`: automatically generate `results/ibm_forcing.png` from the latest snapshot
- `auto_generate_vorticity_video`: automatically generate `results/vorticity.gif` from all saved snapshots
- `auto_vorticity_video_frame_stride`: use every `n`th snapshot when building the vorticity GIF
- `draw_cylinder_overlay`: draw the cylinder/body outline on generated flow plots and GIFs
- `surface_force_sample_offset_factor`: near-cylinder offset used by surface-stress aerodynamic post-processing
- `auto_coeff_t_min`: minimum time for automatic coefficient-history plotting
- `auto_aero_t_min`: minimum time for automatic aerodynamic analysis

## Runtime Notes

### Initial Perturbation

`initial_v_perturbation_percent` is easy to miss but useful for wake development studies and vortex-shedding startup tests.

If:

- `inflow_u = 1.0`
- `initial_v_perturbation_percent = 2.0`

then the solver applies a one-time interior perturbation of:

```text
dv = 0.02 * inflow_u = 0.02
```

This does not permanently change the configured boundary condition values. It only seeds the initial interior field.

### Nonuniform Grid

The current nonuniform grid uses a uniform central core with tanh-stretched outer regions.

- `beta_x` and `beta_y` control how strongly the outer regions stretch away from the core
- `grid_uniform_x_start` and `grid_uniform_x_end` define the flat-spacing core in `x`
- `grid_uniform_y_start` and `grid_uniform_y_end` define the flat-spacing core in `y`
- When the `y` domain is symmetric about `0`, the nonuniform `y` core must also be symmetric

Prepared grid metadata is saved automatically into `outdir` as either:

- `uniform_grid.npz`
- `nonuniform_grid.npz`

## Prepared Grid Generation

Generate prepared grid metadata without running the full solve:

```bash
python pre_generate_grid.py
```

Generate a nonuniform prepared grid:

```bash
python pre_generate_grid.py --grid-type nonuniform --beta-x 2.5 --beta-y 2.0 --uniform-x-start 6.0 --uniform-x-end 10.0 --uniform-y-start 3.5 --uniform-y-end 6.5
```

Common options:

- `--nx`, `--ny`, `--lx`, `--ly`
- `--grid-type uniform|nonuniform`
- `--beta-x`, `--beta-y`
- `--uniform-x-start`, `--uniform-x-end`
- `--uniform-y-start`, `--uniform-y-end`
- `--outdir`
- `--output-name`

## Outputs

Common outputs:

- `output/snap_*.npz`: solution snapshots
- `output/uniform_grid.npz` or `output/nonuniform_grid.npz`: prepared grid metadata
- `results/result.png`: standard flow plot from `main.py` when `plot = true` in `post_config.txt`
- `results/grid.png`: physical grid / spacing plot
- `results/aero.csv`: coefficient and force history
- `results/drag_decomposition.csv`: pressure and viscous force/coefficient history
- `results/drag_decomposition.png`: pressure and viscous force/coefficient plot
- `results/coeff_history.png`: drag/lift history figure
- `results/aero_report.txt`: aerodynamic summary report

## Post-Processing

### Grid Spacing Plot

Enable in [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt):

```text
auto_generate_grid_spacing = true
auto_generate_vorticity_video = true
auto_vorticity_video_frame_stride = 5
```

This produces `results/grid.png` with:

- the physical grid
- point concentration / cell density
- spacing curves

### Coefficient History Plot

Generate it manually from the viewer:

```bash
python view_snapshot_viewer.py --plot-coeffs --save coeff_history.png
```

Useful viewer options:

- `--coeff-file`
- `--coeff-indir`
- `--coeff-t-min`
- `--save`

### IBM Forcing Plot

Generate a one-command IBM forcing visualization (x-component, y-component, magnitude):

```bash
python view_snapshot_viewer.py --plot-ibm --save ibm_forcing.png
```

Or for a specific snapshot:

```bash
python view_snapshot_viewer.py output/snap_005.0000.npz --plot-ibm --save ibm_forcing.png
```

### Aerodynamic Analysis

Run directly:

```bash
python analyze_aerodynamics.py --indir output --config config.txt --use-cylinder-diameter --t-min 1.0 --save-series results/aero.csv --save-report results/aero_report.txt
```

This script computes:

- force histories
- `C_d` and `C_l`
- a dominant lift frequency / Strouhal-style estimate
- summary statistics in a text report

Useful analysis options:

- `--indir`
- `--pattern`
- `--config`
- `--probe-x`, `--probe-y`
- `--u-ref`
- `--length-scale`
- `--use-cylinder-diameter`
- `--cylinder-radius`
- `--t-min`
- `--f-min`, `--f-max`
- `--save-series`
- `--save-report`

### Time-Averaged Fields

Run directly:

```bash
python time_average_snapshots.py --indir output --t-min 1.0 --save-name time_averaged_fields.npz --plot
```

This script saves:

- `u_mean`, `v_mean`, `p_mean`
- `u_rms`, `v_rms`
- `xc`, `yc`
- `t_start`, `t_end`, `n_snapshots`, `averaging_duration`

## Snapshot Viewer

Examples:

```bash
python view_snapshot_viewer.py --list
python view_snapshot_viewer.py -k p --save pressure.png
python view_snapshot_viewer.py output/snap_005.0000.npz -k u --save velocity.png
```

Useful viewer options:

- `-k`, `--key`
- `-s`, `--slice`
- `-c`, `--comp`
- `--save`
- `--x-scale`
- `--y-scale`
- `--plot-coeffs`

## Notes

- The README now reflects the current split-config workflow
- There is no `evaluate_strouhal.py` in this repo; use [analyze_aerodynamics.py](/Users/Carolyn/Desktop/NS_Solver_Claude/analyze_aerodynamics.py) for current frequency/report analysis
