# Coarse tandem-cylinder spacing sweep

Grid: 256 x 512. Configs copy the repository's current Re_D=100, D=1,
t_end=100, save_dt=0.1, stretched grid, and independent free-Y motion for
both cylinders. All other physical parameters are held constant.

Four cases use center-to-center spacings L/D=2, 3, 5, 7. Their surface gaps
are respectively 1D, 2D, 4D, 6D. With the upstream center initially at
(0, 0) and D=1, the downstream centers initially have x=2, 3, 5, 7.

From the repository root, launch every prepared case concurrently:

```powershell
.\cases\tandem_spacing_256x512\run_all.ps1
```

Validate configs without simulating:

```powershell
.\cases\tandem_spacing_256x512\run_all.ps1 -Check
```

Use `-Jobs 1` for sequential runs, or `-Python 'C:\path\to\python.exe'`
to select an interpreter with the repository requirements installed.
The Python equivalent is `python cases/tandem_spacing_256x512/run_all.py`.
For one case, add `--case spacing_3D` to the Python command.

Each case has its own `config.txt` and `experimental_config.txt`. The shared
`post_config.txt` includes the requested aerodynamic output flags exactly;
the existing flow plot and vorticity GIF settings are also retained.

Snapshots and tandem velocity-squared histories go under
`runs/<case>/output/`; plots and aerodynamic reports go under
`runs/<case>/results/`. Console output goes to `runs/<case>/run.log`.
The runner isolates the solver's shared plot paths in separate processes.
Existing snapshot folders are rejected to avoid mixing fresh and old runs.

The automatic aerodynamic coefficients, drag decomposition, and lift-based
shedding spectrum integrate pressure and viscous stress around the primary
(upstream) cylinder, following its moving center. They do not report the
rear cylinder or combined forces. The tandem velocity-squared plot includes
both cylinders separately and their summed squared Y velocities.

These are coarse exploratory runs. No fine-grid campaign is included yet.

Generate mechanical harvested-power graphs from the saved velocity histories:

```powershell
.\.venv\Scripts\python.exe .\cases\tandem_spacing_256x512\generate_power_graphs.py
```

Power is `P=c*y_dot^2` with the saved Y damping coefficient (c=2 for these
runs). Each case's `results/harvested_power.png` and `harvested_power.csv`
show upstream, downstream, and total power in solver units. This is mechanical
damper power before any conversion efficiency. The four-panel overview and
time-weighted mean powers over the full saved interval are in `power_graphs/`.
The overview uses independent vertical scales, as stated in its title.

`auto_include_harvested_power = true` in `post_config.txt` adds a harvested
power section to `aero_report.txt` when `auto_generate_aero_report = true`.
It gives mean power, sampled peak power, and integrated energy for each
cylinder and their combined total, over the full available motion history
(including startup, independent of aerodynamic coefficient trimming).
Every-step velocity CSVs are preferred; if monitoring was disabled, saved
snapshot velocities are used with an explicit accuracy note. Values are in
solver units and represent mechanical free-Y damper extraction.
