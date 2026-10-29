# Re=100 640x1280 Restart Campaign

This folder contains runnable configuration files for four runs:

1. `00_stationary_spinup_t0_t100`: stationary cylinder from `t=0` to `t=100`
2. `01_stationary_from_t100_t225`: stationary cylinder resumed from the spin-up
3. `02_y_oscillation_from_t100_t225`: free transverse cylinder DoF resumed from the spin-up
4. `03_rotation_from_t100_t225`: free angular/theta cylinder DoF resumed from the spin-up

All cases use the same `640 x 1280` grid and `Re_D = 100` with `D = 1`.

## Timing

For a stationary cylinder near `Re=100`, the expected Strouhal number is roughly
`St ~= 0.16`, so the shedding period is approximately:

```text
T_shed ~= D / (St * U) = 1 / 0.16 = 6.25
```

The production cases run from `t=100` to `t=225`, or about 20 shedding periods
after the shared established stationary state. They use `save_dt = 0.25`, which
is about 25 binary `.npz` snapshots per shedding period.

If your measured shedding period differs, edit `t_end` and/or `save_dt` in the
three production config files.

## Run Order

Run from the repository root.

```powershell
.\results\cases_for_furi\run_00_spinup.ps1
.\results\cases_for_furi\run_01_stationary.ps1
.\results\cases_for_furi\run_02_y_oscillation.ps1
.\results\cases_for_furi\run_03_rotation.ps1
```

Or run everything sequentially:

```powershell
.\results\cases_for_furi\run_all.ps1
```

The three production cases all resume from:

```text
results/cases_for_furi/output/00_stationary_spinup_t0_t100/snap_100.9001.npz
```

## Case Parameters To Edit

- Free Y motion is controlled by:
  `experimental_config_y_dof.txt`
- Free theta motion is controlled by:
  `experimental_config_theta_dof.txt`
- Prescribed cylinder translation/rotation in the run configs should stay:
  `cylinder_translation_mode = stationary`
  `cylinder_rotation_mode = stationary`
