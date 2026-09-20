# Documentation

Project notes, examples, and supporting references live here.

## Contents

- [examples/config_example.txt](examples/config_example.txt): current example runtime configuration
- [examples/experimental_config_example.txt](examples/experimental_config_example.txt): current example experimental cylinder configuration
- [examples/post_config_example.txt](examples/post_config_example.txt): current example post-processing configuration
- [reports/BRANCH_EXPERIMENT_REPORT.md](reports/BRANCH_EXPERIMENT_REPORT.md): branch experiment notes
- [reports/OPTIMIZATION_NOTES.md](reports/OPTIMIZATION_NOTES.md): implementation and optimization notes
- [math/MATHEMATICAL_FORMULAS.md](math/MATHEMATICAL_FORMULAS.md): mathematical formulas and derivations

## Usage Notes

- The recommended runtime config fields are the unified keys used by `main.py` and `pre_generate_grid.py`, such as `uniform_grid`, `grid_beta_x`, `grid_beta_y`, `grid_uniform_x_start`, and `grid_uniform_y_start`
- Experimental cylinder controls live in `experimental_config.txt`
- `cylinder_free_x_dof`, `cylinder_free_y_dof`, and `cylinder_free_theta_dof` enable experimental streamwise/transverse/angular spring-mass-damper coupling for the selected bluff body
- `cylinder_free_x_max_displacement_percent` and `cylinder_free_y_max_displacement_percent` accept a positive percent or `off` to disable the displacement clamp
- `cylinder_free_x_release_time` and `cylinder_free_y_release_time` hold an enabled translation axis fixed until the configured simulation time
- Post-run plotting and reporting controls live in `post_config.txt`, not in the main runtime config
