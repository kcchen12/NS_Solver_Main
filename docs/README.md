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
- Experimental cylinder and jet-actuator controls live in `experimental_config.txt`
- Post-run plotting and reporting controls live in `post_config.txt`, not in the main runtime config
