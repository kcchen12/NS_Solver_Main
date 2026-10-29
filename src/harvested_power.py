"""Mechanical power and energy from the free-Y dampers, in solver units."""

from pathlib import Path

import numpy as np


REPORT_HEADING = "HARVESTED MECHANICAL POWER AND ENERGY"


def harvested_power_report_lines(indir, pattern="snap_*.npz"):
    """Use every-step monitors when available, otherwise saved motion metadata.

    The interval is the complete available motion history, including startup.
    Coefficient statistics trimming does not apply to this energy integral.
    """
    lines = ["", "-" * 70, REPORT_HEADING, "-" * 70]
    try:
        folder = Path(indir)
        snapshots = sorted(folder.glob(pattern))
        if not snapshots:
            raise ValueError("no snapshots available for damping metadata")
        with np.load(snapshots[-1], allow_pickle=False) as data:
            damping = float(data["meta_cylinder_free_y_damping"])
            tandem = bool(data.get("meta_cylinder_downstream_enabled", False))
            free_y = bool(data.get("meta_cylinder_free_y_dof", False))
        if not free_y:
            raise ValueError("free-Y damper motion was not enabled")
        if not np.isfinite(damping) or damping < 0:
            raise ValueError("invalid Y damping coefficient")
        monitor = folder / ("tandem_y_velocity_squared.csv" if tandem
                            else "y_velocity_squared.csv")
        labels = ["Upstream" if tandem else "Primary"]
        if tandem:
            labels.append("Downstream")
        if monitor.exists():
            data = np.atleast_1d(np.genfromtxt(monitor, delimiter=",", names=True))
            t = data["t"]
            columns = (["primary_velocity_y_squared", "downstream_velocity_y_squared"]
                       if tandem else ["velocity_y_squared"])
            squared = np.column_stack([data[column] for column in columns])
            source = "every-step velocity monitor"
        else:
            rows = []
            for path in snapshots:
                with np.load(path, allow_pickle=False) as data:
                    velocities = [float(data["meta_cylinder_free_y_velocity"])]
                    if tandem:
                        velocities.append(float(data["meta_cylinder_downstream_velocity_y"]))
                    rows.append([float(data["t"]), *np.square(velocities)])
            values = np.asarray(rows)
            t, squared = values[:, 0], values[:, 1:]
            source = "saved snapshots (energy accuracy depends on save_dt)"
        if (len(t) < 2 or not np.all(np.isfinite(t))
                or not np.all(np.diff(t) > 0)
                or not np.all(np.isfinite(squared)) or np.any(squared < 0)):
            raise ValueError("motion history must have finite, increasing times and nonnegative velocity squared")
        powers = damping * squared
        powers = np.column_stack([powers, powers.sum(axis=1)])
        labels.append("Total")
        # Explicit trapezoidal integration supports NumPy 1.x and variable dt.
        energies = np.sum(0.5 * (powers[1:] + powers[:-1]) * np.diff(t)[:, None], axis=0)
        means = energies / (t[-1] - t[0])
        peaks = powers.max(axis=0)
        lines.extend([
            "Definition        : P=c*y_dot^2; E=integral(P dt)",
            f"Y damping (c)     : {damping:.9g}",
            f"Source            : {source}",
            f"Energy interval   : t=[{t[0]:.9g}, {t[-1]:.9g}] (full available history)",
            "Units             : solver power / solver energy; before conversion efficiency",
        ])
        for label, mean, peak, energy in zip(labels, means, peaks, energies):
            lines.extend([f"{label} mean power : {mean:.9g}",
                          f"{label} peak power : {peak:.9g}",
                          f"{label} energy     : {energy:.9g}"])
    except (OSError, ValueError, KeyError, IndexError) as exc:
        lines.append(f"Unavailable       : {exc}")
    return lines
