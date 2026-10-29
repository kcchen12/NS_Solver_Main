"""Plot mechanical damper power P=c*y_dot**2 for the spacing sweep."""

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAMPAIGN = Path(__file__).resolve().parent
sys.path.insert(0, str(CAMPAIGN.parents[1]))
from src.config import ConfigParser


def draw(ax, t, primary, downstream, total, spacing):
    mean = np.trapezoid(total, t) / (t[-1] - t[0])
    ax.plot(t, primary, label="Upstream", color="#2878b5", linewidth=1.1)
    ax.plot(t, downstream, label="Downstream", color="#e07b27", linewidth=1.1)
    ax.plot(t, total, label="Total", color="#238b57", linewidth=1.3)
    ax.axhline(mean, color="#555555", linestyle="--", linewidth=1,
               label=f"Mean total = {mean:.4g}")
    ax.set(title=f"{spacing} center spacing", xlabel="Simulation time",
           ylabel=r"Mechanical damper power $P=c\dot{y}^{2}$ (solver units)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper left")


def main():
    summary_dir = CAMPAIGN / "power_graphs"
    summary_dir.mkdir(exist_ok=True)
    overview, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    summary = []
    for ax, spacing in zip(axes.flat, ("2D", "3D", "5D", "7D")):
        name = f"spacing_{spacing}"
        run = CAMPAIGN / "runs" / name
        source = run / "output" / "tandem_y_velocity_squared.csv"
        data = np.genfromtxt(source, delimiter=",", names=True)
        t = data["t"]
        if len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError(f"{name}: invalid or insufficient time samples")
        # Prefer the damping actually saved by the simulation over an editable config.
        latest = sorted((run / "output").glob("snap_*.npz"))[-1]
        with np.load(latest, allow_pickle=False) as snapshot:
            if "meta_cylinder_free_y_damping" in snapshot:
                damping = float(snapshot["meta_cylinder_free_y_damping"])
            else:
                damping = ConfigParser(str(CAMPAIGN / name / "experimental_config.txt")).get(
                    "cylinder_free_y_damping", dtype=float)
        primary = damping * data["primary_velocity_y_squared"]
        downstream = damping * data["downstream_velocity_y_squared"]
        total = primary + downstream
        if not np.all(np.isfinite(total)) or np.any(total < 0):
            raise ValueError(f"{name}: invalid power samples")
        if not np.allclose(total, damping * data["total_velocity_y_squared"]):
            raise ValueError(f"{name}: total power is inconsistent")
        destination = run / "results"
        destination.mkdir(exist_ok=True)
        np.savetxt(destination / "harvested_power.csv",
                   np.column_stack((t, primary, downstream, total)), delimiter=",",
                   header="t,upstream_power,downstream_power,total_power", comments="")
        fig, single_ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        draw(single_ax, t, primary, downstream, total, spacing)
        fig.suptitle(f"Harvested mechanical power | damping c={damping:g}")
        fig.savefig(destination / "harvested_power.png", dpi=180)
        plt.close(fig)
        draw(ax, t, primary, downstream, total, spacing)
        means = [np.trapezoid(p, t) / (t[-1] - t[0])
                 for p in (primary, downstream, total)]
        summary.append([spacing, damping, t[0], t[-1], *means])
        print(f"{name}: t={t[-1]:g}, c={damping:g}, mean total={means[-1]:.6g}")
    overview.suptitle("Harvested mechanical power | 256 x 512 | independent vertical scales")
    overview.savefig(summary_dir / "harvested_power_all_cases.png", dpi=180)
    plt.close(overview)
    with (summary_dir / "mean_power.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["center_spacing", "damping", "t_start", "t_end",
                         "mean_upstream_power", "mean_downstream_power", "mean_total_power"])
        writer.writerows(summary)


if __name__ == "__main__":
    main()
