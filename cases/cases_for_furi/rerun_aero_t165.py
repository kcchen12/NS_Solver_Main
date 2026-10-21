"""Reprocess the three saved coefficient histories from t=165 with Hann/4 FFTs."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_aerodynamics import (
    _coefficient_window_stats,
    _compute_fourier_power_spectrum,
    plot_shedding_spectrum,
)
from view_snapshot_viewer import plot_coeff_history


def main():
    base = Path(__file__).resolve().parent / "results"
    destination = base / "analysis_t165_hann4"
    cases = {
        "01_stationary_from_t100_t225": base / "01_stationary_from_t100_t225/latest_run/aero.csv",
        "02_y_oscillation_from_t100_t225": base / "02_y_oscillation_from_t100_t225/aero.csv",
        "03_rotation_from_t100_t225": base / "03_rotation_from_t100_t225/aero.csv",
    }
    summary = []
    for case, source in cases.items():
        data = np.atleast_1d(np.genfromtxt(source, delimiter=",", names=True))
        if not np.all(np.isfinite(np.column_stack([data[k] for k in ("t", "c_d", "c_l")]))):
            raise ValueError(f"Invalid coefficient samples in {source}")
        if not np.all(np.diff(data["t"]) > 0):
            raise ValueError(f"Unordered coefficient samples in {source}")
        selected = data[data["t"] >= 165.0]
        if selected.size < 32:
            raise ValueError(f"Insufficient samples after t=165 in {source}")
        out = destination / case
        out.mkdir(parents=True, exist_ok=True)
        selected_csv = out / "aero.csv"
        np.savetxt(selected_csv, np.column_stack([selected[k] for k in data.dtype.names]),
                   delimiter=",", header=",".join(data.dtype.names), comments="")
        stats = _coefficient_window_stats(data["t"], data["c_d"], data["c_l"], 165.0)
        stats["c_d_rms"] = float(np.sqrt(np.mean(selected["c_d"] ** 2)))
        ts, _, freq, power = _compute_fourier_power_spectrum(
            data["t"], data["c_l"], 165.0, 0.05, 2.0, window="hann", num_windows=4)
        peak = int(np.argmax(power))
        segment_size = ts.size // 4
        dt = float(np.median(np.diff(ts)))
        result = {
            "case": case, "source_csv": str(source.relative_to(ROOT)),
            "requested_t_min": 165.0, "samples": int(selected.size),
            "window": "hann", "num_windows": 4,
            "fft_uniform_samples": int(ts.size), "samples_per_window": int(segment_size),
            "discarded_fft_samples": int(ts.size % 4),
            "fft_dt": dt, "frequency_resolution": 1.0 / (segment_size * dt),
            "peak_frequency": float(freq[peak]), "peak_power": float(power[peak]),
            "strouhal": float(freq[peak]), "reference_length": 1.0, "reference_velocity": 1.0,
            **stats,
        }
        np.savetxt(out / "shedding_spectrum.csv", np.column_stack((freq, power)),
                   delimiter=",", header="frequency,power", comments="")
        plot_shedding_spectrum(str(source), t_min=165.0, window="hann", num_windows=4,
                               char_length=1.0, u_ref=1.0, results_dir=str(out))
        plot_coeff_history(str(selected_csv), save_name=str(out / "coeff_history.png"),
                           coeff_t_min=165.0, settled_coeff_t_min=165.0)
        plt.close("all")
        original_report = source.with_name("aero_report.txt")
        provenance = []
        if original_report.exists():
            provenance = [line for line in original_report.read_text().splitlines()
                          if line.startswith(("Force source", "WARNING"))]
        report = ["AERODYNAMIC ANALYSIS FROM SAVED COEFFICIENTS", "=" * 70,
                  f"Source CSV: {result['source_csv']}",
                  "No additional endpoint sample was dropped from the saved series.",
                  "Coefficient statistics use unweighted samples; FFT uses periodic Hann.",
                  "Four equal-length, non-overlapping segments; mean removed per segment.",
                  "FFT power uses taper-energy normalization and undoubled positive bins.",
                  "Strouhal = f L/U, with L=1 and U=1.", *provenance, ""]
        report.extend(f"{key}: {value}" for key, value in result.items())
        (out / "aero_report.txt").write_text("\n".join(report) + "\n")
        (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        summary.append(result)
        print(f"{case}: Cd mean={stats['c_d_mean']:.6g}, Cl RMS={stats['c_l_rms']:.6g}, "
              f"f_peak={freq[peak]:.6g}, t=[{stats['t_min']:.4f}, {stats['t_max']:.4f}]")
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved all analyses to {destination}")


if __name__ == "__main__":
    main()
