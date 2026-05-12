"""Tests for post-processing helper scripts."""

import os
import numpy as np

from analyze_aerodynamics import _read_config, plot_shedding_spectrum, save_pressure_coefficient_report
from time_average_snapshots import compute_time_averaged_fields, save_time_averaged_fields
from view_snapshot_viewer import (
    _compute_snapshot_vorticity,
    pick_slice_and_component,
    plot_vorticity_video,
)


class TestAnalyzeAerodynamicsHelpers:
    def test_read_config_parses_bool_and_float(self, tmp_path):
        config_path = tmp_path / "config.txt"
        config_path.write_text(
            "cylinder = true\n"
            "cylinder_radius = 0.25\n"
            "note = ignored\n"
            "re = 100  # inline comment\n",
            encoding="utf-8",
        )
        config = _read_config(str(config_path))
        assert config["cylinder"] == 1.0
        assert config["cylinder_radius"] == 0.25
        assert config["re"] == 100.0
        assert "note" not in config

    def test_plot_shedding_spectrum_writes_png(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            t = np.linspace(0.0, 20.0, 400, endpoint=False)
            c_l = np.sin(2.0 * np.pi * 0.4 * t) + 0.2 * np.sin(2.0 * np.pi * 0.8 * t)
            c_d = np.zeros_like(c_l)
            arr = np.column_stack((t, c_d, c_l))
            csv_path = tmp_path / "aero.csv"
            np.savetxt(csv_path, arr, delimiter=",", header="t,c_d,c_l", comments="")

            plot_shedding_spectrum(
                str(csv_path),
                save_name="test_shedding_spectrum.png",
                t_min=0.0,
                f_min=0.1,
                f_max=1.5,
                char_length=1.0,
                u_ref=1.0,
            )

            assert (results_dir / "test_shedding_spectrum.png").exists()
        finally:
            os.chdir(old_cwd)

    def test_save_pressure_coefficient_report_writes_csv_and_png(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        nx = ny = 64
        xf = np.linspace(0.0, 1.0, nx + 1)
        yf = np.linspace(0.0, 1.0, ny + 1)
        xc = 0.5 * (xf[:-1] + xf[1:])
        yc = 0.5 * (yf[:-1] + yf[1:])
        p = np.broadcast_to(xc[:, np.newaxis], (nx, ny)).copy()

        np.savez(outdir / "uniform_grid.npz", xf=xf, yf=yf)
        snap_path = outdir / "snap_000.0000.npz"
        np.savez(snap_path, p=p, t=0.0)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_pressure_coefficient_report(
                str(snap_path),
                u_ref=1.0,
                save_csv="test_cp_theta.csv",
                save_plot="test_cp_theta.png",
                cylinder_center=(0.5, 0.5),
                cylinder_radius=0.25,
                n_samples=180,
            )
            csv_path = results_dir / "test_cp_theta.csv"
            png_path = results_dir / "test_cp_theta.png"
            assert csv_path.exists()
            assert png_path.exists()

            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            theta_deg = np.atleast_1d(data["theta_deg"]).astype(float)
            c_p = np.atleast_1d(data["c_p_zero_mean_surface"]).astype(float)
            idx0 = int(np.argmin(np.abs(theta_deg - 0.0)))
            idx180 = int(np.argmin(np.abs(theta_deg - 180.0)))
            assert c_p[idx0] > 0.0
            assert c_p[idx180] < 0.0
        finally:
            os.chdir(old_cwd)

    def test_time_average_snapshots_computes_mean_and_rms(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        xf = np.array([0.0, 1.0, 2.0], dtype=float)
        yf = np.array([0.0, 1.0, 2.0], dtype=float)
        np.savez(outdir / "uniform_grid.npz", xf=xf, yf=yf)

        p0 = np.full((2, 2), 10.0, dtype=float)
        p1 = np.full((2, 2), 14.0, dtype=float)

        u0 = np.full((3, 2), 1.0, dtype=float)
        u1 = np.full((3, 2), 3.0, dtype=float)
        v0 = np.full((2, 3), 2.0, dtype=float)
        v1 = np.full((2, 3), 6.0, dtype=float)

        np.savez(outdir / "snap_000.0000.npz", u=u0, v=v0, p=p0, t=0.0)
        np.savez(outdir / "snap_001.0000.npz", u=u1, v=v1, p=p1, t=1.0)

        stats = compute_time_averaged_fields(indir=str(outdir), t_min=0.0)
        assert np.allclose(stats["u_mean"], 2.0)
        assert np.allclose(stats["v_mean"], 4.0)
        assert np.allclose(stats["p_mean"], 12.0)
        assert np.allclose(stats["u_rms"], 1.0)
        assert np.allclose(stats["v_rms"], 2.0)
        assert np.allclose(stats["xc"], np.array([0.5, 1.5]))
        assert np.allclose(stats["yc"], np.array([0.5, 1.5]))
        assert stats["n_snapshots"] == 2

        save_path = save_time_averaged_fields(
            indir=str(outdir),
            results_dir=str(results_dir),
            save_name="test_time_avg.npz",
        )
        assert os.path.exists(save_path)


class TestSnapshotViewerHelpers:
    def test_pick_slice_and_component_for_component_last(self):
        arr = np.arange(3 * 4 * 2).reshape(3, 4, 2)
        picked = pick_slice_and_component(arr, slice_idx=None, comp_idx=1)
        assert np.array_equal(picked, arr[:, :, 1])

    def test_pick_slice_and_component_for_3d_slice(self):
        arr = np.arange(5 * 3 * 6).reshape(5, 3, 6)
        picked = pick_slice_and_component(arr, slice_idx=2, comp_idx=None)
        assert np.array_equal(picked, arr[2, :, :])

    def test_compute_snapshot_vorticity_for_uniform_flow(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()

        nx, ny = 4, 3
        u = np.ones((nx + 1, ny), dtype=float)
        v = np.zeros((nx, ny + 1), dtype=float)
        np.savez(outdir / "snap_000.0000.npz", u=u, v=v, t=0.0)

        xc, yc, omega = _compute_snapshot_vorticity(str(outdir / "snap_000.0000.npz"))
        assert xc.shape == (nx,)
        assert yc.shape == (ny,)
        assert omega.shape == (nx, ny)
        assert np.allclose(omega, 0.0)

    def test_plot_vorticity_video_writes_gif(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"

        nx, ny = 4, 3
        u0 = np.ones((nx + 1, ny), dtype=float)
        v0 = np.zeros((nx, ny + 1), dtype=float)
        u1 = np.ones((nx + 1, ny), dtype=float)
        v1 = np.zeros((nx, ny + 1), dtype=float)
        v1[:, :] = np.linspace(0.0, 1.0, nx)[:, np.newaxis]

        np.savez(outdir / "snap_000.0000.npz", u=u0, v=v0, t=0.0)
        np.savez(outdir / "snap_000.1000.npz", u=u1, v=v1, t=0.1)

        plot_vorticity_video(
            snapshot_dir=str(outdir),
            save_name="test_vorticity.gif",
            fps=2,
            results_dir=str(results_dir),
        )

        assert (results_dir / "test_vorticity.gif").exists()

    def test_plot_vorticity_video_accepts_frame_stride(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"

        nx, ny = 4, 3
        for idx in range(5):
            u = np.ones((nx + 1, ny), dtype=float)
            v = np.zeros((nx, ny + 1), dtype=float)
            v[:, :] = float(idx) * np.linspace(0.0, 1.0, nx)[:, np.newaxis]
            np.savez(outdir / f"snap_{idx:03d}.0000.npz", u=u, v=v, t=float(idx))

        plot_vorticity_video(
            snapshot_dir=str(outdir),
            save_name="test_vorticity_stride.gif",
            fps=2,
            frame_stride=2,
            results_dir=str(results_dir),
        )

        assert (results_dir / "test_vorticity_stride.gif").exists()
