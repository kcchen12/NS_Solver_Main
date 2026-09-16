"""Tests for post-processing helper scripts."""

import os
import numpy as np

import view_snapshot_viewer
from analyze_aerodynamics import (
    _compute_coefficients,
    _compute_forces,
    _compute_pressure_forces,
    _coefficient_window_stats,
    _estimate_cylinder_geometry,
    _estimate_scales,
    _build_surface_force_plan,
    _compute_surface_stress_forces,
    _estimate_kinematic_viscosity,
    CylinderGeometry,
    _read_config,
    _resolve_force_source,
    plot_drag_decomposition,
    plot_shedding_spectrum,
    run_analysis,
    save_pressure_coefficient_report,
)
from time_average_snapshots import compute_time_averaged_fields, save_time_averaged_fields
from time_average_snapshots import plot_time_averaged_fields
from view_snapshot_viewer import (
    _load_cylinder_overlay_geometry_for_snapshot,
    _compute_snapshot_vorticity,
    detect_startup_trim_index,
    pick_slice_and_component,
    plot_snapshot_key,
    plot_vorticity_video,
    resolve_window_start_time,
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
            c_l = np.sin(2.0 * np.pi * 0.4 * t) + 0.2 * \
                np.sin(2.0 * np.pi * 0.8 * t)
            c_d = np.zeros_like(c_l)
            arr = np.column_stack((t, c_d, c_l))
            csv_path = tmp_path / "aero.csv"
            np.savetxt(csv_path, arr, delimiter=",",
                       header="t,c_d,c_l", comments="")

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

    def test_surface_stress_force_runs_on_simple_snapshot_fields(self):
        nx = ny = 24
        xf = np.linspace(-1.0, 1.0, nx + 1)
        yf = np.linspace(-1.0, 1.0, ny + 1)
        xc = 0.5 * (xf[:-1] + xf[1:])
        yc = 0.5 * (yf[:-1] + yf[1:])
        u = np.ones((nx + 1, ny))
        v = np.zeros((nx, ny + 1))
        p = np.zeros((nx, ny))

        fx, fy = _compute_surface_stress_forces(
            u,
            v,
            p,
            xc,
            yc,
            geom=type("Geom", (), {
                "center_x": 0.0,
                "center_y": 0.0,
                "radius": 0.25,
            })(),
            nu=0.01,
            n_samples=64,
        )

        assert np.isfinite(fx)
        assert np.isfinite(fy)

    def test_pressure_force_integral_separates_drag_and_lift_axes(self):
        radius = 0.25
        nx = ny = 160
        xf = np.linspace(-1.0, 1.0, nx + 1)
        yf = np.linspace(-1.0, 1.0, ny + 1)
        xc = 0.5 * (xf[:-1] + xf[1:])
        yc = 0.5 * (yf[:-1] + yf[1:])
        x, y = np.meshgrid(xc, yc, indexing="ij")
        plan = _build_surface_force_plan(
            xc,
            yc,
            CylinderGeometry(center_x=0.0, center_y=0.0, radius=radius),
            n_samples=720,
        )

        fx_x, fy_x = _compute_pressure_forces(x, plan)
        fx_y, fy_y = _compute_pressure_forces(y, plan)
        expected = -np.pi * radius**2

        assert np.isclose(fx_x, expected)
        assert np.isclose(fy_x, 0.0, atol=1e-12)
        assert np.isclose(fx_y, 0.0, atol=1e-12)
        assert np.isclose(fy_y, expected)

    def test_compute_coefficients_uses_standard_dynamic_pressure_form(self):
        c_d, c_l = _compute_coefficients(
            1.0, -2.0, u_ref=1.0, char_length=2.0, rho=1.0)
        assert np.isclose(c_d, 1.0)
        assert np.isclose(c_l, -2.0)

    def test_coefficient_stats_report_lift_amplitude_and_peak(self):
        t = np.arange(5.0)
        c_d = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        c_l = np.array([-0.5, 0.0, 0.25, 0.75, -0.25])

        stats = _coefficient_window_stats(t, c_d, c_l, start=1.0)

        assert stats is not None
        assert np.isclose(stats["c_l_min"], -0.25)
        assert np.isclose(stats["c_l_max"], 0.75)
        assert np.isclose(stats["c_l_amp_half_range"], 0.5)
        assert np.isclose(stats["c_l_abs_max"], 0.75)

    def test_surface_viscosity_prefers_snapshot_re_metadata(self, tmp_path):
        snap_path = tmp_path / "snap_000.0000.npz"
        config_path = tmp_path / "config.txt"
        np.savez(
            snap_path,
            p=np.zeros((2, 2)),
            meta_re=200.0,
            meta_re_is_cylinder_based=True,
        )
        config_path.write_text(
            "re = 100.0\nre_is_cylinder_based = true\n",
            encoding="utf-8",
        )
        geom = type("Geom", (), {"radius": 0.5})()

        nu = _estimate_kinematic_viscosity(
            str(snap_path),
            str(config_path),
            u_ref=1.0,
            geom=geom,
            char_length=1.0,
        )

        assert np.isclose(nu, 0.005)

    def test_surface_viscosity_uses_snapshot_re_basis_metadata(self, tmp_path):
        snap_path = tmp_path / "snap_000.0000.npz"
        config_path = tmp_path / "config.txt"
        np.savez(
            snap_path,
            p=np.zeros((2, 2)),
            meta_re=200.0,
            meta_re_is_cylinder_based=False,
        )
        config_path.write_text(
            "re = 100.0\nre_is_cylinder_based = true\n",
            encoding="utf-8",
        )
        geom = type("Geom", (), {"radius": 0.5})()

        nu = _estimate_kinematic_viscosity(
            str(snap_path),
            str(config_path),
            u_ref=1.0,
            geom=geom,
            char_length=2.0,
        )

        assert np.isclose(nu, 0.01)

    def test_aero_geometry_prefers_snapshot_metadata(self, tmp_path):
        snap_path = tmp_path / "snap_000.0000.npz"
        config_path = tmp_path / "config.txt"
        np.savez(
            snap_path,
            p=np.zeros((3, 4)),
            meta_lx=10.0,
            meta_ly=8.0,
            meta_nx=3,
            meta_ny=4,
            meta_cylinder_center_x=1.25,
            meta_cylinder_center_y=-0.5,
            meta_cylinder_radius=0.75,
        )
        config_path.write_text(
            "cylinder = true\n"
            "cylinder_center_x = 9.0\n"
            "cylinder_center_y = 9.0\n"
            "cylinder_radius = 2.0\n",
            encoding="utf-8",
        )

        geom = _estimate_cylinder_geometry(str(snap_path), str(config_path))
        l_char, *_ = _estimate_scales(
            str(snap_path),
            length_scale=None,
            use_cylinder_diameter=True,
            u_ref=1.0,
            config_path=str(config_path),
        )

        assert geom.center_x == 1.25
        assert geom.center_y == -0.5
        assert geom.radius == 0.75
        assert np.isclose(l_char, 1.5)

    def test_ibm_force_metadata_is_preferred_for_coefficients(self, tmp_path):
        snap_path = tmp_path / "snap_000.0000.npz"
        p = np.zeros((2, 2), dtype=float)
        np.savez(
            snap_path,
            u=np.zeros((3, 2), dtype=float),
            v=np.zeros((2, 3), dtype=float),
            p=p,
            t=0.0,
            meta_ibm_force_x=3.5,
            meta_ibm_force_y=-1.25,
        )

        snapshots = [(0.0, str(snap_path))]
        assert _resolve_force_source("auto", snapshots) == "surface-full"
        assert _resolve_force_source("ibm", snapshots) == "ibm"
        assert _resolve_force_source("surface", snapshots) == "surface"
        assert _resolve_force_source("surface-full", snapshots) == "surface-full"

        fx, fy = _compute_forces(
            str(snap_path),
            xc=np.array([0.5, 1.5]),
            yc=np.array([0.5, 1.5]),
            geom=None,
            force_source="ibm",
        )

        assert fx == 3.5
        assert fy == -1.25

    def test_auto_force_source_prefers_surface_full_when_fields_are_available(self, tmp_path):
        snap_path = tmp_path / "snap_000.0000.npz"
        np.savez(
            snap_path,
            u=np.zeros((3, 2), dtype=float),
            v=np.zeros((2, 3), dtype=float),
            p=np.zeros((2, 2), dtype=float),
            t=0.0,
            meta_ibm_force_x=3.5,
            meta_ibm_force_y=-1.25,
        )

        snapshots = [(0.0, str(snap_path))]
        assert _resolve_force_source("auto", snapshots) == "surface-full"
        assert _resolve_force_source("ibm", snapshots) == "ibm"

    def test_run_analysis_writes_drag_decomposition_csv(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        config_path = tmp_path / "config.txt"
        config_path.write_text(
            "cylinder = true\n"
            "cylinder_radius = 0.25\n"
            "re = 100.0\n"
            "re_is_cylinder_based = true\n",
            encoding="utf-8",
        )

        nx = ny = 24
        xf = np.linspace(-1.0, 1.0, nx + 1)
        yf = np.linspace(-1.0, 1.0, ny + 1)
        xc = 0.5 * (xf[:-1] + xf[1:])
        p = np.broadcast_to(xc[:, np.newaxis], (nx, ny)).copy()
        u = np.ones((nx + 1, ny), dtype=float)
        v = np.zeros((nx, ny + 1), dtype=float)

        np.savez(outdir / "uniform_grid.npz", xf=xf, yf=yf)
        for t in (0.0, 1.0):
            np.savez(
                outdir / f"snap_{t:08.4f}.npz",
                u=u,
                v=v,
                p=p,
                t=t,
                meta_lx=2.0,
                meta_ly=2.0,
                meta_x_min=-1.0,
                meta_y_min=-1.0,
                meta_cylinder_center_x=0.0,
                meta_cylinder_center_y=0.0,
                meta_cylinder_radius=0.25,
                meta_re=100.0,
                meta_re_is_cylinder_based=True,
            )

        decomp_path = results_dir / "drag_decomposition.csv"
        status = run_analysis(
            indir=str(outdir),
            config=str(config_path),
            u_ref=1.0,
            use_cylinder_diameter=True,
            t_min=0.0,
            save_series=None,
            save_drag_decomposition=str(decomp_path),
            save_report=None,
            force_source="surface-full",
            drop_final_sample=False,
        )

        assert status == 0
        assert decomp_path.exists()
        data = np.genfromtxt(decomp_path, delimiter=",", names=True)
        names = data.dtype.names or ()
        assert "pressure_c_d" in names
        assert "viscous_c_d" in names
        assert "total_c_d" in names
        assert np.allclose(
            data["pressure_c_d"] + data["viscous_c_d"],
            data["total_c_d"],
        )

    def test_plot_drag_decomposition_writes_png(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        csv_path = tmp_path / "drag_decomposition.csv"
        t = np.linspace(0.0, 1.0, 8)
        pressure_cd = np.linspace(1.0, 1.1, t.size)
        viscous_cd = np.linspace(0.2, 0.25, t.size)
        pressure_cl = np.sin(t)
        viscous_cl = 0.1 * np.sin(t)
        out = np.column_stack(
            (
                t,
                pressure_cd,
                viscous_cd,
                pressure_cd + viscous_cd,
                pressure_cl,
                viscous_cl,
                pressure_cl + viscous_cl,
            )
        )
        np.savetxt(
            csv_path,
            out,
            delimiter=",",
            header=(
                "t,pressure_c_d,viscous_c_d,total_c_d,"
                "pressure_c_l,viscous_c_l,total_c_l"
            ),
            comments="",
        )

        save_path = plot_drag_decomposition(
            str(csv_path),
            save_name="decomp.png",
            results_dir=str(results_dir),
        )

        assert save_path == str(results_dir / "decomp.png")
        assert (results_dir / "decomp.png").exists()

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

        plot_path = plot_time_averaged_fields(
            save_path,
            save_name="test_time_avg.png",
            results_dir=str(results_dir),
            x_scale=1.5,
            y_scale=0.75,
        )
        assert os.path.exists(plot_path)


class TestSnapshotViewerHelpers:
    def test_fractional_coeff_t_min_uses_run_fraction(self):
        t = np.linspace(0.0, 100.0, 101)
        c_d = np.ones_like(t)
        c_l = np.ones_like(t)

        idx = detect_startup_trim_index(t, c_d, c_l, coeff_t_min=0.70)

        assert idx == 70

    def test_fractional_settled_coeff_t_min_uses_run_fraction(self):
        t = np.linspace(10.0, 110.0, 101)

        start = resolve_window_start_time(t, 0.53)

        assert start == 63.0

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

        xc, yc, omega = _compute_snapshot_vorticity(
            str(outdir / "snap_000.0000.npz"))
        assert xc.shape == (nx,)
        assert yc.shape == (ny,)
        assert omega.shape == (nx, ny)
        assert np.allclose(omega, 0.0)

    def test_plot_snapshot_key_accepts_independent_axis_scales(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        snap_path = outdir / "snap_000.0000.npz"
        np.savez(snap_path, p=np.arange(12, dtype=float).reshape(3, 4), t=0.0)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            plot_snapshot_key(
                str(snap_path),
                "p",
                save_name="scaled_pressure.png",
                x_scale=1.8,
                y_scale=0.6,
            )
            assert (results_dir / "scaled_pressure.png").exists()
        finally:
            os.chdir(old_cwd)

    def test_cylinder_overlay_prefers_snapshot_metadata(self, tmp_path):
        outdir = tmp_path / "output"
        outdir.mkdir()
        config_path = tmp_path / "config.txt"
        config_path.write_text(
            "cylinder = true\n"
            "lx = 10.0\n"
            "ly = 10.0\n"
            "cylinder_center_x = 9.0\n"
            "cylinder_center_y = 9.0\n"
            "cylinder_radius = 1.0\n",
            encoding="utf-8",
        )

        nx, ny = 4, 3
        u = np.ones((nx + 1, ny), dtype=float)
        v = np.zeros((nx, ny + 1), dtype=float)
        snap_path = outdir / "snap_000.0000.npz"
        np.savez(
            snap_path,
            u=u,
            v=v,
            t=0.0,
            meta_cylinder_enabled=np.array(True),
            meta_cylinder_center_x=np.array(0.0),
            meta_cylinder_center_y=np.array(0.0),
            meta_cylinder_radius=np.array(0.5),
        )

        geom = _load_cylinder_overlay_geometry_for_snapshot(
            str(snap_path),
            config_path=str(config_path),
        )
        assert geom == (0.0, 0.0, 0.5)

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
            np.savez(outdir / f"snap_{idx:03d}.0000.npz",
                     u=u, v=v, t=float(idx))

        plot_vorticity_video(
            snapshot_dir=str(outdir),
            save_name="test_vorticity_stride.gif",
            fps=2,
            frame_stride=2,
            results_dir=str(results_dir),
        )

        assert (results_dir / "test_vorticity_stride.gif").exists()

    def test_plot_vorticity_video_can_disable_cylinder_overlay(self, tmp_path, monkeypatch):
        outdir = tmp_path / "output"
        outdir.mkdir()
        results_dir = tmp_path / "results"

        nx, ny = 4, 3
        u = np.ones((nx + 1, ny), dtype=float)
        v = np.zeros((nx, ny + 1), dtype=float)
        np.savez(outdir / "snap_000.0000.npz", u=u, v=v, t=0.0)
        np.savez(outdir / "snap_000.1000.npz", u=u, v=v, t=0.1)

        def fail_overlay_load(*args, **kwargs):
            raise AssertionError("overlay geometry should not be loaded")

        monkeypatch.setattr(
            view_snapshot_viewer,
            "_load_cylinder_overlay_geometry_for_snapshot",
            fail_overlay_load,
        )

        plot_vorticity_video(
            snapshot_dir=str(outdir),
            save_name="test_vorticity_no_overlay.gif",
            fps=2,
            results_dir=str(results_dir),
            draw_cylinder_overlay=False,
        )

        assert (results_dir / "test_vorticity_no_overlay.gif").exists()
