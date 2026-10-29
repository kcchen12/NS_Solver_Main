"""Tandem-cylinder configuration and short solver integration checks."""
import sys

import numpy as np
import pytest

import main
from src.grid import CartesianGrid
from src.ibm import ImmersedBoundary
from src.io_utils import load_snapshot
from view_snapshot_viewer import _load_downstream_cylinder_overlay_geometry_for_snapshot


def parse_case(tmp_path, monkeypatch, enabled=True, ratio="3.0", master=True, extra=()):
    config = tmp_path / "config.txt"
    experiment = tmp_path / "experimental_config.txt"
    post = tmp_path / "post_config.txt"
    config.write_text("\n".join([
        "nx = 48", "ny = 32", "lx = 8.0", "ly = 4.0", "uniform_grid = true",
        "cylinder = true", "cylinder_center_x = 2.0", "cylinder_center_y = 2.0",
        "cylinder_radius = 0.4", "t_end = 0.002", "save_dt = 0.001", "verbose = false",
        f"outdir = {tmp_path / 'output'}",
    ]))
    experiment.write_text(
        f"enable_experimental_config = {str(master).lower()}\n"
        f"cylinder_downstream = {str(enabled).lower()}\n"
        f"cylinder_downstream_spacing_ratio = {ratio}\n")
    post.write_text("plot = false\n")
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--config", str(config), "--experiment-config", str(experiment),
        "--post-config", str(post), *extra])
    return main.parse_args()


def test_downstream_spacing_uses_original_diameter(tmp_path, monkeypatch):
    args = parse_case(tmp_path, monkeypatch)
    assert main._resolve_downstream_cylinder_geometry(args) == pytest.approx((4.4, 2.0, 0.4))


@pytest.mark.parametrize("master,enabled", [(False, True), (True, False)])
def test_downstream_is_optional_and_experiment_gated(tmp_path, monkeypatch, master, enabled):
    args = parse_case(tmp_path, monkeypatch, master=master, enabled=enabled)
    assert main._resolve_downstream_cylinder_geometry(args) is None


def test_downstream_cli_overrides_experimental_values(tmp_path, monkeypatch):
    args = parse_case(tmp_path, monkeypatch, extra=["--cylinder-downstream-spacing-ratio", "4"])
    assert main._resolve_downstream_cylinder_geometry(args)[0] == pytest.approx(5.2)


@pytest.mark.parametrize("ratio", ["0", "1", "-1", "nan", "inf", "10"])
def test_downstream_rejects_overlap_nonfinite_or_outside_domain(tmp_path, monkeypatch, ratio):
    with pytest.raises(SystemExit):
        parse_case(tmp_path, monkeypatch, ratio=ratio)


@pytest.mark.parametrize("extra", [
    ["--cylinder", "false"], ["--cylinder-experiment", "square"],
])
def test_downstream_requires_primary_circle(tmp_path, monkeypatch, extra):
    with pytest.raises(SystemExit):
        parse_case(tmp_path, monkeypatch, extra=extra)


@pytest.mark.parametrize("motion", ["stationary", "oscillatory-y", "free-y"])
def test_tandem_solver_enforces_both_bodies_and_saves_geometry(tmp_path, monkeypatch, motion):
    extra = []
    if motion == "free-y":
        extra = ["--cylinder-free-y-dof", "true"]
    elif motion != "stationary":
        extra = ["--cylinder-translation-mode", motion,
                 "--cylinder-translation-y-percent", "10",
                 "--cylinder-translation-frequency", "0.5"]
    args = parse_case(tmp_path, monkeypatch, extra=extra)
    monkeypatch.setattr(main, "_run_auto_outputs", lambda grid, args: None)
    solver = main.run(args)
    grid = solver.grid
    rear_u = (grid.xf[:, None] - 4.4)**2 + (grid.yc[None, :] - 2.0)**2 < 0.4**2
    rear_v = (grid.xc[:, None] - 4.4)**2 + (grid.yf[None, :] - 2.0)**2 < 0.4**2
    if motion == "free-y":
        # Free bodies use a smoothed IBM interface; enforce exact velocity in the core.
        weight_u, weight_v = solver.ibm._free_body_force_weights(solver.ibm.free_y_circles[1])
        rear_u = weight_u >= 1.0 - 1e-12
        rear_v = weight_v >= 1.0 - 1e-12
    assert np.any(rear_u) and np.any(rear_v)
    assert np.all(solver.ibm.mask_u[rear_u]) and np.all(solver.ibm.mask_v[rear_v])
    np.testing.assert_allclose(solver.u[rear_u], 0.0, atol=1e-12)
    rear_velocity = (solver.ibm.free_y_circles[1].velocity_y if motion == "free-y" else 0.0)
    np.testing.assert_allclose(solver.v[rear_v], rear_velocity, atol=1e-12)
    assert np.all(np.isfinite(solver.u)) and np.all(np.isfinite(solver.p))
    if motion == "stationary":
        primary_u = (grid.xf[:, None] - 2.0)**2 + (grid.yc[None, :] - 2.0)**2 < 0.4**2
        primary_v = (grid.xc[:, None] - 2.0)**2 + (grid.yf[None, :] - 2.0)**2 < 0.4**2
        np.testing.assert_allclose(solver.u[primary_u], 0.0, atol=1e-12)
        np.testing.assert_allclose(solver.v[primary_v], 0.0, atol=1e-12)
    snapshot = sorted((tmp_path / "output").glob("snap_*.npz"))[-1]
    *_, metadata = load_snapshot(str(snapshot), fmt="numpy")
    assert bool(metadata["cylinder_downstream_enabled"])
    assert metadata["cylinder_downstream_center_x"] == pytest.approx(4.4)
    assert metadata["cylinder_downstream_spacing_ratio"] == pytest.approx(3.0)
    assert _load_downstream_cylinder_overlay_geometry_for_snapshot(str(snapshot)) == pytest.approx((4.4, 2, 0.4))


def test_main_outline_includes_both_cylinders(tmp_path, monkeypatch):
    args = parse_case(tmp_path, monkeypatch)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    main._plot_ibm_outline(ax, args)
    assert len(ax.lines) == 2
    centers = sorted((line.get_xdata().min() + line.get_xdata().max()) / 2 for line in ax.lines)
    assert centers == pytest.approx([2.0, 4.4])
    plt.close(fig)


def test_fixed_cylinder_contributes_to_total_without_changing_free_body_load():
    grid = CartesianGrid(48, 32, lx=8.0, ly=4.0)
    moving_only = ImmersedBoundary(grid)
    tandem = ImmersedBoundary(grid)
    for ibm in (moving_only, tandem):
        ibm.add_free_y_circle(2.0, 2.0, 0.4, mass=1.0, damping=0.0,
                              stiffness=0.0, force_relaxation=1.0)
    tandem.add_circle(4.4, 2.0, 0.4)
    u = np.ones_like(grid.zeros_u())
    v = np.full_like(grid.zeros_v(), 0.2)
    dt = 0.1
    moving_force = moving_only.apply(u.copy(), v.copy(), dt=dt)
    total_force = tandem.apply(u.copy(), v.copy(), dt=dt)
    static = ImmersedBoundary(grid)
    static.add_circle(4.4, 2.0, 0.4)
    static_force = static.apply(u.copy(), v.copy(), dt=dt)
    np.testing.assert_allclose(total_force, np.array(moving_force) + static_force)
    assert tandem.free_y_circles[0].last_force_x == pytest.approx(moving_only.free_y_circles[0].last_force_x)
    assert tandem.free_y_circles[0].last_force_y == pytest.approx(moving_only.free_y_circles[0].last_force_y)


@pytest.mark.parametrize("release_time", [0.0, 1.0])
def test_tandem_free_y_velocity_squared_monitor(tmp_path, monkeypatch, release_time):
    args = parse_case(tmp_path, monkeypatch, extra=[
        "--cylinder-free-y-dof", "true",
        "--cylinder-free-y-initial-velocity", "-0.12",
        "--cylinder-free-y-release-time", str(release_time),
        "--cylinder-free-y-monitor-velocity-squared", "true",
    ])
    monkeypatch.setattr(main, "_run_auto_outputs", lambda grid, args: None)
    step_times = []
    original_step = main.FractionalStepSolver.step

    def track_step(solver, dt):
        result = original_step(solver, dt)
        step_times.append(solver.t)
        return result

    monkeypatch.setattr(main.FractionalStepSolver, "step", track_step)
    solver = main.run(args)
    history = np.loadtxt(tmp_path / "output" / "y_velocity_squared.csv",
                         delimiter=",", skiprows=1, ndmin=2)
    np.testing.assert_allclose(history[:, 0], [0.0, *step_times])
    initial_velocity = -0.12 if release_time == 0.0 else 0.0
    assert history[0] == pytest.approx([0.0, initial_velocity, initial_velocity ** 2])
    if release_time > solver.t:
        np.testing.assert_array_equal(history[:, 2], 0.0)
    state = solver.ibm.first_free_y_circle_state(time=solver.t)
    assert history[-1, 1] == pytest.approx(state["velocity_y"])
    np.testing.assert_allclose(history[:, 2], history[:, 1] ** 2)
    assert (tmp_path / "output" / "y_velocity_squared.png").stat().st_size > 1000


@pytest.mark.parametrize("master", [False, True])
def test_velocity_squared_experimental_config_gate(tmp_path, monkeypatch, master):
    parse_case(tmp_path, monkeypatch, master=master)
    experiment = tmp_path / "experimental_config.txt"
    with experiment.open("a") as stream:
        stream.write("cylinder_free_y_monitor_velocity_squared = true\n")
    assert main.parse_args().cylinder_free_y_monitor_velocity_squared is master


def test_velocity_squared_monitor_ignored_without_free_y(tmp_path, monkeypatch):
    args = parse_case(tmp_path, monkeypatch, extra=[
        "--cylinder-free-y-monitor-velocity-squared", "true"])
    monkeypatch.setattr(main, "_run_auto_outputs", lambda grid, args: None)
    main.run(args)
    assert not (tmp_path / "output" / "y_velocity_squared.csv").exists()


@pytest.mark.parametrize("monitor,primary_monitor", [(False, False), (True, True), (True, False)])
def test_both_cylinders_oscillate_independently_of_power_monitor(
        tmp_path, monkeypatch, monitor, primary_monitor):
    args = parse_case(tmp_path, monkeypatch, extra=[
        "--cylinder-free-y-dof", "true",
        "--cylinder-free-y-initial-velocity", "-0.12",
        "--cylinder-free-y-monitor-velocity-squared", str(primary_monitor),
        "--cylinder-downstream-monitor-velocity-squared", str(monitor),
    ])
    args.t_end = 0.01
    monkeypatch.setattr(main, "_run_auto_outputs", lambda grid, args: None)
    monkeypatch.setattr(main.FractionalStepSolver, "suggest_dt", lambda *a, **k: 0.001)
    solver = main.run(args)
    front, rear = solver.ibm.free_y_circles
    assert front.y != front.cy0 and rear.y != rear.cy0
    assert rear.x == pytest.approx(4.4)
    assert front.last_force_y != pytest.approx(rear.last_force_y)
    assert front.velocity_y != pytest.approx(rear.velocity_y)
    history_path = tmp_path / "output" / "tandem_y_velocity_squared.csv"
    assert history_path.exists() is monitor
    assert (tmp_path / "output" / "y_velocity_squared.csv").exists() is primary_monitor
    if monitor:
        history = np.loadtxt(history_path, delimiter=",", skiprows=1)
        assert history.shape == (11, 6)
        np.testing.assert_allclose(history[:, 2], history[:, 1] ** 2)
        np.testing.assert_allclose(history[:, 4], history[:, 3] ** 2)
        np.testing.assert_allclose(history[:, 5], history[:, 2] + history[:, 4])
        assert history[-1, 1] == pytest.approx(front.velocity_y)
        assert history[-1, 3] == pytest.approx(rear.velocity_y)
        assert (tmp_path / "output" / "tandem_y_velocity_squared.png").stat().st_size > 1000

    snapshot = sorted((tmp_path / "output").glob("snap_*.npz"))[-1]
    *_, metadata = load_snapshot(str(snapshot), fmt="numpy")
    assert metadata["cylinder_downstream_center_y"] == pytest.approx(rear.y)
    assert metadata["cylinder_downstream_velocity_y"] == pytest.approx(rear.velocity_y)
    assert _load_downstream_cylinder_overlay_geometry_for_snapshot(str(snapshot)) == pytest.approx(
        (rear.x, rear.y, rear.radius))
    args.resume_from = str(snapshot)
    args.t_end = 0.012
    states_before_step = []
    original_step = main.FractionalStepSolver.step

    def inspect_restore(resumed_solver, dt):
        if not states_before_step:
            states_before_step.extend([
                (spec.y, spec.velocity_y, spec.last_force_y)
                for spec in resumed_solver.ibm.free_y_circles])
        return original_step(resumed_solver, dt)

    monkeypatch.setattr(main.FractionalStepSolver, "step", inspect_restore)
    main.run(args)
    np.testing.assert_allclose(states_before_step, [
        (spec.y, spec.velocity_y, spec.last_force_y) for spec in (front, rear)])


def test_adding_second_free_cylinder_keeps_primary_mask():
    grid = CartesianGrid(48, 32, lx=8.0, ly=4.0)
    ibm = ImmersedBoundary(grid)
    ibm.add_free_y_circle(2.0, 2.0, 0.4)
    primary_mask = ibm.mask_u.copy()
    ibm.add_free_y_circle(4.4, 2.0, 0.4)
    assert np.all(ibm.mask_u[primary_mask])


@pytest.mark.parametrize("master", [False, True])
def test_tandem_monitor_experimental_gate(tmp_path, monkeypatch, master):
    parse_case(tmp_path, monkeypatch, master=master)
    with (tmp_path / "experimental_config.txt").open("a") as stream:
        stream.write("cylinder_downstream_monitor_velocity_squared = true\n")
    assert main.parse_args().cylinder_downstream_monitor_velocity_squared is master
