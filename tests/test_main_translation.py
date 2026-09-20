"""Tests for cylinder translation configuration helpers."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from main import (
    _apply_y_truncation_to_args,
    _airfoil_outline_points,
    _normalize_cylinder_experiment_mode,
    _normalize_cylinder_geometry_mode,
    _parse_polygon_points,
    _normalize_cylinder_translation_mode,
    parse_args,
    _plot_ibm_outline,
    _resolve_experiment_overrides,
    _resolve_cylinder_translation,
)


def _translation_args(**overrides):
    args = SimpleNamespace(
        cylinder_translation_mode="oscillatory-xy",
        cylinder_translation_x_percent=10.0,
        cylinder_translation_y_percent=5.0,
        cylinder_translation_frequency=0.25,
        cylinder_translation_phase_deg=0.0,
        grid_type="nonuniform",
        uniform_x_start=-1.0,
        uniform_x_end=1.0,
        uniform_y_start=-1.0,
        uniform_y_end=1.0,
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_translation_mode_aliases():
    assert _normalize_cylinder_translation_mode("left-right") == "oscillatory-x"
    assert _normalize_cylinder_translation_mode("up_down") == "oscillatory-y"
    assert _normalize_cylinder_translation_mode("both") == "oscillatory-xy"


def test_square_bluff_body_aliases():
    assert _normalize_cylinder_experiment_mode("none") == "circle"
    assert _normalize_cylinder_experiment_mode("square-bluff-body") == "square"
    assert _normalize_cylinder_geometry_mode("square-cylinder") == "square"
    assert _normalize_cylinder_experiment_mode("naca0012") == "airfoil"
    assert _normalize_cylinder_geometry_mode("naca-0012") == "airfoil"
    assert _normalize_cylinder_experiment_mode("custom") == "polygon"
    assert _normalize_cylinder_geometry_mode("coordinates") == "polygon"


def test_square_experiment_overrides_geometry_mode():
    args = argparse.Namespace(
        cylinder_experiment="square",
        cylinder_geometry_mode="circle",
        ibm_shape="circle",
    )

    assert _resolve_experiment_overrides(args) == "square"


def test_airfoil_experiment_overrides_geometry_mode():
    args = argparse.Namespace(
        cylinder_experiment="airfoil",
        cylinder_geometry_mode="circle",
        ibm_shape="circle",
    )

    assert _resolve_experiment_overrides(args) == "airfoil"


def test_polygon_coordinate_parser():
    points = _parse_polygon_points("-0.3,-0.2; 0.3,-0.1; 0.2,0.25")

    assert points.shape == (3, 2)
    assert points[0, 0] == pytest.approx(-0.3)
    assert points[2, 1] == pytest.approx(0.25)


def test_polygon_coordinate_parser_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least three"):
        _parse_polygon_points("0,0; 1,0")


def test_translation_percentages_resolve_from_diameter():
    cfg = _resolve_cylinder_translation(
        _translation_args(),
        cx=0.0,
        cy=0.0,
        radius=0.5,
    )

    assert cfg["mode"] == "oscillatory-xy"
    assert cfg["amplitude_x"] == pytest.approx(0.1)
    assert cfg["amplitude_y"] == pytest.approx(0.05)


def test_translation_sweep_must_stay_inside_concentrated_core():
    args = _translation_args(uniform_x_start=-0.55, uniform_x_end=0.55)

    with pytest.raises(ValueError, match="concentrated mesh core"):
        _resolve_cylinder_translation(args, cx=0.0, cy=0.0, radius=0.5)


def _truncation_args(**overrides):
    args = SimpleNamespace(
        truncate_y_cells=10,
        grid_type="uniform",
        nx=16,
        ny=64,
        lx=8.0,
        ly=4.0,
        x_min=0.0,
        x_max=8.0,
        y_min=-2.0,
        y_max=2.0,
        beta_x=2.0,
        beta_y=2.0,
        uniform_x_start=1.0,
        uniform_x_end=4.0,
        uniform_y_start=-0.75,
        uniform_y_end=0.75,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_y_truncation_drops_cells_from_uniform_grid_bounds():
    args = _truncation_args(grid_type="uniform")

    _apply_y_truncation_to_args(args, argparse.ArgumentParser())

    assert args.ny == 44
    assert args.y_truncation_enabled
    assert args.source_ny_before_y_truncation == 64
    assert args.y_min == pytest.approx(-1.375)
    assert args.y_max == pytest.approx(1.375)
    assert args.ly == pytest.approx(2.75)
    assert np.allclose(args._truncated_yf, np.linspace(-2.0, 2.0, 65)[10:-10])


def test_y_truncation_preserves_nonuniform_reference_faces():
    args = _truncation_args(grid_type="nonuniform")

    _apply_y_truncation_to_args(args, argparse.ArgumentParser())

    assert args.ny == 44
    assert args.y_truncation_enabled
    assert args.y_min == pytest.approx(args._truncated_yf[0])
    assert args.y_max == pytest.approx(args._truncated_yf[-1])
    assert np.all(np.diff(args._truncated_yf) > 0.0)


def _write_minimal_configs(tmp_path, experiment_text: str):
    config_path = tmp_path / "config.txt"
    experiment_path = tmp_path / "experimental_config.txt"
    post_path = tmp_path / "post_config.txt"
    config_path.write_text(
        "\n".join(
            [
                "nx = 16",
                "ny = 64",
                "lx = 8.0",
                "ly = 4.0",
                "x_min = 0.0",
                "x_max = 8.0",
                "y_min = -2.0",
                "y_max = 2.0",
                "uniform_grid = true",
            ]
        ),
        encoding="utf-8",
    )
    experiment_path.write_text(experiment_text, encoding="utf-8")
    post_path.write_text("", encoding="utf-8")
    return config_path, experiment_path, post_path


def test_disabled_experimental_config_ignores_experimental_values(tmp_path, monkeypatch):
    config_path, experiment_path, post_path = _write_minimal_configs(
        tmp_path,
        "\n".join(
            [
                "enable_experimental_config = false",
                "cylinder_experiment = top-indent",
                "cylinder_indent_width = 0.25",
                "cylinder_indent_depth = 0.10",
                "truncate_y_cells = 10",
                "cylinder_free_y_dof = true",
                "cylinder_free_y_mass = 2.0",
                "cylinder_free_y_damping = 0.3",
                "cylinder_free_y_stiffness = 4.0",
                "cylinder_free_y_initial_velocity = 0.1",
                "cylinder_free_y_force_relaxation = 0.2",
                "cylinder_free_y_max_displacement_percent = 15.0",
                "cylinder_free_y_max_speed = 0.4",
            ]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--experiment-config",
            str(experiment_path),
            "--post-config",
            str(post_path),
        ],
    )

    args = parse_args()

    assert not args.experimental_config_enabled
    assert not args.cylinder_free_x_dof
    assert args.cylinder_free_x_mass == pytest.approx(100.0)
    assert args.cylinder_free_x_damping == pytest.approx(5.0)
    assert args.cylinder_free_x_stiffness == pytest.approx(20.0)
    assert args.cylinder_free_x_initial_velocity == pytest.approx(0.0)
    assert args.cylinder_free_x_force_relaxation == pytest.approx(0.05)
    assert args.cylinder_free_x_max_displacement_percent == pytest.approx(25.0)
    assert args.cylinder_free_x_max_speed == pytest.approx(0.25)
    assert args.cylinder_free_x_release_time == pytest.approx(0.0)
    assert args.cylinder_experiment == "circle"
    assert args.cylinder_indent_width == 0.0
    assert args.cylinder_indent_depth == 0.0
    assert args.truncate_y_cells == 0
    assert not args.cylinder_free_y_dof
    assert args.cylinder_free_y_mass == pytest.approx(100.0)
    assert args.cylinder_free_y_damping == pytest.approx(5.0)
    assert args.cylinder_free_y_stiffness == pytest.approx(20.0)
    assert args.cylinder_free_y_initial_velocity == pytest.approx(0.0)
    assert args.cylinder_free_y_force_relaxation == pytest.approx(0.05)
    assert args.cylinder_free_y_max_displacement_percent == pytest.approx(25.0)
    assert args.cylinder_free_y_max_speed == pytest.approx(0.25)
    assert args.cylinder_free_y_release_time == pytest.approx(0.0)
    assert not args.cylinder_free_theta_dof
    assert args.cylinder_free_theta_inertia == pytest.approx(10.0)
    assert args.cylinder_free_theta_damping == pytest.approx(1.0)
    assert args.cylinder_free_theta_stiffness == pytest.approx(5.0)
    assert args.cylinder_free_theta_initial_angle_deg == pytest.approx(0.0)
    assert args.cylinder_free_theta_initial_angular_velocity == pytest.approx(0.0)
    assert args.cylinder_free_theta_moment_relaxation == pytest.approx(0.05)
    assert args.cylinder_free_theta_max_angle_deg == pytest.approx(45.0)
    assert args.cylinder_free_theta_max_angular_speed == pytest.approx(1.0)
    assert args.ny == 64
    assert args.y_min == pytest.approx(-2.0)
    assert args.y_max == pytest.approx(2.0)


def test_enabled_experimental_config_reads_experimental_values(tmp_path, monkeypatch):
    config_path, experiment_path, post_path = _write_minimal_configs(
        tmp_path,
        "\n".join(
            [
                "enable_experimental_config = true",
                "cylinder_experiment = top-indent",
                "cylinder_indent_width = 0.25",
                "cylinder_indent_depth = 0.10",
                "truncate_y_cells = 10",
                "cylinder_free_x_dof = true",
                "cylinder_free_x_mass = 2.0",
                "cylinder_free_x_damping = 0.3",
                "cylinder_free_x_stiffness = 4.0",
                "cylinder_free_x_initial_velocity = 0.05",
                "cylinder_free_x_force_relaxation = 0.2",
                "cylinder_free_x_max_displacement_percent = 15.0",
                "cylinder_free_x_max_speed = 0.4",
                "cylinder_free_x_release_time = 100.0",
                "cylinder_free_y_dof = true",
                "cylinder_free_y_mass = 2.0",
                "cylinder_free_y_damping = 0.3",
                "cylinder_free_y_stiffness = 4.0",
                "cylinder_free_y_initial_velocity = 0.1",
                "cylinder_free_y_force_relaxation = 0.2",
                "cylinder_free_y_max_displacement_percent = 15.0",
                "cylinder_free_y_max_speed = 0.4",
                "cylinder_free_y_release_time = 120.0",
                "cylinder_free_theta_dof = true",
                "cylinder_free_theta_inertia = 3.0",
                "cylinder_free_theta_damping = 0.7",
                "cylinder_free_theta_stiffness = 6.0",
                "cylinder_free_theta_initial_angle_deg = 2.5",
                "cylinder_free_theta_initial_angular_velocity = 0.06",
                "cylinder_free_theta_moment_relaxation = 0.25",
                "cylinder_free_theta_max_angle_deg = 20.0",
                "cylinder_free_theta_max_angular_speed = 0.8",
            ]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--experiment-config",
            str(experiment_path),
            "--post-config",
            str(post_path),
        ],
    )

    args = parse_args()

    assert args.experimental_config_enabled
    assert args.cylinder_experiment == "top-indent"
    assert args.cylinder_indent_width == pytest.approx(0.25)
    assert args.cylinder_indent_depth == pytest.approx(0.10)
    assert args.truncate_y_cells == 10
    assert args.cylinder_free_x_dof
    assert args.cylinder_free_x_mass == pytest.approx(2.0)
    assert args.cylinder_free_x_damping == pytest.approx(0.3)
    assert args.cylinder_free_x_stiffness == pytest.approx(4.0)
    assert args.cylinder_free_x_initial_velocity == pytest.approx(0.05)
    assert args.cylinder_free_x_force_relaxation == pytest.approx(0.2)
    assert args.cylinder_free_x_max_displacement_percent == pytest.approx(15.0)
    assert args.cylinder_free_x_max_speed == pytest.approx(0.4)
    assert args.cylinder_free_x_release_time == pytest.approx(100.0)
    assert args.cylinder_free_y_dof
    assert args.cylinder_free_y_mass == pytest.approx(2.0)
    assert args.cylinder_free_y_damping == pytest.approx(0.3)
    assert args.cylinder_free_y_stiffness == pytest.approx(4.0)
    assert args.cylinder_free_y_initial_velocity == pytest.approx(0.1)
    assert args.cylinder_free_y_force_relaxation == pytest.approx(0.2)
    assert args.cylinder_free_y_max_displacement_percent == pytest.approx(15.0)
    assert args.cylinder_free_y_max_speed == pytest.approx(0.4)
    assert args.cylinder_free_y_release_time == pytest.approx(120.0)
    assert args.cylinder_free_theta_dof
    assert args.cylinder_free_theta_inertia == pytest.approx(3.0)
    assert args.cylinder_free_theta_damping == pytest.approx(0.7)
    assert args.cylinder_free_theta_stiffness == pytest.approx(6.0)
    assert args.cylinder_free_theta_initial_angle_deg == pytest.approx(2.5)
    assert args.cylinder_free_theta_initial_angular_velocity == pytest.approx(0.06)
    assert args.cylinder_free_theta_moment_relaxation == pytest.approx(0.25)
    assert args.cylinder_free_theta_max_angle_deg == pytest.approx(20.0)
    assert args.cylinder_free_theta_max_angular_speed == pytest.approx(0.8)
    assert args.ny == 44


def test_enabled_experimental_config_allows_free_displacement_clamp_off(
    tmp_path,
    monkeypatch,
):
    config_path, experiment_path, post_path = _write_minimal_configs(
        tmp_path,
        "\n".join(
            [
                "enable_experimental_config = true",
                "cylinder_free_x_dof = true",
                "cylinder_free_x_max_displacement_percent = off",
                "cylinder_free_y_dof = true",
                "cylinder_free_y_max_displacement_percent = off",
            ]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--experiment-config",
            str(experiment_path),
            "--post-config",
            str(post_path),
        ],
    )

    args = parse_args()

    assert np.isinf(args.cylinder_free_x_max_displacement_percent)
    assert np.isinf(args.cylinder_free_y_max_displacement_percent)


def test_enabled_experimental_config_reads_square_body(tmp_path, monkeypatch):
    config_path, experiment_path, post_path = _write_minimal_configs(
        tmp_path,
        "\n".join(
            [
                "enable_experimental_config = true",
                "cylinder_experiment = square",
            ]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--experiment-config",
            str(experiment_path),
            "--post-config",
            str(post_path),
        ],
    )

    args = parse_args()

    assert args.cylinder_experiment == "square"
    assert _resolve_experiment_overrides(args) == "square"


def test_enabled_experimental_config_reads_airfoil_body(tmp_path, monkeypatch):
    config_path, experiment_path, post_path = _write_minimal_configs(
        tmp_path,
        "\n".join(
            [
                "enable_experimental_config = true",
                "cylinder_experiment = airfoil",
                "airfoil_chord = 1.25",
                "airfoil_thickness_percent = 15.0",
                "airfoil_angle_deg = 4.0",
            ]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--experiment-config",
            str(experiment_path),
            "--post-config",
            str(post_path),
        ],
    )

    args = parse_args()

    assert args.cylinder_experiment == "airfoil"
    assert _resolve_experiment_overrides(args) == "airfoil"
    assert args.airfoil_chord == pytest.approx(1.25)
    assert args.airfoil_thickness_percent == pytest.approx(15.0)
    assert args.airfoil_angle_deg == pytest.approx(4.0)


def test_square_outline_draws_closed_box():
    class Axis:
        def __init__(self):
            self.calls = []

        def plot(self, x, y, **kwargs):
            self.calls.append((list(x), list(y), kwargs))

    args = argparse.Namespace(
        cylinder_center_x=1.0,
        cylinder_center_y=2.0,
        cylinder_radius=0.5,
        cylinder_translation_mode="stationary",
        cylinder_translation_x_percent=0.0,
        cylinder_translation_y_percent=0.0,
        cylinder_translation_frequency=0.0,
        cylinder_translation_phase_deg=0.0,
        cylinder_experiment="square",
        cylinder_geometry_mode="circle",
        ibm_shape="circle",
    )
    ax = Axis()

    _plot_ibm_outline(ax, args)

    assert len(ax.calls) == 1
    x, y, _ = ax.calls[0]
    assert x == pytest.approx([0.5, 1.5, 1.5, 0.5, 0.5])
    assert y == pytest.approx([1.5, 1.5, 2.5, 2.5, 1.5])


def test_airfoil_outline_spans_chord():
    x, y = _airfoil_outline_points(
        cx=1.0,
        cy=2.0,
        chord=1.0,
        thickness_ratio=0.12,
        angle_deg=0.0,
        n_points=20,
    )

    assert np.min(x) == pytest.approx(0.5)
    assert np.max(x) == pytest.approx(1.5)
    assert np.max(y) > 2.0
    assert np.min(y) < 2.0
