"""Tests for cylinder translation configuration helpers."""

from types import SimpleNamespace

import pytest

from main import (
    _normalize_cylinder_translation_mode,
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
