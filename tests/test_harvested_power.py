import numpy as np

from src.harvested_power import harvested_power_report_lines


def metadata(path, t=2.0, **extra):
    np.savez(path, t=t, meta_cylinder_free_y_damping=2.0,
             meta_cylinder_free_y_dof=True,
             meta_cylinder_downstream_enabled=True,
             meta_cylinder_free_y_velocity=1.0,
             meta_cylinder_downstream_velocity_y=2.0, **extra)


def test_tandem_monitor_integrates_nonuniform_time(tmp_path):
    metadata(tmp_path / "snap_002.npz")
    (tmp_path / "tandem_y_velocity_squared.csv").write_text(
        "t,primary_velocity_y_squared,downstream_velocity_y_squared\n"
        "0,0,0\n0.5,1,4\n2,4,16\n")
    report = "\n".join(harvested_power_report_lines(tmp_path))
    assert "Upstream energy     : 8" in report
    assert "Downstream energy     : 32" in report
    assert "Total energy     : 40" in report
    assert "Total mean power : 20" in report
    assert "Total peak power : 40" in report
    assert "every-step velocity monitor" in report


def test_snapshot_fallback_includes_both_cylinders(tmp_path):
    metadata(tmp_path / "snap_000.npz", t=0)
    metadata(tmp_path / "snap_002.npz")
    report = "\n".join(harvested_power_report_lines(tmp_path))
    assert "Total energy     : 20" in report
    assert "Total mean power : 10" in report
    assert "saved snapshots" in report


def test_missing_motion_is_explicitly_unavailable(tmp_path):
    report = "\n".join(harvested_power_report_lines(tmp_path))
    assert "Unavailable" in report
    assert "Total mean power" not in report


def test_duplicate_times_are_rejected(tmp_path):
    metadata(tmp_path / "snap_002.npz")
    (tmp_path / "tandem_y_velocity_squared.csv").write_text(
        "t,primary_velocity_y_squared,downstream_velocity_y_squared\n"
        "0,1,4\n0,1,4\n")
    report = "\n".join(harvested_power_report_lines(tmp_path))
    assert "Unavailable" in report


def test_single_cylinder_monitor(tmp_path):
    np.savez(tmp_path / "snap_002.npz", meta_cylinder_free_y_damping=2.0,
             meta_cylinder_free_y_dof=True)
    (tmp_path / "y_velocity_squared.csv").write_text("t,velocity_y_squared\n0,1\n2,1\n")
    report = "\n".join(harvested_power_report_lines(tmp_path))
    assert "Primary energy     : 4" in report
    assert "Total energy     : 4" in report
    assert "Downstream" not in report
