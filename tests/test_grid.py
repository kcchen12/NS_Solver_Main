"""Tests for CartesianGrid."""
import numpy as np
import pytest
from src.grid import (
    CartesianGrid,
    build_nonuniform_grid_metadata,
    stretched_faces_center_uniform,
    stretched_faces_tanh,
)
from src.io_utils import load_grid_metadata, load_grid_metadata_dict, save_grid_metadata, save_grid_metadata_dict


class TestCartesianGrid2D:
    def test_cell_count(self):
        g = CartesianGrid(8, 4, lx=2.0, ly=1.0)
        assert g.nx == 8
        assert g.ny == 4
        assert not g.is_3d

    def test_spacing(self):
        g = CartesianGrid(8, 4, lx=2.0, ly=1.0)
        assert np.isclose(g.dx, 0.25)
        assert np.isclose(g.dy, 0.25)

    def test_face_coordinates(self):
        g = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        # x-faces: 0, 0.25, 0.5, 0.75, 1.0
        assert np.allclose(g.xf, [0.0, 0.25, 0.5, 0.75, 1.0])
        # y-faces: 0, 0.5, 1.0
        assert np.allclose(g.yf, [0.0, 0.5, 1.0])

    def test_cell_centre_coordinates(self):
        g = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        assert np.allclose(g.xc, [0.125, 0.375, 0.625, 0.875])
        assert np.allclose(g.yc, [0.25, 0.75])

    def test_p_shape(self):
        g = CartesianGrid(8, 4)
        assert g.p_shape == (8, 4)

    def test_u_shape(self):
        g = CartesianGrid(8, 4)
        assert g.u_shape == (9, 4)

    def test_v_shape(self):
        g = CartesianGrid(8, 4)
        assert g.v_shape == (8, 5)

    def test_zeros_u(self):
        g = CartesianGrid(4, 3)
        u = g.zeros_u()
        assert u.shape == (5, 3)
        assert np.all(u == 0.0)

    def test_zeros_v(self):
        g = CartesianGrid(4, 3)
        v = g.zeros_v()
        assert v.shape == (4, 4)
        assert np.all(v == 0.0)

    def test_zeros_p(self):
        g = CartesianGrid(4, 3)
        p = g.zeros_p()
        assert p.shape == (4, 3)

    def test_metadata_contains_coordinates(self):
        g = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        meta = g.to_metadata()
        assert meta["nx"] == 4
        assert meta["ny"] == 2
        assert np.allclose(meta["xf"], [0.0, 0.25, 0.5, 0.75, 1.0])
        assert np.allclose(meta["yc"], [0.25, 0.75])

    def test_grid_metadata_round_trip(self, tmp_path):
        path = tmp_path / "uniform_grid.npz"
        original = CartesianGrid(4, 2, lx=1.0, ly=1.0)
        save_grid_metadata(str(path), original)
        loaded = load_grid_metadata(str(path))
        assert loaded.nx == original.nx
        assert loaded.ny == original.ny
        assert np.isclose(loaded.lx, original.lx)
        assert np.isclose(loaded.ly, original.ly)
        assert np.allclose(loaded.xf, original.xf)
        assert np.allclose(loaded.yc, original.yc)

    def test_custom_domain_bounds(self):
        g = CartesianGrid(4, 2, lx=4.0, ly=2.0, x_min=-2.0, y_min=-1.0)
        assert np.isclose(g.x_min, -2.0)
        assert np.isclose(g.x_max, 2.0)
        assert np.isclose(g.y_min, -1.0)
        assert np.isclose(g.y_max, 1.0)
        assert np.allclose(g.xf, [-2.0, -1.0, 0.0, 1.0, 2.0])
        assert np.allclose(g.yf, [-1.0, 0.0, 1.0])


class TestCartesianGrid3D:
    def test_3d_flag(self):
        g = CartesianGrid(4, 3, nz=2)
        assert g.is_3d

    def test_3d_shapes(self):
        g = CartesianGrid(4, 3, nz=2)
        assert g.p_shape == (4, 3, 2)
        assert g.u_shape == (5, 3, 2)
        assert g.v_shape == (4, 4, 2)
        assert g.w_shape == (4, 3, 3)

    def test_3d_w_raises_in_2d(self):
        g = CartesianGrid(4, 3)
        with pytest.raises(RuntimeError):
            g.zeros_w()


class TestPreparedNonuniformGrid:
    def test_stretched_faces_tanh_uniform_when_beta_nonpositive(self):
        xf = stretched_faces_tanh(4, 1.0, beta=0.0)
        assert np.allclose(xf, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_center_uniform_has_uniform_core_and_tanh_outer_stretch(self):
        xf, core_start, core_end = stretched_faces_center_uniform(
            80, 8.0, beta=2.0, core_interval=(2.0, 6.0)
        )
        dx = np.diff(xf)
        xc = 0.5 * (xf[:-1] + xf[1:])
        core_mask = (xc >= core_start) & (xc <= core_end)
        left_mask = xc < core_start
        right_mask = xc > core_end

        assert np.all(dx > 0.0)
        assert np.allclose(dx[core_mask], dx[core_mask]
                           [0], rtol=1e-10, atol=1e-12)
        assert dx[core_mask][0] < 8.0 / 80.0
        assert dx[left_mask][0] > dx[left_mask][-1]
        assert dx[right_mask][-1] > dx[right_mask][0]

    def test_nonuniform_metadata_has_expected_shapes(self):
        meta = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.5,
            uniform_x_start=0.5,
            uniform_x_end=1.5,
            uniform_y_start=0.25,
            uniform_y_end=0.75,
        )
        assert meta["grid_type"] == "nonuniform"
        assert meta["xf"].shape == (9,)
        assert meta["yf"].shape == (5,)
        assert meta["dx"].shape == (8,)
        assert meta["dy"].shape == (4,)
        assert meta["nonuniform_mode"] == "center-uniform"
        assert np.isclose(meta["xf"][0], 0.0)
        assert np.isclose(meta["xf"][-1], 2.0)

    def test_center_uniform_metadata_tracks_mode_and_core_bounds(self):
        meta = build_nonuniform_grid_metadata(
            nx=80,
            ny=40,
            lx=8.0,
            ly=4.0,
            beta_x=2.0,
            beta_y=1.5,
            uniform_x_start=2.0,
            uniform_x_end=6.0,
            uniform_y_start=1.2,
            uniform_y_end=2.8,
        )
        dx = np.asarray(meta["dx"])
        dy = np.asarray(meta["dy"])
        xc = np.asarray(meta["xc"])
        yc = np.asarray(meta["yc"])
        x_core_mask = (xc >= meta["band_start_x"]) & (xc <= meta["band_end_x"])
        y_core_mask = (yc >= meta["band_start_y"]) & (yc <= meta["band_end_y"])

        assert meta["nonuniform_mode"] == "center-uniform"
        assert np.isclose(meta["uniform_x_start"], 2.0)
        assert np.isclose(meta["uniform_x_end"], 6.0)
        assert np.isclose(meta["uniform_y_start"], 1.2)
        assert np.isclose(meta["uniform_y_end"], 2.8)
        assert np.allclose(dx[x_core_mask], dx[x_core_mask]
                           [0], rtol=1e-10, atol=1e-12)
        assert np.allclose(dy[y_core_mask], dy[y_core_mask]
                           [0], rtol=1e-10, atol=1e-12)
        assert dx[x_core_mask][0] < 8.0 / 80.0
        assert dy[y_core_mask][0] < 4.0 / 40.0

    def test_center_uniform_metadata_accepts_explicit_core_ranges(self):
        meta = build_nonuniform_grid_metadata(
            nx=160,
            ny=128,
            lx=25.0,
            ly=80.0,
            x_min=-10.0,
            y_min=-40.0,
            beta_x=2.0,
            beta_y=2.0,
            uniform_x_start=-2.0,
            uniform_x_end=7.5,
            uniform_y_start=-8.0,
            uniform_y_end=8.0,
        )
        assert np.isclose(meta["band_start_x"], -2.0)
        assert np.isclose(meta["band_end_x"], 7.5)
        assert np.isclose(meta["band_start_y"], -8.0)
        assert np.isclose(meta["band_end_y"], 8.0)

    def test_center_uniform_y_is_symmetric_when_domain_crosses_zero(self):
        meta = build_nonuniform_grid_metadata(
            nx=80,
            ny=128,
            lx=8.0,
            ly=80.0,
            x_min=-1.0,
            y_min=-40.0,
            beta_x=2.0,
            beta_y=2.0,
            uniform_x_start=1.0,
            uniform_x_end=4.0,
            uniform_y_start=-16.0,
            uniform_y_end=16.0,
        )
        yf = np.asarray(meta["yf"])
        dy = np.asarray(meta["dy"])
        assert np.allclose(yf, -yf[::-1], rtol=1e-12, atol=1e-12)
        assert np.allclose(dy[: dy.size // 2], dy[::-1]
                           [: dy.size // 2], rtol=1e-12, atol=1e-12)

    def test_center_uniform_transition_is_continuous_at_core_edges(self):
        meta = build_nonuniform_grid_metadata(
            nx=200,
            ny=80,
            lx=25.0,
            ly=80.0,
            x_min=-10.0,
            y_min=-40.0,
            beta_x=2.0,
            beta_y=2.0,
            uniform_x_start=-2.0,
            uniform_x_end=7.5,
            uniform_y_start=-8.0,
            uniform_y_end=8.0,
        )
        xc = np.asarray(meta["xc"])
        dx = np.asarray(meta["dx"])
        core_mask = (xc >= meta["band_start_x"]) & (xc <= meta["band_end_x"])
        core_indices = np.flatnonzero(core_mask)
        left_core = int(core_indices[0])
        right_core = int(core_indices[-1])

        assert left_core > 0
        assert right_core < dx.size - 1
        assert dx[left_core - 1] >= 0.98 * dx[left_core]
        assert dx[right_core + 1] >= 0.98 * dx[right_core]

    def test_nonuniform_metadata_respects_custom_bounds(self):
        meta = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.5,
            x_min=-1.0,
            y_min=-0.5,
            uniform_x_start=-0.5,
            uniform_x_end=0.5,
            uniform_y_start=-0.25,
            uniform_y_end=0.25,
        )
        assert np.isclose(meta["x_min"], -1.0)
        assert np.isclose(meta["x_max"], 1.0)
        assert np.isclose(meta["y_min"], -0.5)
        assert np.isclose(meta["y_max"], 0.5)
        assert np.isclose(meta["xf"][0], -1.0)
        assert np.isclose(meta["xf"][-1], 1.0)

    def test_nonuniform_grid_concentrates_cells_in_uniform_core(self):
        meta = build_nonuniform_grid_metadata(
            nx=90,
            ny=30,
            lx=9.0,
            ly=3.0,
            beta_x=2.0,
            beta_y=2.0,
            uniform_x_start=3.0,
            uniform_x_end=6.0,
            uniform_y_start=1.0,
            uniform_y_end=2.0,
        )
        xc = meta["xc"]
        dx = meta["dx"]
        core_mask = (xc >= meta["band_start_x"]) & (xc <= meta["band_end_x"])
        outer_mask = ~core_mask
        assert np.mean(dx[core_mask]) < np.mean(dx[outer_mask])

    def test_nonuniform_grid_outside_spacing_stays_close_to_uniform(self):
        meta = build_nonuniform_grid_metadata(
            nx=90,
            ny=30,
            lx=9.0,
            ly=3.0,
            beta_x=2.0,
            beta_y=2.0,
            uniform_x_start=3.0,
            uniform_x_end=6.0,
            uniform_y_start=1.0,
            uniform_y_end=2.0,
        )
        xc = meta["xc"]
        dx = meta["dx"]
        uniform_dx = 9.0 / 90.0
        outer_mask = (xc < meta["band_start_x"]) | (xc > meta["band_end_x"])
        assert np.mean(dx[outer_mask]) == pytest.approx(uniform_dx, rel=0.25)

    def test_nonuniform_metadata_round_trip_dict_loader(self, tmp_path):
        path = tmp_path / "nonuniform_grid.npz"
        original = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.0,
            uniform_x_start=0.5,
            uniform_x_end=1.5,
            uniform_y_start=0.25,
            uniform_y_end=0.75,
        )
        save_grid_metadata_dict(str(path), original)
        loaded = load_grid_metadata_dict(str(path))
        assert loaded["grid_type"] == "nonuniform"
        assert np.allclose(loaded["xf"], original["xf"])
        assert np.allclose(loaded["dy"], original["dy"])

    def test_uniform_loader_rejects_nonuniform_metadata(self, tmp_path):
        path = tmp_path / "nonuniform_grid.npz"
        meta = build_nonuniform_grid_metadata(
            nx=8,
            ny=4,
            lx=2.0,
            ly=1.0,
            beta_x=2.0,
            beta_y=1.0,
            uniform_x_start=0.5,
            uniform_x_end=1.5,
            uniform_y_start=0.25,
            uniform_y_end=0.75,
        )
        save_grid_metadata_dict(str(path), meta)
        with pytest.raises(ValueError):
            load_grid_metadata(str(path))
