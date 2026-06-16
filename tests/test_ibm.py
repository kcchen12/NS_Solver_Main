"""Tests for the immersed boundary method."""
import numpy as np
import pytest
from src.grid import CartesianGrid
from src.ibm  import ImmersedBoundary


class TestIBMCircle:
    def setup_method(self):
        self.g = CartesianGrid(16, 12, lx=2.0, ly=1.5)
        self.ibm = ImmersedBoundary(self.g)

    def test_no_solid_by_default(self):
        assert not self.ibm.has_solid

    def test_add_circle_marks_solid(self):
        self.ibm.add_circle(1.0, 0.75, 0.2)
        assert self.ibm.has_solid

    def test_circle_centre_is_solid(self):
        cx, cy, r = 1.0, 0.75, 0.3
        self.ibm.add_circle(cx, cy, r)
        g = self.g
        # Find u-face index closest to centre
        i_c = int(np.argmin(np.abs(g.xf - cx)))
        j_c = int(np.argmin(np.abs(g.yc - cy)))
        assert self.ibm.mask_u[i_c, j_c]

    def test_apply_zeros_solid_cells(self):
        self.ibm.add_circle(1.0, 0.75, 0.2)
        u = np.ones(self.g.u_shape)
        v = np.ones(self.g.v_shape)
        self.ibm.apply(u, v)
        # All solid u-faces must be 0
        assert np.all(u[self.ibm.mask_u] == 0.0)
        assert np.all(v[self.ibm.mask_v] == 0.0)

    def test_fluid_cells_unchanged(self):
        self.ibm.add_circle(1.0, 0.75, 0.1)
        u = np.ones(self.g.u_shape) * 2.0
        v = np.ones(self.g.v_shape) * 3.0
        self.ibm.apply(u, v)
        # Fluid u-faces must remain at original value
        assert np.all(u[~self.ibm.mask_u] == 2.0)
        assert np.all(v[~self.ibm.mask_v] == 3.0)

    def test_rotating_circle_imposes_tangential_velocity(self):
        cx, cy, r = 1.0, 0.75, 0.3
        omega = 2.5
        self.ibm.add_rotating_circle(
            cx, cy, r, omega_amplitude=omega, frequency=0.0, phase=np.pi / 2.0
        )
        u = np.zeros(self.g.u_shape)
        v = np.zeros(self.g.v_shape)
        self.ibm.apply(u, v, time=0.0)

        i_u = int(np.argmin(np.abs(self.g.xf - cx)))
        j_u = int(np.argmin(np.abs(self.g.yc - (cy + 0.1))))
        assert self.ibm.mask_u[i_u, j_u]
        assert np.isclose(u[i_u, j_u], -omega * (self.g.yc[j_u] - cy))

        i_v = int(np.argmin(np.abs(self.g.xc - (cx + 0.1))))
        j_v = int(np.argmin(np.abs(self.g.yf - cy)))
        assert self.ibm.mask_v[i_v, j_v]
        assert np.isclose(v[i_v, j_v], omega * (self.g.xc[i_v] - cx))

    def test_constant_rotating_circle_imposes_tangential_velocity(self):
        cx, cy, r = 1.0, 0.75, 0.3
        omega = 1.75
        self.ibm.add_constant_rotating_circle(cx, cy, r, omega=omega)
        u = np.zeros(self.g.u_shape)
        v = np.zeros(self.g.v_shape)
        self.ibm.apply(u, v, time=2.0)

        i_u = int(np.argmin(np.abs(self.g.xf - cx)))
        j_u = int(np.argmin(np.abs(self.g.yc - (cy + 0.1))))
        assert self.ibm.mask_u[i_u, j_u]
        assert np.isclose(u[i_u, j_u], -omega * (self.g.yc[j_u] - cy))

        i_v = int(np.argmin(np.abs(self.g.xc - (cx + 0.1))))
        j_v = int(np.argmin(np.abs(self.g.yf - cy)))
        assert self.ibm.mask_v[i_v, j_v]
        assert np.isclose(v[i_v, j_v], omega * (self.g.xc[i_v] - cx))

    def test_translating_circle_imposes_linear_velocity_and_moves_mask(self):
        cx, cy, r = 1.0, 0.75, 0.2
        amplitude_x = 0.1
        frequency = 0.5
        self.ibm.add_translating_circle(
            cx,
            cy,
            r,
            amplitude_x=amplitude_x,
            amplitude_y=0.0,
            frequency=frequency,
            phase=0.0,
        )
        u = np.zeros(self.g.u_shape)
        v = np.zeros(self.g.v_shape)
        self.ibm.apply(u, v, time=0.0)

        expected_u = 2.0 * np.pi * frequency * amplitude_x
        assert np.allclose(u[self.ibm.mask_u], expected_u)
        assert np.allclose(v[self.ibm.mask_v], 0.0)

        self.ibm.apply(u, v, time=0.5)
        moved_cx = cx + amplitude_x
        i_u = int(np.argmin(np.abs(self.g.xf - moved_cx)))
        j_u = int(np.argmin(np.abs(self.g.yc - cy)))
        assert self.ibm.mask_u[i_u, j_u]

    def test_translating_circle_supports_vertical_motion(self):
        cx, cy, r = 1.0, 0.75, 0.2
        amplitude_y = 0.075
        frequency = 0.25
        self.ibm.add_translating_circle(
            cx,
            cy,
            r,
            amplitude_x=0.0,
            amplitude_y=amplitude_y,
            frequency=frequency,
            phase=0.0,
        )
        u = np.zeros(self.g.u_shape)
        v = np.zeros(self.g.v_shape)
        self.ibm.apply(u, v, time=0.0)

        expected_v = 2.0 * np.pi * frequency * amplitude_y
        assert np.allclose(u[self.ibm.mask_u], 0.0)
        assert np.allclose(v[self.ibm.mask_v], expected_v)

    def test_translating_force_diagnostic_uses_regularized_circle_weight(self):
        cx, cy, r = 1.0, 0.75, 0.22
        self.ibm.add_translating_circle(
            cx,
            cy,
            r,
            amplitude_x=0.0,
            amplitude_y=0.0,
            frequency=0.0,
        )
        u = np.ones(self.g.u_shape)
        v = np.ones(self.g.v_shape) * 2.0

        force_x, force_y = self.ibm.apply(u, v, dt=1.0, rho=1.0, time=0.0)

        weight_u = self.ibm._circle_force_weight(
            self.ibm._u_x,
            self.ibm._u_y,
            cx,
            cy,
            r,
            self.ibm._force_regularization_width,
        )
        weight_v = self.ibm._circle_force_weight(
            self.ibm._v_x,
            self.ibm._v_y,
            cx,
            cy,
            r,
            self.ibm._force_regularization_width,
        )
        expected_fx = np.sum(self.ibm._u_face_measure * weight_u)
        expected_fy = np.sum(2.0 * self.ibm._v_face_measure * weight_v)

        assert np.isclose(force_x, expected_fx)
        assert np.isclose(force_y, expected_fy)

    def test_force_diagnostic_uses_local_face_measures_on_nonuniform_grid(self):
        xf = np.array([0.0, 0.2, 0.6, 1.0], dtype=float)
        yf = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)
        g = CartesianGrid(3, 3, lx=1.0, ly=1.0, xf=xf, yf=yf)
        ibm = ImmersedBoundary(g)

        mask_u = np.zeros(g.u_shape, dtype=bool)
        mask_v = np.zeros(g.v_shape, dtype=bool)
        mask_u[1, 0] = True
        mask_u[2, 2] = True
        mask_v[0, 1] = True
        mask_v[2, 2] = True
        ibm.add_mask(mask_u, mask_v)

        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        u[1, 0] = 2.0
        u[2, 2] = 3.0
        v[0, 1] = 5.0
        v[2, 2] = 7.0

        force_x, force_y = ibm.apply(u, v, dt=0.5, rho=1.0, time=0.0)

        expected_fx = (2.0 * g.dy_cells[0] + 3.0 * g.dy_cells[2]) / 0.5
        expected_fy = (5.0 * g.dx_cells[0] + 7.0 * g.dx_cells[2]) / 0.5
        assert np.isclose(force_x, expected_fx)
        assert np.isclose(force_y, expected_fy)


class TestIBMRectangle:
    def test_add_rectangle(self):
        g = CartesianGrid(8, 6, lx=1.0, ly=1.0)
        ibm = ImmersedBoundary(g)
        ibm.add_rectangle(0.2, 0.4, 0.3, 0.7)
        assert ibm.has_solid

    def test_rectangle_interior_solid(self):
        g = CartesianGrid(20, 20, lx=1.0, ly=1.0)
        ibm = ImmersedBoundary(g)
        x0, x1, y0, y1 = 0.3, 0.7, 0.3, 0.7
        ibm.add_rectangle(x0, x1, y0, y1)
        # u-face at centre of rectangle should be solid
        i_c = int(np.argmin(np.abs(g.xf - 0.5)))
        j_c = int(np.argmin(np.abs(g.yc - 0.5)))
        assert ibm.mask_u[i_c, j_c]

    def test_add_mask_rejects_wrong_shape(self):
        g = CartesianGrid(8, 6, lx=1.0, ly=1.0)
        ibm = ImmersedBoundary(g)
        with pytest.raises(ValueError):
            ibm.add_mask(np.zeros((1, 1), dtype=bool), np.zeros(g.v_shape, dtype=bool))
        with pytest.raises(ValueError):
            ibm.add_mask(np.zeros(g.u_shape, dtype=bool), np.zeros((1, 1), dtype=bool))


class TestIBMIndentedCircle:
    def test_top_indent_clears_top_center(self):
        g = CartesianGrid(80, 80, lx=2.0, ly=2.0)
        ibm = ImmersedBoundary(g)
        cx, cy, r = 1.0, 1.0, 0.5
        ibm.add_circle_with_top_indent(
            cx=cx,
            cy=cy,
            radius=r,
            indent_width=0.3,
            indent_depth=0.2,
        )

        i_center = int(np.argmin(np.abs(g.xf - cx)))
        j_notch = int(np.argmin(np.abs(g.yc - (cy + r - 0.1))))
        j_body = int(np.argmin(np.abs(g.yc - cy)))

        assert not ibm.mask_u[i_center, j_notch]
        assert ibm.mask_u[i_center, j_body]

    def test_invalid_indent_rejected(self):
        g = CartesianGrid(20, 20, lx=2.0, ly=2.0)
        ibm = ImmersedBoundary(g)
        with pytest.raises(ValueError):
            ibm.add_circle_with_top_indent(
                cx=1.0,
                cy=1.0,
                radius=0.3,
                indent_width=0.7,
                indent_depth=0.1,
            )


