"""Tests for boundary condition enforcement."""
import numpy as np
import pytest
from src.grid     import CartesianGrid
from src.boundary import (
    BoundaryConfig,
    BCType,
    FarfieldMode,
    apply_velocity_bc,
    apply_post_correction_bc,
    apply_pressure_bc,
)


def make_grid(nx=8, ny=6):
    return CartesianGrid(nx, ny, lx=2.0, ly=1.0)


class TestInflowBC:
    def test_left_inflow_sets_u(self):
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.5, v_inf=0.0)
        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(u[0, :], 1.5)

    def test_left_inflow_sets_v(self):
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.0, v_inf=0.2)
        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        # Wall BCs override the corners (j=0 and j=ny); check interior faces only
        assert np.allclose(v[0, 1:-1], 0.2)


class TestWallBC:
    def test_bottom_wall_no_penetration(self):
        """v[:, 0] = 0 at solid bottom wall."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        u = np.ones(g.u_shape)
        v = np.ones(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(v[:, 0], 0.0)

    def test_top_wall_no_penetration(self):
        """v[:, ny] = 0 at solid top wall."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        u = np.ones(g.u_shape)
        v = np.ones(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(v[:, g.ny], 0.0)


class TestConvectiveOutflow:
    def test_outflow_zero_gradient_fallback(self):
        """Without dt, outflow uses zero-gradient extrapolation."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.0)
        u = np.random.rand(*g.u_shape)
        v = np.zeros(g.v_shape)
        u_before = u[g.nx - 1, :].copy()
        apply_velocity_bc(u, v, g, bc, dt=None)
        assert np.allclose(u[g.nx, :], u_before)

    def test_outflow_with_dt(self):
        """With dt provided, outflow applies convective update."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL,
                            u_inf=1.0)
        u = np.ones(g.u_shape) * 1.0
        v = np.zeros(g.v_shape)
        dt = 0.01
        apply_velocity_bc(u, v, g, bc, dt=dt)
        # For uniform u=1, the convective update gives no change
        assert np.allclose(u[g.nx, :], 1.0, atol=1e-12)

    def test_outflow_uses_speed_magnitude_for_negative_reference_velocity(self):
        """Convective outflow should stay dissipative even for negative reference speed."""
        g = make_grid()
        bc = BoundaryConfig(
            left=BCType.INFLOW,
            right=BCType.OUTFLOW,
            bottom=BCType.WALL,
            top=BCType.OUTFLOW,
            u_inf=-2.0,
            v_inf=-3.0,
        )
        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        u[g.nx - 1, :] = 1.0
        u[g.nx, :] = 4.0
        v[:, g.ny - 1] = 2.0
        v[:, g.ny] = 5.0

        apply_velocity_bc(u, v, g, bc, dt=0.05)

        right_expected = 4.0 - (2.0 * 0.05 / g.dx_min) * (4.0 - 1.0)
        top_expected = 5.0 - (3.0 * 0.05 / g.dy_min) * (5.0 - 2.0)
        assert np.allclose(u[g.nx, :], right_expected)
        assert np.allclose(v[1:, g.ny], top_expected)


class TestPressureBC:
    def test_neumann_left(self):
        """Left Neumann: phi[0] = phi[1] for interior j."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        phi = np.random.rand(*g.p_shape)
        phi_interior = phi[1, :].copy()
        apply_pressure_bc(phi, g, bc)
        # Bottom/top Neumann BCs also modify phi[0,0] and phi[0,ny-1],
        # so check only interior j indices
        assert np.allclose(phi[0, 1:-1], phi_interior[1:-1])

    def test_dirichlet_right_outflow(self):
        """Right outflow (Dirichlet): phi[-1] = 0."""
        g = make_grid()
        bc = BoundaryConfig(left=BCType.INFLOW, right=BCType.OUTFLOW,
                            bottom=BCType.WALL, top=BCType.WALL)
        phi = np.random.rand(*g.p_shape) + 1.0
        apply_pressure_bc(phi, g, bc)
        assert np.allclose(phi[g.nx - 1, :], 0.0)


class TestFarfieldModes:
    def test_farfield_dirichlet_sets_prescribed_velocity(self):
        g = make_grid()
        bc = BoundaryConfig(
            left=BCType.INFLOW,
            right=BCType.OUTFLOW,
            bottom=BCType.FARFIELD,
            top=BCType.FARFIELD,
            u_inf=1.25,
            v_inf=-0.15,
            farfield_mode=FarfieldMode.DIRICHLET,
        )
        u = np.zeros(g.u_shape)
        v = np.zeros(g.v_shape)
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(v[:, 0], -0.15)
        assert np.allclose(v[:, g.ny], -0.15)
        assert np.allclose(u[1:-1, 0], 1.25)
        assert np.allclose(u[1:-1, g.ny - 1], 1.25)

    def test_farfield_neumann_copies_adjacent_velocity(self):
        g = make_grid()
        bc = BoundaryConfig(
            left=BCType.FARFIELD,
            right=BCType.FARFIELD,
            bottom=BCType.WALL,
            top=BCType.WALL,
            farfield_mode=FarfieldMode.NEUMANN,
        )
        u = np.random.rand(*g.u_shape)
        v = np.random.rand(*g.v_shape)
        u_left_interior = u[1, :].copy()
        u_right_interior = u[g.nx - 1, :].copy()
        v_left_interior = v[1, :].copy()
        v_right_interior = v[g.nx - 2, :].copy()
        apply_velocity_bc(u, v, g, bc)
        assert np.allclose(u[0, :], u_left_interior)
        assert np.allclose(u[g.nx, :], u_right_interior)
        assert np.allclose(v[0, 1:-1], v_left_interior[1:-1])
        assert np.allclose(v[g.nx - 1, 1:-1], v_right_interior[1:-1])

    def test_post_correction_farfield_dirichlet_reapplies_full_boundary(self):
        g = make_grid()
        bc = BoundaryConfig(
            left=BCType.INFLOW,
            right=BCType.OUTFLOW,
            bottom=BCType.FARFIELD,
            top=BCType.FARFIELD,
            u_inf=0.9,
            v_inf=0.2,
            farfield_mode=FarfieldMode.DIRICHLET,
        )
        u = np.random.rand(*g.u_shape)
        v = np.random.rand(*g.v_shape)
        apply_post_correction_bc(u, v, g, bc)
        assert np.allclose(v[:, 0], 0.2)
        assert np.allclose(v[:, g.ny], 0.2)
        assert np.allclose(u[1:-1, 0], 0.9)
        assert np.allclose(u[1:-1, g.ny - 1], 0.9)

    def test_farfield_dirichlet_sets_pressure_zero(self):
        g = make_grid()
        bc = BoundaryConfig(
            left=BCType.INFLOW,
            right=BCType.OUTFLOW,
            bottom=BCType.FARFIELD,
            top=BCType.FARFIELD,
            farfield_mode=FarfieldMode.DIRICHLET,
        )
        phi = np.random.rand(*g.p_shape) + 1.0
        apply_pressure_bc(phi, g, bc)
        assert np.allclose(phi[:, 0], 0.0)
        assert np.allclose(phi[:, g.ny - 1], 0.0)

    def test_farfield_neumann_copies_pressure(self):
        g = make_grid()
        bc = BoundaryConfig(
            left=BCType.FARFIELD,
            right=BCType.OUTFLOW,
            bottom=BCType.WALL,
            top=BCType.WALL,
            farfield_mode=FarfieldMode.NEUMANN,
        )
        phi = np.random.rand(*g.p_shape)
        phi_ref = phi[1, :].copy()
        apply_pressure_bc(phi, g, bc)
        assert np.allclose(phi[0, 1:-1], phi_ref[1:-1])
