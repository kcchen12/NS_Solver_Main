"""
Boundary condition types and enforcement for MAC staggered grids.

Supported conditions
--------------------
INFLOW          : prescribed velocity (u=U_inf, v=V_inf)
FARFIELD        : free-stream boundary using either Dirichlet or Neumann mode
OUTFLOW         : first-order convective outflow
WALL            : no-slip, no-penetration solid wall
PERIODIC        : periodic (handled externally via ghost-cell exchange)
"""

import numpy as np
from src.grid import CartesianGrid


class BCType:
    INFLOW = "inflow"
    FARFIELD = "farfield"
    OUTFLOW = "outflow"
    WALL = "wall"
    PERIODIC = "periodic"


class WallSlipMode:
    NO_SLIP = "no-slip"
    FREE_SLIP = "free-slip"


class OutflowMode:
    CONVECTIVE = "convective"
    ZERO_GRADIENT = "zero-gradient"


class FarfieldMode:
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class BoundaryConfig:
    """Stores boundary-condition type for each domain face."""

    def __init__(self, left: str, right: str, bottom: str, top: str,
                 front: str = BCType.WALL, back: str = BCType.WALL,
                 u_inf: float = 1.0, v_inf: float = 0.0, w_inf: float = 0.0,
                 wall_slip_mode: str = WallSlipMode.NO_SLIP,
                 wall_penetration: bool = False,
                 wall_normal_velocity: float = 0.0,
                 farfield_mode: str = FarfieldMode.DIRICHLET,
                 outflow_mode: str = OutflowMode.CONVECTIVE,
                 outflow_speed: float = None):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.front = front
        self.back = back
        self.u_inf = u_inf
        self.v_inf = v_inf
        self.w_inf = w_inf
        self.wall_slip_mode = wall_slip_mode
        self.wall_penetration = wall_penetration
        self.wall_normal_velocity = wall_normal_velocity
        self.farfield_mode = farfield_mode
        self.outflow_mode = outflow_mode
        self.outflow_speed = outflow_speed


def _is_free_slip(bc: BoundaryConfig) -> bool:
    return str(bc.wall_slip_mode).lower() == WallSlipMode.FREE_SLIP


def _is_farfield_dirichlet(bc: BoundaryConfig) -> bool:
    return str(bc.farfield_mode).lower() == FarfieldMode.DIRICHLET


def _wall_normal_velocity(bc: BoundaryConfig) -> float:
    if bc.wall_penetration:
        return float(bc.wall_normal_velocity)
    return 0.0


def _outflow_convective_speed(bc: BoundaryConfig, normal_reference: float) -> float:
    if bc.outflow_speed is not None:
        return max(abs(float(bc.outflow_speed)), 1e-10)
    return max(abs(float(normal_reference)), 1e-10)


def _ghost_no_slip(interior_row: np.ndarray, wall_value: float) -> np.ndarray:
    """Return ghost-cell values that enforce wall_value at the boundary face."""
    return 2.0 * wall_value - interior_row


def _is_pressure_dirichlet(face_type: str, bc: BoundaryConfig) -> bool:
    return face_type == BCType.OUTFLOW or (
        face_type == BCType.FARFIELD and _is_farfield_dirichlet(bc)
    )


def apply_velocity_bc(u: np.ndarray, v: np.ndarray,
                      grid: CartesianGrid, bc: BoundaryConfig,
                      dt: float = None,
                      w: np.ndarray = None) -> None:
    """Apply boundary conditions to velocity fields in-place."""
    nx, ny = grid.nx, grid.ny
    free_slip = _is_free_slip(bc)
    wall_vn = _wall_normal_velocity(bc)
    farfield_dirichlet = _is_farfield_dirichlet(bc)

    if bc.left == BCType.INFLOW:
        u[0, :] = bc.u_inf
        v[0, :] = bc.v_inf
    elif bc.left == BCType.FARFIELD:
        if farfield_dirichlet:
            u[0, :] = bc.u_inf
            v[0, :] = bc.v_inf
        else:
            u[0, :] = u[1, :]
            v[0, :] = v[1, :]
    elif bc.left == BCType.WALL:
        u[0, :] = wall_vn
        if free_slip:
            v[0, :] = v[1, :]
        else:
            v[0, :] = _ghost_no_slip(v[1, :], 0.0)

    if bc.right == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Uc = _outflow_convective_speed(bc, bc.u_inf)
            cfl = abs(Uc) * dt / grid.dx_min
            u[nx, :] = u[nx, :] - cfl * (u[nx, :] - u[nx - 1, :])
        else:
            u[nx, :] = u[nx - 1, :]
        v[nx - 1, :] = v[nx - 2, :]
    elif bc.right == BCType.INFLOW:
        u[nx, :] = bc.u_inf
        v[nx - 1, :] = bc.v_inf
    elif bc.right == BCType.FARFIELD:
        if farfield_dirichlet:
            u[nx, :] = bc.u_inf
            v[nx - 1, :] = bc.v_inf
        else:
            u[nx, :] = u[nx - 1, :]
            v[nx - 1, :] = v[nx - 2, :]
    elif bc.right == BCType.WALL:
        u[nx, :] = wall_vn
        if free_slip:
            v[nx - 1, :] = v[nx - 2, :]
        else:
            v[nx - 1, :] = _ghost_no_slip(v[nx - 2, :], 0.0)

    if bc.bottom == BCType.WALL:
        v[:, 0] = wall_vn
        if free_slip:
            u[1:-1, 0] = u[1:-1, 1]
    elif bc.bottom == BCType.INFLOW:
        v[:, 0] = bc.v_inf
        u[1:-1, 0] = bc.u_inf
    elif bc.bottom == BCType.FARFIELD:
        if farfield_dirichlet:
            v[:, 0] = bc.v_inf
            u[1:-1, 0] = bc.u_inf
        else:
            v[:, 0] = v[:, 1]
            u[1:-1, 0] = u[1:-1, 1]
    elif bc.bottom == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = abs(Vc) * dt / grid.dy_min
            v[:, 0] = v[:, 0] - cfl * (v[:, 0] - v[:, 1])
        else:
            v[:, 0] = v[:, 1]
        u[1:-1, 0] = u[1:-1, 1]

    if bc.top == BCType.WALL:
        v[:, ny] = wall_vn
        if free_slip:
            u[1:-1, ny - 1] = u[1:-1, ny - 2]
    elif bc.top == BCType.INFLOW:
        v[:, ny] = bc.v_inf
        u[1:-1, ny - 1] = bc.u_inf
    elif bc.top == BCType.FARFIELD:
        if farfield_dirichlet:
            v[:, ny] = bc.v_inf
            u[1:-1, ny - 1] = bc.u_inf
        else:
            v[:, ny] = v[:, ny - 1]
            u[1:-1, ny - 1] = u[1:-1, ny - 2]
    elif bc.top == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = abs(Vc) * dt / grid.dy_min
            v[:, ny] = v[:, ny] - cfl * (v[:, ny] - v[:, ny - 1])
        else:
            v[:, ny] = v[:, ny - 1]
        u[1:-1, ny - 1] = u[1:-1, ny - 2]

    if grid.is_3d and w is not None:
        nz = grid.nz
        if bc.front == BCType.WALL:
            w[:, :, 0] = 0.0
        elif bc.front == BCType.INFLOW:
            w[:, :, 0] = bc.w_inf
        elif bc.front == BCType.FARFIELD:
            if farfield_dirichlet:
                w[:, :, 0] = bc.w_inf
            else:
                w[:, :, 0] = w[:, :, 1]

        if bc.back == BCType.WALL:
            w[:, :, nz] = 0.0
        elif bc.back == BCType.INFLOW:
            w[:, :, nz] = bc.w_inf
        elif bc.back == BCType.FARFIELD:
            if farfield_dirichlet:
                w[:, :, nz] = bc.w_inf
            else:
                w[:, :, nz] = w[:, :, nz - 1]


def apply_post_correction_bc(u: np.ndarray, v: np.ndarray,
                             grid: CartesianGrid, bc: BoundaryConfig,
                             dt: float = None,
                             w: np.ndarray = None) -> None:
    """Re-enforce the required post-projection boundary conditions."""
    nx, ny = grid.nx, grid.ny
    free_slip = _is_free_slip(bc)
    wall_vn = _wall_normal_velocity(bc)
    farfield_dirichlet = _is_farfield_dirichlet(bc)

    if bc.left == BCType.INFLOW:
        u[0, :] = bc.u_inf
    elif bc.left == BCType.FARFIELD:
        if farfield_dirichlet:
            u[0, :] = bc.u_inf
            v[0, :] = bc.v_inf
        else:
            u[0, :] = u[1, :]
            v[0, :] = v[1, :]
    elif bc.left == BCType.WALL:
        u[0, :] = wall_vn
        if free_slip:
            v[0, :] = v[1, :]

    if bc.right == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Uc = _outflow_convective_speed(bc, bc.u_inf)
            cfl = abs(Uc) * dt / grid.dx_min
            u[nx, :] = u[nx, :] - cfl * (u[nx, :] - u[nx - 1, :])
        else:
            u[nx, :] = u[nx - 1, :]
        v[nx - 1, :] = v[nx - 2, :]
    elif bc.right == BCType.INFLOW:
        u[nx, :] = bc.u_inf
    elif bc.right == BCType.FARFIELD:
        if farfield_dirichlet:
            u[nx, :] = bc.u_inf
            v[nx - 1, :] = bc.v_inf
        else:
            u[nx, :] = u[nx - 1, :]
            v[nx - 1, :] = v[nx - 2, :]
    elif bc.right == BCType.WALL:
        u[nx, :] = wall_vn
        if free_slip:
            v[nx - 1, :] = v[nx - 2, :]

    if bc.bottom == BCType.WALL:
        v[:, 0] = wall_vn
        if free_slip:
            u[1:-1, 0] = u[1:-1, 1]
    elif bc.bottom == BCType.FARFIELD:
        if farfield_dirichlet:
            v[:, 0] = bc.v_inf
            u[1:-1, 0] = bc.u_inf
        else:
            v[:, 0] = v[:, 1]
            u[1:-1, 0] = u[1:-1, 1]
    elif bc.bottom == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = abs(Vc) * dt / grid.dy_min
            v[:, 0] = v[:, 0] - cfl * (v[:, 0] - v[:, 1])
        else:
            v[:, 0] = v[:, 1]

    if bc.top == BCType.WALL:
        v[:, ny] = wall_vn
        if free_slip:
            u[1:-1, ny - 1] = u[1:-1, ny - 2]
    elif bc.top == BCType.FARFIELD:
        if farfield_dirichlet:
            v[:, ny] = bc.v_inf
            u[1:-1, ny - 1] = bc.u_inf
        else:
            v[:, ny] = v[:, ny - 1]
            u[1:-1, ny - 1] = u[1:-1, ny - 2]
    elif bc.top == BCType.OUTFLOW:
        if dt is not None and str(bc.outflow_mode).lower() == OutflowMode.CONVECTIVE:
            Vc = _outflow_convective_speed(bc, bc.v_inf)
            cfl = abs(Vc) * dt / grid.dy_min
            v[:, ny] = v[:, ny] - cfl * (v[:, ny] - v[:, ny - 1])
        else:
            v[:, ny] = v[:, ny - 1]

    if grid.is_3d and w is not None:
        nz = grid.nz
        if bc.front == BCType.WALL:
            w[:, :, 0] = 0.0
        elif bc.front == BCType.FARFIELD:
            if farfield_dirichlet:
                w[:, :, 0] = bc.w_inf
            else:
                w[:, :, 0] = w[:, :, 1]

        if bc.back == BCType.WALL:
            w[:, :, nz] = 0.0
        elif bc.back == BCType.FARFIELD:
            if farfield_dirichlet:
                w[:, :, nz] = bc.w_inf
            else:
                w[:, :, nz] = w[:, :, nz - 1]


def apply_pressure_bc(phi: np.ndarray,
                      grid: CartesianGrid, bc: BoundaryConfig) -> None:
    """Apply pressure-correction boundary conditions in-place."""
    nx, ny = grid.nx, grid.ny

    if _is_pressure_dirichlet(bc.left, bc):
        phi[0, ...] = 0.0
    else:
        phi[0, ...] = phi[1, ...]

    if _is_pressure_dirichlet(bc.right, bc):
        phi[nx - 1, ...] = 0.0
    else:
        phi[nx - 1, ...] = phi[nx - 2, ...]

    if _is_pressure_dirichlet(bc.bottom, bc):
        phi[:, 0, ...] = 0.0
    else:
        phi[:, 0, ...] = phi[:, 1, ...]

    if _is_pressure_dirichlet(bc.top, bc):
        phi[:, ny - 1, ...] = 0.0
    else:
        phi[:, ny - 1, ...] = phi[:, ny - 2, ...]

    if grid.is_3d:
        nz = grid.nz
        phi[..., 0] = phi[..., 1]
        phi[..., nz - 1] = phi[..., nz - 2]
