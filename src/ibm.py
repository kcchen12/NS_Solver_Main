"""
Immersed boundary method (IBM) — direct forcing approach.

Theory
------
The immersed boundary method represents solid geometry *inside* the
computational domain by adding a body-force term f to the momentum
equations:

    ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f

where f is chosen to enforce the desired velocity (typically zero) at
solid-body points.

Direct-forcing approach
~~~~~~~~~~~~~~~~~~~~~~~
At each time step, after computing the intermediate velocity u*, we
correct it so that solid-body points match the prescribed velocity:

    u*[solid] = u_body[solid]

This is equivalent to an infinite forcing that drives u to u_body
instantaneously.  While first-order in time at the boundary, it is
simple to implement, robust, and widely used in the literature
(Mohd-Yusof 1997, Fadlun et al. 2000).

Geometry
--------
Solid regions are specified as level-set / mask arrays:

    mask_u[i,j] = 1  if x-face (xf[i], yc[j]) is inside solid
    mask_v[i,j] = 1  if y-face (xc[i], yf[j]) is inside solid

Helper methods allow adding:
    - Circular cylinders
    - Rectangular blocks
    - Arbitrary masks (load from array)
"""

from dataclasses import dataclass

import numpy as np
from src.grid import CartesianGrid


@dataclass
class RotatingCircleSpec:
    cx: float
    cy: float
    radius: float
    omega_amplitude: float
    frequency: float
    phase: float
    mask_u: np.ndarray
    mask_v: np.ndarray
    u_y_offset: np.ndarray
    v_x_offset: np.ndarray

    def angular_velocity(self, time: float) -> float:
        return float(
            self.omega_amplitude * np.sin(2.0 * np.pi * self.frequency * time + self.phase)
        )


@dataclass
class TranslatingCircleSpec:
    cx: float
    cy: float
    radius: float
    amplitude_x: float
    amplitude_y: float
    frequency: float
    phase: float
    mask_u: np.ndarray
    mask_v: np.ndarray

    def center(self, time: float) -> tuple[float, float]:
        theta = 2.0 * np.pi * self.frequency * time + self.phase
        displacement = np.sin(theta)
        return (
            float(self.cx + self.amplitude_x * displacement),
            float(self.cy + self.amplitude_y * displacement),
        )

    def velocity(self, time: float) -> tuple[float, float]:
        theta = 2.0 * np.pi * self.frequency * time + self.phase
        scale = 2.0 * np.pi * self.frequency * np.cos(theta)
        return (
            float(self.amplitude_x * scale),
            float(self.amplitude_y * scale),
        )


class ImmersedBoundary:
    """
    Manages solid-body masks for the IBM direct-forcing approach.

    Parameters
    ----------
    grid : CartesianGrid
    """

    def __init__(self, grid: CartesianGrid):
        self.grid = grid
        # Binary masks: 1 = solid, 0 = fluid
        self.mask_u = np.zeros(grid.u_shape, dtype=bool)
        self.mask_v = np.zeros(grid.v_shape, dtype=bool)
        self._u_x = np.broadcast_to(grid.xf[:, np.newaxis], grid.u_shape)
        self._u_y = np.broadcast_to(grid.yc[np.newaxis, :], grid.u_shape)
        self._v_x = np.broadcast_to(grid.xc[:, np.newaxis], grid.v_shape)
        self._v_y = np.broadcast_to(grid.yf[np.newaxis, :], grid.v_shape)
        self._u_face_measure = np.broadcast_to(
            grid.dy_cells[np.newaxis, :], grid.u_shape
        )
        self._v_face_measure = np.broadcast_to(
            grid.dx_cells[:, np.newaxis], grid.v_shape
        )
        self._force_regularization_width = max(
            float(np.median(grid.dx_cells)),
            float(np.median(grid.dy_cells)),
            np.finfo(float).eps,
        )
        self._base_mask_u = np.zeros(grid.u_shape, dtype=bool)
        self._base_mask_v = np.zeros(grid.v_shape, dtype=bool)
        self.rotating_circles: list[RotatingCircleSpec] = []
        self.translating_circles: list[TranslatingCircleSpec] = []

    @staticmethod
    def _circle_mask(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> np.ndarray:
        radius_sq = float(radius) ** 2
        return ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) <= radius_sq

    @staticmethod
    def _circle_force_weight(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        transition_width: float,
    ) -> np.ndarray:
        distance = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        width = max(float(transition_width), np.finfo(float).eps)
        return np.clip(0.5 + (float(radius) - distance) / width, 0.0, 1.0)

    @staticmethod
    def _top_indent_mask(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
    ) -> np.ndarray:
        if indent_width <= 0.0 or indent_depth <= 0.0:
            return np.zeros_like(x_coords, dtype=bool)

        half_width = 0.5 * float(indent_width)
        y0 = float(cy + radius - indent_depth)
        y1 = float(cy + radius)
        return (
            (x_coords >= cx - half_width)
            & (x_coords <= cx + half_width)
            & (y_coords >= y0)
            & (y_coords <= y1)
        )

    def _circle_with_top_indent_masks(
        self,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid

        xf = grid.xf[:, np.newaxis]
        yc = grid.yc[np.newaxis, :]
        mask_u = self._circle_mask(xf, yc, cx, cy, radius)
        mask_u &= ~self._top_indent_mask(
            xf, yc, cx, cy, radius, indent_width, indent_depth
        )

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        mask_v = self._circle_mask(xc, yf, cx, cy, radius)
        mask_v &= ~self._top_indent_mask(
            xc, yf, cx, cy, radius, indent_width, indent_depth
        )
        return mask_u, mask_v

    def _rectangle_masks(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid
        mask_u = (
            (grid.xf[:, np.newaxis] >= x0)
            & (grid.xf[:, np.newaxis] <= x1)
            & (grid.yc[np.newaxis, :] >= y0)
            & (grid.yc[np.newaxis, :] <= y1)
        )
        mask_v = (
            (grid.xc[:, np.newaxis] >= x0)
            & (grid.xc[:, np.newaxis] <= x1)
            & (grid.yf[np.newaxis, :] >= y0)
            & (grid.yf[np.newaxis, :] <= y1)
        )
        return mask_u, mask_v

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def add_circle(self, cx: float, cy: float, radius: float,
                   u_body: float = 0.0, v_body: float = 0.0) -> None:
        """
        Mark all MAC faces inside a circular cylinder as solid.

        Parameters
        ----------
        cx, cy : float
            Centre of the cylinder in physical coordinates.
        radius : float
            Cylinder radius.
        u_body, v_body : float
            Prescribed velocity on the body surface (0 for stationary wall).
        """
        del u_body, v_body
        mask_u = self._circle_mask(self._u_x, self._u_y, cx, cy, radius)
        mask_v = self._circle_mask(self._v_x, self._v_y, cx, cy, radius)
        self._base_mask_u |= mask_u
        self._base_mask_v |= mask_v
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    def add_circle_with_top_indent(
        self,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
        u_body: float = 0.0,
        v_body: float = 0.0,
    ) -> None:
        """
        Mark a circular body with a rectangular cut-out at the top as solid.

        The indent is centered at x=cx and removes the region
        [cx-indent_width/2, cx+indent_width/2] x [cy+radius-indent_depth, cy+radius].
        """
        del u_body, v_body
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if indent_width <= 0.0 or indent_depth <= 0.0:
            raise ValueError("indent_width and indent_depth must be positive")
        if indent_width >= 2.0 * radius:
            raise ValueError("indent_width must be smaller than the cylinder diameter")
        if indent_depth >= 2.0 * radius:
            raise ValueError("indent_depth must be smaller than the cylinder diameter")

        mask_u, mask_v = self._circle_with_top_indent_masks(
            cx=cx,
            cy=cy,
            radius=radius,
            indent_width=indent_width,
            indent_depth=indent_depth,
        )
        self._base_mask_u |= mask_u
        self._base_mask_v |= mask_v
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    def add_rotating_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        omega_amplitude: float,
        frequency: float,
        phase: float = 0.0,
    ) -> None:
        """Add a circular cylinder with sinusoidal back-and-forth rotation.

        The imposed angular velocity is

            omega(t) = omega_amplitude * sin(2*pi*frequency*t + phase)

        with tangential wall velocity

            u = -omega(t) * (y - cy)
            v =  omega(t) * (x - cx)
        """
        mask_u = self._circle_mask(self._u_x, self._u_y, cx, cy, radius)
        mask_v = self._circle_mask(self._v_x, self._v_y, cx, cy, radius)

        self._base_mask_u |= mask_u
        self._base_mask_v |= mask_v
        self.mask_u |= mask_u
        self.mask_v |= mask_v
        self.rotating_circles.append(
            RotatingCircleSpec(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                omega_amplitude=float(omega_amplitude),
                frequency=float(frequency),
                phase=float(phase),
                mask_u=mask_u,
                mask_v=mask_v,
                u_y_offset=self._u_y[mask_u] - float(cy),
                v_x_offset=self._v_x[mask_v] - float(cx),
            )
        )

    def add_constant_rotating_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        omega: float,
    ) -> None:
        """Add a circular cylinder with constant one-direction rotation.

        The imposed angular velocity is

            omega(t) = omega

        with tangential wall velocity

            u = -omega * (y - cy)
            v =  omega * (x - cx)
        """
        self.add_rotating_circle(
            cx=cx,
            cy=cy,
            radius=radius,
            omega_amplitude=float(omega),
            frequency=0.0,
            phase=0.5 * np.pi,
        )

    def add_translating_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        amplitude_x: float,
        amplitude_y: float,
        frequency: float,
        phase: float = 0.0,
    ) -> None:
        """Add a non-rotating circular cylinder with sinusoidal translation.

        The center follows

            x_c(t) = cx + amplitude_x * sin(2*pi*frequency*t + phase)
            y_c(t) = cy + amplitude_y * sin(2*pi*frequency*t + phase)

        and the imposed wall velocity is the matching rigid-body
        translational velocity.
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if frequency < 0.0:
            raise ValueError("frequency must be non-negative")
        mask_u = self._circle_mask(self._u_x, self._u_y, cx, cy, radius)
        mask_v = self._circle_mask(self._v_x, self._v_y, cx, cy, radius)
        self.mask_u = self._base_mask_u | mask_u
        self.mask_v = self._base_mask_v | mask_v
        self.translating_circles.append(
            TranslatingCircleSpec(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                amplitude_x=float(amplitude_x),
                amplitude_y=float(amplitude_y),
                frequency=float(frequency),
                phase=float(phase),
                mask_u=mask_u,
                mask_v=mask_v,
            )
        )

    def add_rectangle(self, x0: float, x1: float,
                      y0: float, y1: float,
                      u_body: float = 0.0, v_body: float = 0.0) -> None:
        """
        Mark all MAC faces inside axis-aligned rectangle [x0,x1]×[y0,y1].
        """
        del u_body, v_body
        grid = self.grid
        mask_u = (
            (grid.xf[:, np.newaxis] >= x0)
            & (grid.xf[:, np.newaxis] <= x1)
            & (grid.yc[np.newaxis, :] >= y0)
            & (grid.yc[np.newaxis, :] <= y1)
        )
        mask_v = (
            (grid.xc[:, np.newaxis] >= x0)
            & (grid.xc[:, np.newaxis] <= x1)
            & (grid.yf[np.newaxis, :] >= y0)
            & (grid.yf[np.newaxis, :] <= y1)
        )
        self._base_mask_u |= mask_u
        self._base_mask_v |= mask_v
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    def add_mask(self, mask_u: np.ndarray, mask_v: np.ndarray) -> None:
        """
        Directly supply boolean masks for u and v faces.
        """
        if mask_u.shape != self.grid.u_shape:
            raise ValueError(
                f"mask_u must have shape {self.grid.u_shape}, got {mask_u.shape}"
            )
        if mask_v.shape != self.grid.v_shape:
            raise ValueError(
                f"mask_v must have shape {self.grid.v_shape}, got {mask_v.shape}"
            )
        self._base_mask_u |= mask_u
        self._base_mask_v |= mask_v
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    def _refresh_translating_circles(self, time: float) -> None:
        if not self.translating_circles:
            return
        self.mask_u = self._base_mask_u.copy()
        self.mask_v = self._base_mask_v.copy()
        for spec in self.translating_circles:
            cx, cy = spec.center(time)
            spec.mask_u = self._circle_mask(self._u_x, self._u_y, cx, cy, spec.radius)
            spec.mask_v = self._circle_mask(self._v_x, self._v_y, cx, cy, spec.radius)
            self.mask_u |= spec.mask_u
            self.mask_v |= spec.mask_v

    def _compute_translating_circle_force(
        self,
        u: np.ndarray,
        v: np.ndarray,
        time: float,
        dt: float,
        rho: float,
    ) -> tuple[float, float]:
        force_x = 0.0
        force_y = 0.0
        for spec in self.translating_circles:
            cx, cy = spec.center(time)
            u_body_t, v_body_t = spec.velocity(time)
            weight_u = self._circle_force_weight(
                self._u_x,
                self._u_y,
                cx,
                cy,
                spec.radius,
                self._force_regularization_width,
            )
            weight_v = self._circle_force_weight(
                self._v_x,
                self._v_y,
                cx,
                cy,
                spec.radius,
                self._force_regularization_width,
            )
            force_x += float(
                np.sum((u - u_body_t) * self._u_face_measure * weight_u)
            )
            force_y += float(
                np.sum((v - v_body_t) * self._v_face_measure * weight_v)
            )
        return rho * force_x / dt, rho * force_y / dt

    # ------------------------------------------------------------------
    # Forcing
    # ------------------------------------------------------------------

    def apply(self, u: np.ndarray, v: np.ndarray,
              u_body: float = 0.0, v_body: float = 0.0,
              dt: float = None, rho: float = 1.0,
              return_face_forcing: bool = False,
              time: float = 0.0):
        """
        Apply direct forcing **in-place**: set solid-face velocities to the
        prescribed body velocity (default: 0 for stationary body).

        Call this *after* computing the intermediate velocity u* and
        *before* the pressure-correction step.
        """
        force_x = 0.0
        force_y = 0.0
        forcing_u = None
        forcing_v = None
        self._refresh_translating_circles(time)
        enforce_u_mask = self.mask_u
        enforce_v_mask = self.mask_v

        has_moving_body = bool(self.translating_circles or self.rotating_circles)
        if not has_moving_body:
            u_body = float(u_body)
            v_body = float(v_body)
            if dt is not None and dt > 0.0:
                force_x = (
                    rho
                    * float(
                        np.sum(
                            (u[enforce_u_mask] - u_body)
                            * self._u_face_measure[enforce_u_mask]
                        )
                    )
                    / dt
                )
                force_y = (
                    rho
                    * float(
                        np.sum(
                            (v[enforce_v_mask] - v_body)
                            * self._v_face_measure[enforce_v_mask]
                        )
                    )
                    / dt
                )
                if return_face_forcing:
                    forcing_u = np.zeros_like(u)
                    forcing_v = np.zeros_like(v)
                    forcing_u[enforce_u_mask] = (u_body - u[enforce_u_mask]) / dt
                    forcing_v[enforce_v_mask] = (v_body - v[enforce_v_mask]) / dt
            elif return_face_forcing:
                forcing_u = np.zeros_like(u)
                forcing_v = np.zeros_like(v)

            u[enforce_u_mask] = u_body
            v[enforce_v_mask] = v_body
            if return_face_forcing:
                return force_x, force_y, forcing_u, forcing_v
            return force_x, force_y

        u_target = np.full_like(u, float(u_body))
        v_target = np.full_like(v, float(v_body))

        if self.translating_circles:
            enforce_u_mask = self.mask_u.copy()
            enforce_v_mask = self.mask_v.copy()
            for spec in self.translating_circles:
                u_body_t, v_body_t = spec.velocity(time)
                u_target[spec.mask_u] = u_body_t
                v_target[spec.mask_v] = v_body_t
                enforce_u_mask |= spec.mask_u
                enforce_v_mask |= spec.mask_v

        if self.rotating_circles:
            for spec in self.rotating_circles:
                omega = spec.angular_velocity(time)
                u_target[spec.mask_u] = -omega * spec.u_y_offset
                v_target[spec.mask_v] = omega * spec.v_x_offset

        if dt is not None and dt > 0.0:
            if self.translating_circles:
                force_x, force_y = self._compute_translating_circle_force(
                    u,
                    v,
                    time=time,
                    dt=dt,
                    rho=rho,
                )
            else:
                force_x = (
                    rho
                    * float(
                        np.sum(
                            (u[enforce_u_mask] - u_target[enforce_u_mask])
                            * self._u_face_measure[enforce_u_mask]
                        )
                    )
                    / dt
                )
                force_y = (
                    rho
                    * float(
                        np.sum(
                            (v[enforce_v_mask] - v_target[enforce_v_mask])
                            * self._v_face_measure[enforce_v_mask]
                        )
                    )
                    / dt
                )
            if return_face_forcing:
                forcing_u = np.zeros_like(u)
                forcing_v = np.zeros_like(v)
                forcing_u[enforce_u_mask] = (
                    u_target[enforce_u_mask] - u[enforce_u_mask]) / dt
                forcing_v[enforce_v_mask] = (
                    v_target[enforce_v_mask] - v[enforce_v_mask]) / dt
        elif return_face_forcing:
            forcing_u = np.zeros_like(u)
            forcing_v = np.zeros_like(v)

        u[enforce_u_mask] = u_target[enforce_u_mask]
        v[enforce_v_mask] = v_target[enforce_v_mask]
        if return_face_forcing:
            return force_x, force_y, forcing_u, forcing_v
        return force_x, force_y

    @property
    def has_solid(self) -> bool:
        """True if any solid cells are defined."""
        return bool(self.mask_u.any() or self.mask_v.any())
