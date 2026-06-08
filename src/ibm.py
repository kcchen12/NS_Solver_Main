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


@dataclass
class SweepingJetSpec:
    cx: float
    cy: float
    radius: float
    jet_speed: float
    slot_center_angle: float
    slot_width_angle: float
    slot_depth: float
    sweep_amplitude: float
    frequency: float
    phase: float
    mask_u: np.ndarray
    mask_v: np.ndarray
    u_normal_x: np.ndarray
    u_tangent_x: np.ndarray
    v_normal_y: np.ndarray
    v_tangent_y: np.ndarray

    def sweep_angle(self, time: float) -> float:
        return float(
            self.sweep_amplitude * np.sin(2.0 * np.pi * self.frequency * time + self.phase)
        )


@dataclass
class OscillatingJetPatchSpec:
    jet_speed: float
    base_angle: float
    sweep_amplitude: float
    frequency: float
    phase: float
    mask_u: np.ndarray
    mask_v: np.ndarray

    def direction_angle(self, time: float) -> float:
        return float(
            self.base_angle
            + self.sweep_amplitude * np.sin(2.0 * np.pi * self.frequency * time + self.phase)
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
        self.sweeping_jets: list[SweepingJetSpec] = []
        self.oscillating_jet_patches: list[OscillatingJetPatchSpec] = []

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

    @staticmethod
    def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _circular_slot_masks(
        self,
        cx: float,
        cy: float,
        radius: float,
        slot_center_angle: float,
        slot_width_angle: float,
        slot_depth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid

        xf = grid.xf[:, np.newaxis]
        yc = grid.yc[np.newaxis, :]
        r_u = np.sqrt((xf - cx) ** 2 + (yc - cy) ** 2)
        theta_u = np.arctan2(yc - cy, xf - cx)
        mask_u = (
            (r_u <= radius)
            & (r_u >= radius - slot_depth)
            & (np.abs(self._wrap_angle(theta_u - slot_center_angle)) <= 0.5 * slot_width_angle)
        )

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        r_v = np.sqrt((xc - cx) ** 2 + (yf - cy) ** 2)
        theta_v = np.arctan2(yf - cy, xc - cx)
        mask_v = (
            (r_v <= radius)
            & (r_v >= radius - slot_depth)
            & (np.abs(self._wrap_angle(theta_v - slot_center_angle)) <= 0.5 * slot_width_angle)
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

    def _local_rectangle_masks(
        self,
        cx: float,
        cy: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        s0: float,
        s1: float,
        n0: float,
        n1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_dx = self._u_x - float(cx)
        u_dy = self._u_y - float(cy)
        v_dx = self._v_x - float(cx)
        v_dy = self._v_y - float(cy)

        u_s = u_dx * tangent_x + u_dy * tangent_y
        u_n = u_dx * normal_x + u_dy * normal_y
        v_s = v_dx * tangent_x + v_dy * tangent_y
        v_n = v_dx * normal_x + v_dy * normal_y

        mask_u = (u_s >= s0) & (u_s <= s1) & (u_n >= n0) & (u_n <= n1)
        mask_v = (v_s >= s0) & (v_s <= s1) & (v_n >= n0) & (v_n <= n1)
        return mask_u, mask_v

    def _local_tapered_channel_masks(
        self,
        cx: float,
        cy: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        n0: float,
        n1: float,
        width0: float,
        width1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n1 <= n0:
            raise ValueError("tapered channel requires n1 > n0")

        u_dx = self._u_x - float(cx)
        u_dy = self._u_y - float(cy)
        v_dx = self._v_x - float(cx)
        v_dy = self._v_y - float(cy)

        u_s = u_dx * tangent_x + u_dy * tangent_y
        u_n = u_dx * normal_x + u_dy * normal_y
        v_s = v_dx * tangent_x + v_dy * tangent_y
        v_n = v_dx * normal_x + v_dy * normal_y

        u_alpha = np.clip((u_n - n0) / (n1 - n0), 0.0, 1.0)
        v_alpha = np.clip((v_n - n0) / (n1 - n0), 0.0, 1.0)
        u_half_width = 0.5 * ((1.0 - u_alpha) * width0 + u_alpha * width1)
        v_half_width = 0.5 * ((1.0 - v_alpha) * width0 + v_alpha * width1)

        mask_u = (
            (u_n >= n0)
            & (u_n <= n1)
            & (np.abs(u_s) <= u_half_width)
        )
        mask_v = (
            (v_n >= n0)
            & (v_n <= n1)
            & (np.abs(v_s) <= v_half_width)
        )
        return mask_u, mask_v

    def _local_triangle_masks(
        self,
        cx: float,
        cy: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        def point_mask(s_vals: np.ndarray, n_vals: np.ndarray) -> np.ndarray:
            x0, y0 = p0
            x1, y1 = p1
            x2, y2 = p2
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) <= 1e-14:
                return np.zeros_like(s_vals, dtype=bool)
            a = ((y1 - y2) * (s_vals - x2) + (x2 - x1) * (n_vals - y2)) / denom
            b = ((y2 - y0) * (s_vals - x2) + (x0 - x2) * (n_vals - y2)) / denom
            c = 1.0 - a - b
            tol = 1e-12
            return (a >= -tol) & (b >= -tol) & (c >= -tol)

        u_dx = self._u_x - float(cx)
        u_dy = self._u_y - float(cy)
        v_dx = self._v_x - float(cx)
        v_dy = self._v_y - float(cy)
        u_s = u_dx * tangent_x + u_dy * tangent_y
        u_n = u_dx * normal_x + u_dy * normal_y
        v_s = v_dx * tangent_x + v_dy * tangent_y
        v_n = v_dx * normal_x + v_dy * normal_y
        return point_mask(u_s, u_n), point_mask(v_s, v_n)

    def _local_convex_polygon_masks(
        self,
        cx: float,
        cy: float,
        tangent_x: float,
        tangent_y: float,
        normal_x: float,
        normal_y: float,
        corners: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(corners) < 3:
            raise ValueError("polygon requires at least three corners")

        def point_mask(s_vals: np.ndarray, n_vals: np.ndarray) -> np.ndarray:
            mask = np.ones_like(s_vals, dtype=bool)
            tol = 1e-12
            for idx, (s0, n0) in enumerate(corners):
                s1, n1 = corners[(idx + 1) % len(corners)]
                edge_s = s1 - s0
                edge_n = n1 - n0
                rel_s = s_vals - s0
                rel_n = n_vals - n0
                cross = edge_s * rel_n - edge_n * rel_s
                mask &= cross >= -tol
            return mask

        u_dx = self._u_x - float(cx)
        u_dy = self._u_y - float(cy)
        v_dx = self._v_x - float(cx)
        v_dy = self._v_y - float(cy)
        u_s = u_dx * tangent_x + u_dy * tangent_y
        u_n = u_dx * normal_x + u_dy * normal_y
        v_s = v_dx * tangent_x + v_dy * tangent_y
        v_n = v_dx * normal_x + v_dy * normal_y
        return point_mask(u_s, u_n), point_mask(v_s, v_n)

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

    def add_sweeping_jet_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        jet_speed: float,
        slot_center_angle_deg: float = 90.0,
        slot_width_angle_deg: float = 18.0,
        slot_depth: float = 0.0,
        sweep_amplitude_deg: float = 25.0,
        frequency: float = 0.0,
        phase: float = 0.0,
    ) -> None:
        """
        Add a finite-width sweeping jet outlet on a circular IBM body.

        The outlet location is fixed on the body surface, while the jet
        direction oscillates relative to the local outward normal:

            u_jet = U_j * (cos(alpha(t)) * n_hat + sin(alpha(t)) * t_hat)

        where alpha(t) = sweep_amplitude * sin(2*pi*frequency*t + phase).
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if jet_speed < 0.0:
            raise ValueError("jet_speed must be non-negative")

        slot_width_angle = np.deg2rad(float(slot_width_angle_deg))
        if slot_width_angle <= 0.0 or slot_width_angle >= 2.0 * np.pi:
            raise ValueError("slot_width_angle_deg must be in (0, 360)")

        if slot_depth <= 0.0:
            slot_depth = 0.15 * radius
        if slot_depth >= radius:
            raise ValueError("slot_depth must be smaller than the cylinder radius")

        mask_u, mask_v = self._circular_slot_masks(
            cx=cx,
            cy=cy,
            radius=radius,
            slot_center_angle=np.deg2rad(float(slot_center_angle_deg)),
            slot_width_angle=slot_width_angle,
            slot_depth=float(slot_depth),
        )
        ru = np.sqrt((self._u_x - cx) ** 2 + (self._u_y - cy) ** 2)
        rv = np.sqrt((self._v_x - cx) ** 2 + (self._v_y - cy) ** 2)
        ru = np.where(ru > 1e-14, ru, 1.0)
        rv = np.where(rv > 1e-14, rv, 1.0)

        nu_x = (self._u_x - cx) / ru
        nu_y = (self._u_y - cy) / ru
        nv_x = (self._v_x - cx) / rv
        nv_y = (self._v_y - cy) / rv

        tu_x = -nu_y
        tv_y = nv_x

        self.add_circle(cx, cy, radius)
        self.sweeping_jets.append(
            SweepingJetSpec(
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                jet_speed=float(jet_speed),
                slot_center_angle=np.deg2rad(float(slot_center_angle_deg)),
                slot_width_angle=slot_width_angle,
                slot_depth=float(slot_depth),
                sweep_amplitude=np.deg2rad(float(sweep_amplitude_deg)),
                frequency=float(frequency),
                phase=float(phase),
                mask_u=mask_u,
                mask_v=mask_v,
                u_normal_x=nu_x[mask_u],
                u_tangent_x=tu_x[mask_u],
                v_normal_y=nv_y[mask_v],
                v_tangent_y=tv_y[mask_v],
            )
        )

    def add_geometry_resolved_sweeping_jet_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        jet_speed: float,
        cavity_width: float,
        cavity_height: float,
        slot_width: float,
        slot_height: float,
        feed_width: float,
        feed_height: float,
        nozzle_length: float = 0.0,
        slot_exit_width: float = 0.0,
        island_wall_gap: float = 0.0,
        island_center_gap: float = 0.0,
        island_leading_gap: float = 0.0,
        island_trailing_gap: float = 0.0,
        island_taper: float = 0.0,
        slot_center_angle_deg: float = 90.0,
        sweep_amplitude_deg: float = 25.0,
        frequency: float = 0.0,
        phase: float = 0.0,
    ) -> None:
        """
        Add a circle with an internal rectangular plenum and a narrow surface slot.

        The cylinder remains an IBM body, but the cavity/slot are carved back
        into the fluid domain. A forcing patch inside the plenum drives the jet.
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if jet_speed < 0.0:
            raise ValueError("jet_speed must be non-negative")
        if cavity_width <= 0.0 or cavity_height <= 0.0:
            raise ValueError("cavity dimensions must be positive")
        if slot_width <= 0.0 or slot_height <= 0.0:
            raise ValueError("slot dimensions must be positive")
        if feed_width <= 0.0 or feed_height <= 0.0:
            raise ValueError("feed patch dimensions must be positive")
        if cavity_width >= 2.0 * radius:
            raise ValueError("cavity_width must be smaller than the cylinder diameter")
        if cavity_height >= 2.0 * radius:
            raise ValueError("cavity_height must be smaller than the cylinder diameter")
        if slot_width >= cavity_width:
            raise ValueError("slot_width must be smaller than cavity_width")
        if slot_height >= cavity_height:
            raise ValueError("slot_height must be smaller than cavity_height")
        if feed_width > cavity_width or feed_height > cavity_height:
            raise ValueError("feed patch must fit inside the cavity")

        slot_center_angle = np.deg2rad(float(slot_center_angle_deg))
        normal_x = float(np.cos(slot_center_angle))
        normal_y = float(np.sin(slot_center_angle))
        tangent_x = float(-np.sin(slot_center_angle))
        tangent_y = float(np.cos(slot_center_angle))

        cavity_n1 = radius - slot_height
        cavity_n0 = cavity_n1 - cavity_height
        if nozzle_length <= 0.0:
            nozzle_length = min(
                max(1.5 * slot_height, 0.35 * cavity_height),
                0.65 * cavity_height,
            )
        plenum_n1 = cavity_n1 - nozzle_length

        slot_s0 = -0.5 * slot_width
        slot_s1 = 0.5 * slot_width
        slot_n0 = radius - slot_height
        slot_n1 = radius
        if slot_exit_width <= 0.0:
            slot_exit_width = min(cavity_width, slot_width + 2.0 * slot_height)

        plenum_depth = max(plenum_n1 - cavity_n0, 1e-12)
        if island_wall_gap <= 0.0:
            island_wall_gap = 0.12 * cavity_width
        if island_center_gap <= 0.0:
            island_center_gap = max(0.22 * cavity_width, feed_width + 0.08 * cavity_width)
        island_height = 0.5 * max(
            cavity_width - 2.0 * island_wall_gap - island_center_gap,
            0.18 * cavity_width,
        )
        if island_leading_gap <= 0.0:
            island_leading_gap = 0.18 * plenum_depth
        if island_trailing_gap <= 0.0:
            island_trailing_gap = 0.18 * plenum_depth
        island_n0 = cavity_n0 + island_leading_gap
        island_n1 = plenum_n1 - island_trailing_gap
        if island_taper <= 0.0:
            island_taper = 0.18 * island_height

        upper_s0 = 0.5 * island_center_gap
        upper_s1 = upper_s0 + island_height
        lower_s1 = -0.5 * island_center_gap
        lower_s0 = lower_s1 - island_height

        upper_island = [
            (upper_s0, island_n0),
            (upper_s1, island_n0),
            (upper_s1 - island_taper, island_n1),
            (upper_s0 + island_taper, island_n1),
        ]
        lower_island = [
            (lower_s0, island_n0),
            (lower_s1, island_n0),
            (lower_s1 - island_taper, island_n1),
            (lower_s0 + island_taper, island_n1),
        ]

        feed_s0 = -0.5 * feed_width
        feed_s1 = 0.5 * feed_width
        # Keep the forcing patch centered inside the plenum so the resolved
        # jet remains visually and numerically aligned with the cylinder axis.
        cavity_nc = 0.5 * (cavity_n0 + cavity_n1)
        feed_n0 = cavity_nc - 0.5 * feed_height
        feed_n1 = cavity_nc + 0.5 * feed_height

        circle_u = self._circle_mask(self._u_x, self._u_y, cx, cy, radius)
        circle_v = self._circle_mask(self._v_x, self._v_y, cx, cy, radius)
        plenum_u, plenum_v = self._local_rectangle_masks(
            cx, cy,
            tangent_x, tangent_y,
            normal_x, normal_y,
            -0.5 * cavity_width, 0.5 * cavity_width,
            cavity_n0, plenum_n1,
        )
        nozzle_u, nozzle_v = self._local_tapered_channel_masks(
            cx, cy,
            tangent_x, tangent_y,
            normal_x, normal_y,
            plenum_n1, cavity_n1,
            cavity_width, slot_width,
        )
        slot_u, slot_v = self._local_tapered_channel_masks(
            cx, cy,
            tangent_x, tangent_y,
            normal_x, normal_y,
            slot_n0, slot_n1,
            slot_width, slot_exit_width,
        )

        cavity_u = plenum_u | nozzle_u
        cavity_v = plenum_v | nozzle_v
        upper_u, upper_v = self._local_convex_polygon_masks(
            cx, cy,
            tangent_x, tangent_y,
            normal_x, normal_y,
            upper_island,
        )
        lower_u, lower_v = self._local_convex_polygon_masks(
            cx, cy,
            tangent_x, tangent_y,
            normal_x, normal_y,
            lower_island,
        )
        splitter_u = upper_u | lower_u
        splitter_v = upper_v | lower_v
        cavity_u &= ~splitter_u
        cavity_v &= ~splitter_v
        solid_u = circle_u & ~(cavity_u | slot_u)
        solid_v = circle_v & ~(cavity_v | slot_v)
        self.add_mask(solid_u, solid_v)

        feed_u, feed_v = self._local_rectangle_masks(
            cx, cy,
            tangent_x, tangent_y,
            normal_x, normal_y,
            feed_s0, feed_s1,
            feed_n0, feed_n1,
        )
        self.oscillating_jet_patches.append(
            OscillatingJetPatchSpec(
                jet_speed=float(jet_speed),
                base_angle=slot_center_angle,
                sweep_amplitude=np.deg2rad(float(sweep_amplitude_deg)),
                frequency=float(frequency),
                phase=float(phase),
                mask_u=feed_u,
                mask_v=feed_v,
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
        u_target = np.full_like(u, float(u_body))
        v_target = np.full_like(v, float(v_body))
        self._refresh_translating_circles(time)
        enforce_u_mask = self.mask_u
        enforce_v_mask = self.mask_v

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

        if self.sweeping_jets:
            enforce_u_mask = self.mask_u.copy()
            enforce_v_mask = self.mask_v.copy()
            for spec in self.sweeping_jets:
                alpha = spec.sweep_angle(time)
                u_target[spec.mask_u] = spec.jet_speed * (
                    np.cos(alpha) * spec.u_normal_x + np.sin(alpha) * spec.u_tangent_x
                )
                v_target[spec.mask_v] = spec.jet_speed * (
                    np.cos(alpha) * spec.v_normal_y + np.sin(alpha) * spec.v_tangent_y
                )
                enforce_u_mask |= spec.mask_u
                enforce_v_mask |= spec.mask_v

        if self.oscillating_jet_patches:
            if enforce_u_mask is self.mask_u:
                enforce_u_mask = self.mask_u.copy()
                enforce_v_mask = self.mask_v.copy()
            for spec in self.oscillating_jet_patches:
                angle = spec.direction_angle(time)
                u_target[spec.mask_u] = spec.jet_speed * np.cos(angle)
                v_target[spec.mask_v] = spec.jet_speed * np.sin(angle)
                enforce_u_mask |= spec.mask_u
                enforce_v_mask |= spec.mask_v

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
