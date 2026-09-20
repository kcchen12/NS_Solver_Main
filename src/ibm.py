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
    - Symmetric NACA 00xx airfoils
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
class FreeYCircleSpec:
    cx0: float
    cy0: float
    radius: float
    mass: float
    damping: float
    stiffness: float
    force_relaxation: float
    max_displacement: float
    max_speed: float
    inertia: float
    angular_damping: float
    angular_stiffness: float
    moment_relaxation: float
    max_angle: float
    max_angular_speed: float
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    initial_velocity_x: float
    initial_velocity_y: float
    release_time_x: float
    release_time_y: float
    released_x: bool
    released_y: bool
    angle: float
    angular_velocity: float
    free_x: bool
    free_y: bool
    free_theta: bool
    shape: str
    indent_width: float
    indent_depth: float
    chord: float
    thickness_ratio: float
    angle_offset: float
    polygon_points: np.ndarray
    mask_u: np.ndarray
    mask_v: np.ndarray
    last_force_x: float = 0.0
    last_force_y: float = 0.0
    last_moment_z: float = 0.0

    def center(self, time: float = 0.0) -> tuple[float, float]:
        del time
        return float(self.x), float(self.y)

    def x_active(self, time: float = 0.0) -> bool:
        return bool(self.free_x and float(time) >= self.release_time_x)

    def y_active(self, time: float = 0.0) -> bool:
        return bool(self.free_y and float(time) >= self.release_time_y)

    def velocity(self, time: float = 0.0) -> tuple[float, float]:
        return (
            float(self.velocity_x if self.x_active(time) else 0.0),
            float(self.velocity_y if self.y_active(time) else 0.0),
        )

    def u_velocity_at(self, y_coords: np.ndarray, time: float = 0.0) -> np.ndarray:
        velocity_x = self.velocity_x if self.x_active(time) else 0.0
        return velocity_x - self.angular_velocity * (y_coords - self.y)

    def v_velocity_at(self, x_coords: np.ndarray, time: float = 0.0) -> np.ndarray:
        velocity_y = self.velocity_y if self.y_active(time) else 0.0
        return velocity_y + self.angular_velocity * (x_coords - self.x)

    def body_angle_deg(self) -> float:
        return float(self.angle_offset + np.rad2deg(self.angle))

    def advance(
        self,
        force_x: float,
        force_y: float,
        moment_z: float,
        dt: float,
        time: float = 0.0,
    ) -> None:
        if not np.isfinite(force_x):
            raise FloatingPointError("free-x cylinder force is not finite")
        if not np.isfinite(force_y):
            raise FloatingPointError("free-y cylinder force is not finite")
        if not np.isfinite(moment_z):
            raise FloatingPointError("free-theta cylinder moment is not finite")
        if (
            not np.isfinite(self.x)
            or not np.isfinite(self.y)
            or not np.isfinite(self.velocity_x)
            or not np.isfinite(self.velocity_y)
            or not np.isfinite(self.angle)
            or not np.isfinite(self.angular_velocity)
        ):
            raise FloatingPointError("free cylinder state is not finite")

        if self.free_x and not self.x_active(time):
            self.velocity_x = 0.0
        if self.free_x and self.x_active(time):
            if not self.released_x:
                self.velocity_x = float(self.initial_velocity_x)
                self.released_x = True
            displacement_x = self.x - self.cx0
            acceleration_x = (
                float(force_x)
                - self.damping * self.velocity_x
                - self.stiffness * displacement_x
            ) / self.mass
            if not np.isfinite(acceleration_x):
                raise FloatingPointError("free-x cylinder acceleration is not finite")
            self.velocity_x += float(dt) * acceleration_x
            self.velocity_x = float(
                np.clip(self.velocity_x, -self.max_speed, self.max_speed)
            )
            self.x += float(dt) * self.velocity_x
            x_min = self.cx0 - self.max_displacement
            x_max = self.cx0 + self.max_displacement
            if self.x <= x_min:
                self.x = float(x_min)
                if self.velocity_x < 0.0:
                    self.velocity_x = 0.0
            elif self.x >= x_max:
                self.x = float(x_max)
                if self.velocity_x > 0.0:
                    self.velocity_x = 0.0

        if self.free_y and not self.y_active(time):
            self.velocity_y = 0.0
        if self.free_y and self.y_active(time):
            if not self.released_y:
                self.velocity_y = float(self.initial_velocity_y)
                self.released_y = True
            displacement_y = self.y - self.cy0
            acceleration_y = (
                float(force_y)
                - self.damping * self.velocity_y
                - self.stiffness * displacement_y
            ) / self.mass
            if not np.isfinite(acceleration_y):
                raise FloatingPointError("free-y cylinder acceleration is not finite")
            self.velocity_y += float(dt) * acceleration_y
            self.velocity_y = float(
                np.clip(self.velocity_y, -self.max_speed, self.max_speed)
            )
            self.y += float(dt) * self.velocity_y
            y_min = self.cy0 - self.max_displacement
            y_max = self.cy0 + self.max_displacement
            if self.y <= y_min:
                self.y = float(y_min)
                if self.velocity_y < 0.0:
                    self.velocity_y = 0.0
            elif self.y >= y_max:
                self.y = float(y_max)
                if self.velocity_y > 0.0:
                    self.velocity_y = 0.0

        if self.free_theta:
            angular_acceleration = (
                float(moment_z)
                - self.angular_damping * self.angular_velocity
                - self.angular_stiffness * self.angle
            ) / self.inertia
            if not np.isfinite(angular_acceleration):
                raise FloatingPointError(
                    "free-theta cylinder angular acceleration is not finite"
                )
            self.angular_velocity += float(dt) * angular_acceleration
            self.angular_velocity = float(
                np.clip(
                    self.angular_velocity,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )
            )
            self.angle += float(dt) * self.angular_velocity
            if self.angle <= -self.max_angle:
                self.angle = float(-self.max_angle)
                if self.angular_velocity < 0.0:
                    self.angular_velocity = 0.0
            elif self.angle >= self.max_angle:
                self.angle = float(self.max_angle)
                if self.angular_velocity > 0.0:
                    self.angular_velocity = 0.0

    def record_force(self, raw_force_x: float, raw_force_y: float) -> None:
        if not np.isfinite(raw_force_x):
            raise FloatingPointError("free-x cylinder raw force is not finite")
        if not np.isfinite(raw_force_y):
            raise FloatingPointError("free-y cylinder raw force is not finite")
        alpha = float(np.clip(self.force_relaxation, 0.0, 1.0))
        self.last_force_x = (1.0 - alpha) * self.last_force_x + alpha * raw_force_x
        self.last_force_y = (1.0 - alpha) * self.last_force_y + alpha * raw_force_y

    def record_moment(self, raw_moment_z: float) -> None:
        if not np.isfinite(raw_moment_z):
            raise FloatingPointError("free-theta cylinder raw moment is not finite")
        alpha = float(np.clip(self.moment_relaxation, 0.0, 1.0))
        self.last_moment_z = (
            (1.0 - alpha) * self.last_moment_z + alpha * raw_moment_z
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
        u_width = np.empty(grid.nx + 1, dtype=float)
        u_width[0] = 0.5 * grid.dx_cells[0]
        u_width[-1] = 0.5 * grid.dx_cells[-1]
        if grid.nx > 1:
            u_width[1:-1] = 0.5 * (grid.dx_cells[:-1] + grid.dx_cells[1:])
        v_height = np.empty(grid.ny + 1, dtype=float)
        v_height[0] = 0.5 * grid.dy_cells[0]
        v_height[-1] = 0.5 * grid.dy_cells[-1]
        if grid.ny > 1:
            v_height[1:-1] = 0.5 * (grid.dy_cells[:-1] + grid.dy_cells[1:])
        self._u_control_area = (
            u_width[:, np.newaxis] * grid.dy_cells[np.newaxis, :]
        )
        self._v_control_area = (
            grid.dx_cells[:, np.newaxis] * v_height[np.newaxis, :]
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
        self.free_y_circles: list[FreeYCircleSpec] = []

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
    def _boundary_faces(mask: np.ndarray) -> np.ndarray:
        """Return solid mask entries touching at least one fluid neighbor."""
        if mask.size == 0:
            return mask.copy()
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        interior = padded[1:-1, 1:-1]
        neighbor_all_solid = (
            padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        return interior & ~neighbor_all_solid

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

    @staticmethod
    def _body_local_coords(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        angle_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        angle = np.deg2rad(float(angle_deg))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_rel = x_coords - float(cx)
        y_rel = y_coords - float(cy)
        return (
            cos_a * x_rel + sin_a * y_rel,
            -sin_a * x_rel + cos_a * y_rel,
        )

    @classmethod
    def _rotated_rectangle_mask(
        cls,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        half_width: float,
        half_height: float,
        angle_deg: float,
    ) -> np.ndarray:
        x_local, y_local = cls._body_local_coords(
            x_coords, y_coords, cx, cy, angle_deg
        )
        return (
            (np.abs(x_local) <= float(half_width))
            & (np.abs(y_local) <= float(half_height))
        )

    @classmethod
    def _circle_with_top_indent_mask(
        cls,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        indent_width: float,
        indent_depth: float,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        x_local, y_local = cls._body_local_coords(
            x_coords, y_coords, cx, cy, angle_deg
        )
        circle = x_local ** 2 + y_local ** 2 <= float(radius) ** 2
        if indent_width <= 0.0 or indent_depth <= 0.0:
            return circle
        half_width = 0.5 * float(indent_width)
        notch = (
            (x_local >= -half_width)
            & (x_local <= half_width)
            & (y_local >= float(radius) - float(indent_depth))
            & (y_local <= float(radius))
        )
        return circle & ~notch

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
        mask_u = self._circle_with_top_indent_mask(
            xf,
            yc,
            cx,
            cy,
            radius,
            indent_width,
            indent_depth,
        )

        xc = grid.xc[:, np.newaxis]
        yf = grid.yf[np.newaxis, :]
        mask_v = self._circle_with_top_indent_mask(
            xc,
            yf,
            cx,
            cy,
            radius,
            indent_width,
            indent_depth,
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

    @staticmethod
    def _naca_00xx_mask(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        chord: float,
        thickness_ratio: float,
        angle_deg: float,
    ) -> np.ndarray:
        chord = float(chord)
        thickness_ratio = float(thickness_ratio)
        angle = np.deg2rad(float(angle_deg))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        x_rel = x_coords - float(cx)
        y_rel = y_coords - float(cy)
        x_local = cos_a * x_rel + sin_a * y_rel + 0.5 * chord
        y_local = -sin_a * x_rel + cos_a * y_rel

        s = np.clip(x_local / chord, 0.0, 1.0)
        y_thickness = 5.0 * thickness_ratio * chord * (
            0.2969 * np.sqrt(s)
            - 0.1260 * s
            - 0.3516 * s ** 2
            + 0.2843 * s ** 3
            - 0.1015 * s ** 4
        )
        return (
            (x_local >= 0.0)
            & (x_local <= chord)
            & (np.abs(y_local) <= y_thickness)
        )

    @classmethod
    def _polygon_mask(
        cls,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cx: float,
        cy: float,
        points: np.ndarray,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        polygon = np.asarray(points, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise ValueError("polygon points must be an Nx2 array with N >= 3")

        x_local, y_local = cls._body_local_coords(
            x_coords, y_coords, cx, cy, angle_deg
        )
        inside = np.zeros_like(x_local, dtype=bool)
        xj, yj = polygon[-1]
        eps = np.finfo(float).eps
        for xi, yi in polygon:
            crosses = ((yi > y_local) != (yj > y_local)) & (
                x_local
                < (xj - xi) * (y_local - yi) / ((yj - yi) + eps) + xi
            )
            inside ^= crosses
            xj, yj = xi, yi
        return inside

    def _free_body_masks(self, spec: FreeYCircleSpec) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = spec.center()
        shape = spec.shape
        if shape == "circle":
            return (
                self._circle_mask(self._u_x, self._u_y, cx, cy, spec.radius),
                self._circle_mask(self._v_x, self._v_y, cx, cy, spec.radius),
            )
        if shape == "circle-with-top-indent":
            return (
                self._circle_with_top_indent_mask(
                    self._u_x,
                    self._u_y,
                    cx,
                    cy,
                    spec.radius,
                    spec.indent_width,
                    spec.indent_depth,
                    spec.body_angle_deg(),
                ),
                self._circle_with_top_indent_mask(
                    self._v_x,
                    self._v_y,
                    cx,
                    cy,
                    spec.radius,
                    spec.indent_width,
                    spec.indent_depth,
                    spec.body_angle_deg(),
                ),
            )
        if shape == "square":
            return (
                self._rotated_rectangle_mask(
                    self._u_x,
                    self._u_y,
                    cx,
                    cy,
                    spec.radius,
                    spec.radius,
                    spec.body_angle_deg(),
                ),
                self._rotated_rectangle_mask(
                    self._v_x,
                    self._v_y,
                    cx,
                    cy,
                    spec.radius,
                    spec.radius,
                    spec.body_angle_deg(),
                ),
            )
        if shape == "airfoil":
            return (
                self._naca_00xx_mask(
                    self._u_x,
                    self._u_y,
                    cx,
                    cy,
                    spec.chord,
                    spec.thickness_ratio,
                    spec.body_angle_deg(),
                ),
                self._naca_00xx_mask(
                    self._v_x,
                    self._v_y,
                    cx,
                    cy,
                    spec.chord,
                    spec.thickness_ratio,
                    spec.body_angle_deg(),
                ),
            )
        if shape == "polygon":
            return (
                self._polygon_mask(
                    self._u_x,
                    self._u_y,
                    cx,
                    cy,
                    spec.polygon_points,
                    spec.body_angle_deg(),
                ),
                self._polygon_mask(
                    self._v_x,
                    self._v_y,
                    cx,
                    cy,
                    spec.polygon_points,
                    spec.body_angle_deg(),
                ),
            )
        raise ValueError(f"unsupported free body shape: {shape}")

    def _free_body_force_weights(
        self,
        spec: FreeYCircleSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = spec.center()
        if spec.shape == "circle":
            return (
                self._circle_force_weight(
                    self._u_x,
                    self._u_y,
                    cx,
                    cy,
                    spec.radius,
                    self._force_regularization_width,
                ),
                self._circle_force_weight(
                    self._v_x,
                    self._v_y,
                    cx,
                    cy,
                    spec.radius,
                    self._force_regularization_width,
                ),
            )
        return spec.mask_u.astype(float), spec.mask_v.astype(float)

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

    def add_naca_00xx_airfoil(
        self,
        cx: float,
        cy: float,
        chord: float,
        thickness_ratio: float = 0.12,
        angle_deg: float = 0.0,
        u_body: float = 0.0,
        v_body: float = 0.0,
    ) -> None:
        """Mark a symmetric NACA 00xx airfoil as a stationary solid body."""
        del u_body, v_body
        if chord <= 0.0:
            raise ValueError("chord must be positive")
        if thickness_ratio <= 0.0:
            raise ValueError("thickness_ratio must be positive")

        mask_u = self._naca_00xx_mask(
            self._u_x,
            self._u_y,
            cx,
            cy,
            chord,
            thickness_ratio,
            angle_deg,
        )
        mask_v = self._naca_00xx_mask(
            self._v_x,
            self._v_y,
            cx,
            cy,
            chord,
            thickness_ratio,
            angle_deg,
        )
        self._base_mask_u |= mask_u
        self._base_mask_v |= mask_v
        self.mask_u |= mask_u
        self.mask_v |= mask_v

    def add_polygon(
        self,
        cx: float,
        cy: float,
        points: np.ndarray,
        angle_deg: float = 0.0,
        u_body: float = 0.0,
        v_body: float = 0.0,
    ) -> None:
        """Mark a user-defined polygon bluff body as a stationary solid body."""
        del u_body, v_body
        polygon = np.asarray(points, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise ValueError("polygon points must be an Nx2 array with N >= 3")
        mask_u = self._polygon_mask(
            self._u_x,
            self._u_y,
            cx,
            cy,
            polygon,
            angle_deg,
        )
        mask_v = self._polygon_mask(
            self._v_x,
            self._v_y,
            cx,
            cy,
            polygon,
            angle_deg,
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

    def add_free_y_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        mass: float = 100.0,
        damping: float = 5.0,
        stiffness: float = 20.0,
        initial_velocity_y: float = 0.0,
        force_relaxation: float = 0.05,
        max_displacement: float | None = None,
        max_speed: float = 0.25,
        free_x: bool = False,
        free_y: bool = True,
        initial_velocity_x: float = 0.0,
        release_time_x: float = 0.0,
        release_time_y: float = 0.0,
        free_theta: bool = False,
        inertia: float = 10.0,
        angular_damping: float = 1.0,
        angular_stiffness: float = 5.0,
        initial_angle: float = 0.0,
        initial_angular_velocity: float = 0.0,
        moment_relaxation: float = 0.05,
        max_angle: float = np.pi / 4.0,
        max_angular_speed: float = 1.0,
        shape: str = "circle",
        indent_width: float = 0.0,
        indent_depth: float = 0.0,
        chord: float = -1.0,
        thickness_ratio: float = 0.12,
        angle_offset: float = 0.0,
        polygon_points: np.ndarray | None = None,
    ) -> None:
        """Add a bluff body with free spring-mass-damper motion.

        Each enabled direction follows a spring-mass-damper model driven by
        the matching computed IBM force:

            mass*x'' + damping*x' + stiffness*(x - cx) = F_x
            mass*y'' + damping*y' + stiffness*(y - cy) = F_y
            inertia*theta'' + angular_damping*theta'
                + angular_stiffness*theta = M_z
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if not free_x and not free_y and not free_theta:
            raise ValueError("at least one free direction must be enabled")
        if mass <= 0.0:
            raise ValueError("mass must be positive")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        if stiffness < 0.0:
            raise ValueError("stiffness must be non-negative")
        if force_relaxation <= 0.0 or force_relaxation > 1.0:
            raise ValueError("force_relaxation must be in (0, 1]")
        if max_displacement is None:
            max_displacement = 0.5 * float(radius)
        if max_displacement <= 0.0:
            raise ValueError("max_displacement must be positive")
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if release_time_x < 0.0:
            raise ValueError("release_time_x must be non-negative")
        if release_time_y < 0.0:
            raise ValueError("release_time_y must be non-negative")
        if inertia <= 0.0:
            raise ValueError("inertia must be positive")
        if angular_damping < 0.0:
            raise ValueError("angular_damping must be non-negative")
        if angular_stiffness < 0.0:
            raise ValueError("angular_stiffness must be non-negative")
        if moment_relaxation <= 0.0 or moment_relaxation > 1.0:
            raise ValueError("moment_relaxation must be in (0, 1]")
        if max_angle <= 0.0:
            raise ValueError("max_angle must be positive")
        if max_angular_speed <= 0.0:
            raise ValueError("max_angular_speed must be positive")
        shape = str(shape).strip().lower()
        if shape not in {
            "circle",
            "circle-with-top-indent",
            "square",
            "airfoil",
            "polygon",
        }:
            raise ValueError(f"unsupported free body shape: {shape}")
        if shape == "circle-with-top-indent":
            if indent_width <= 0.0 or indent_depth <= 0.0:
                raise ValueError("indent_width and indent_depth must be positive")
            if indent_width >= 2.0 * radius:
                raise ValueError("indent_width must be smaller than the body diameter")
            if indent_depth >= 2.0 * radius:
                raise ValueError("indent_depth must be smaller than the body diameter")
        if shape == "airfoil":
            if chord <= 0.0:
                chord = 2.0 * float(radius)
            if thickness_ratio <= 0.0:
                raise ValueError("thickness_ratio must be positive")
        else:
            chord = 2.0 * float(radius) if chord <= 0.0 else float(chord)
        if shape == "polygon":
            polygon = np.asarray(polygon_points, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
                raise ValueError("polygon_points must be an Nx2 array with N >= 3")
        else:
            polygon = np.zeros((0, 2), dtype=float)

        spec = FreeYCircleSpec(
            cx0=float(cx),
            cy0=float(cy),
            radius=float(radius),
            mass=float(mass),
            damping=float(damping),
            stiffness=float(stiffness),
            force_relaxation=float(force_relaxation),
            max_displacement=float(max_displacement),
            max_speed=float(max_speed),
            inertia=float(inertia),
            angular_damping=float(angular_damping),
            angular_stiffness=float(angular_stiffness),
            moment_relaxation=float(moment_relaxation),
            max_angle=float(max_angle),
            max_angular_speed=float(max_angular_speed),
            x=float(cx),
            y=float(cy),
            velocity_x=float(initial_velocity_x if release_time_x <= 0.0 else 0.0),
            velocity_y=float(initial_velocity_y if release_time_y <= 0.0 else 0.0),
            initial_velocity_x=float(initial_velocity_x),
            initial_velocity_y=float(initial_velocity_y),
            release_time_x=float(release_time_x),
            release_time_y=float(release_time_y),
            released_x=bool(release_time_x <= 0.0),
            released_y=bool(release_time_y <= 0.0),
            angle=float(initial_angle),
            angular_velocity=float(initial_angular_velocity),
            free_x=bool(free_x),
            free_y=bool(free_y),
            free_theta=bool(free_theta),
            shape=shape,
            indent_width=float(indent_width),
            indent_depth=float(indent_depth),
            chord=float(chord),
            thickness_ratio=float(thickness_ratio),
            angle_offset=float(angle_offset),
            polygon_points=polygon,
            mask_u=np.zeros(self.grid.u_shape, dtype=bool),
            mask_v=np.zeros(self.grid.v_shape, dtype=bool),
        )
        mask_u, mask_v = self._free_body_masks(spec)
        spec.mask_u = mask_u
        spec.mask_v = mask_v
        self.mask_u = self._base_mask_u | mask_u
        self.mask_v = self._base_mask_v | mask_v
        self.free_y_circles.append(spec)

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

    def _refresh_moving_circles(self, time: float) -> None:
        moving_circles = [*self.translating_circles, *self.free_y_circles]
        if not moving_circles:
            return
        self.mask_u = self._base_mask_u.copy()
        self.mask_v = self._base_mask_v.copy()
        for spec in moving_circles:
            if isinstance(spec, FreeYCircleSpec):
                spec.mask_u, spec.mask_v = self._free_body_masks(spec)
            else:
                cx, cy = spec.center(time)
                spec.mask_u = self._circle_mask(
                    self._u_x, self._u_y, cx, cy, spec.radius
                )
                spec.mask_v = self._circle_mask(
                    self._v_x, self._v_y, cx, cy, spec.radius
                )
            self.mask_u |= spec.mask_u
            self.mask_v |= spec.mask_v

    def _compute_moving_circle_force(
        self,
        u: np.ndarray,
        v: np.ndarray,
        time: float,
        dt: float,
        rho: float,
    ) -> tuple[float, float]:
        force_x = 0.0
        force_y = 0.0
        for spec in [*self.translating_circles, *self.free_y_circles]:
            cx, cy = spec.center(time)
            if isinstance(spec, FreeYCircleSpec):
                weight_u, weight_v = self._free_body_force_weights(spec)
                u_body_field = spec.u_velocity_at(self._u_y, time=time)
                v_body_field = spec.v_velocity_at(self._v_x, time=time)
                force_u = (u - u_body_field) * self._u_face_measure * weight_u
                force_v = (v - v_body_field) * self._v_face_measure * weight_v
                raw_force_x = float(np.sum(force_u))
                raw_force_y = float(np.sum(force_v))
                raw_moment_z = float(
                    np.sum(-(self._u_y - cy) * force_u)
                    + np.sum((self._v_x - cx) * force_v)
                )
                spec.record_force(rho * raw_force_x / dt, rho * raw_force_y / dt)
                spec.record_moment(rho * raw_moment_z / dt)
            else:
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
                u_body_t, v_body_t = spec.velocity(time)
                raw_force_x = float(
                    np.sum((u - u_body_t) * self._u_face_measure * weight_u)
                )
                raw_force_y = float(
                    np.sum((v - v_body_t) * self._v_face_measure * weight_v)
                )
            force_x += raw_force_x
            force_y += raw_force_y
        return rho * force_x / dt, rho * force_y / dt

    def advance_free_y_circles(self, dt: float, time: float = 0.0) -> None:
        """Advance free cylinders using their latest IBM forces."""
        if dt <= 0.0:
            return
        for spec in self.free_y_circles:
            spec.advance(
                spec.last_force_x,
                spec.last_force_y,
                spec.last_moment_z,
                dt,
                time=time,
            )
        self._refresh_moving_circles(0.0)

    def first_free_y_circle_state(self, time: float = 0.0) -> dict | None:
        if not self.free_y_circles:
            return None
        spec = self.free_y_circles[0]
        return {
            "center_x": float(spec.x),
            "center_y": float(spec.y),
            "reference_x": float(spec.cx0),
            "reference_y": float(spec.cy0),
            "velocity_x": float(spec.velocity_x),
            "velocity_y": float(spec.velocity_y),
            "active_x": bool(spec.x_active(time)),
            "active_y": bool(spec.y_active(time)),
            "displacement_x": float(spec.x - spec.cx0),
            "displacement_y": float(spec.y - spec.cy0),
            "angle": float(spec.angle),
            "angular_velocity": float(spec.angular_velocity),
            "free_x": bool(spec.free_x),
            "free_y": bool(spec.free_y),
            "free_theta": bool(spec.free_theta),
            "shape": spec.shape,
            "mass": float(spec.mass),
            "damping": float(spec.damping),
            "stiffness": float(spec.stiffness),
            "inertia": float(spec.inertia),
            "angular_damping": float(spec.angular_damping),
            "angular_stiffness": float(spec.angular_stiffness),
            "force_relaxation": float(spec.force_relaxation),
            "moment_relaxation": float(spec.moment_relaxation),
            "release_time_x": float(spec.release_time_x),
            "release_time_y": float(spec.release_time_y),
            "max_displacement": float(spec.max_displacement),
            "max_speed": float(spec.max_speed),
            "max_angle": float(spec.max_angle),
            "max_angular_speed": float(spec.max_angular_speed),
            "last_force_x": float(spec.last_force_x),
            "last_force_y": float(spec.last_force_y),
            "last_moment_z": float(spec.last_moment_z),
        }

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
        self._refresh_moving_circles(time)
        enforce_u_mask = self.mask_u
        enforce_v_mask = self.mask_v

        has_moving_body = bool(
            self.translating_circles or self.free_y_circles or self.rotating_circles
        )
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

        if self.translating_circles or self.free_y_circles:
            enforce_u_mask = self.mask_u.copy()
            enforce_v_mask = self.mask_v.copy()
            for spec in [*self.translating_circles, *self.free_y_circles]:
                if isinstance(spec, FreeYCircleSpec):
                    u_target[spec.mask_u] = spec.u_velocity_at(
                        self._u_y,
                        time=time,
                    )[spec.mask_u]
                    v_target[spec.mask_v] = spec.v_velocity_at(
                        self._v_x,
                        time=time,
                    )[spec.mask_v]
                else:
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
            if self.translating_circles or self.free_y_circles:
                force_x, force_y = self._compute_moving_circle_force(
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
