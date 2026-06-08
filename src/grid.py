"""
Cartesian MAC staggered grid.

Staggered variable placement (2-D shown; 3-D adds z-faces for w):

    p  : cell centres          shape (nx, ny [, nz])
    u  : x-face centres        shape (nx+1, ny [, nz])
    v  : y-face centres        shape (nx, ny+1 [, nz])
    w  : z-face centres (3-D)  shape (nx, ny, nz+1)

Index convention: first index → x, second → y, third → z.

Face/node indices run from 0 to nx (inclusive) for x-faces, etc.
Cell-centre indices run from 0 to nx-1.

Example (nx=4, ny=3, 2-D):

    y
    ^
    |  [0,2][1,2][2,2][3,2]   ← cell centres (p)
    |  [0,1][1,1][2,1][3,1]
    |  [0,0][1,0][2,0][3,0]
    +-----------------------------> x

    u-faces (4+1=5 in x, 3 in y):
       i=0  i=1  i=2  i=3  i=4

    v-faces (4 in x, 3+1=4 in y):
       j=0(bottom boundary) … j=3(top boundary)
"""

import numpy as np


class CartesianGrid:
    """Cartesian grid supporting uniform and non-uniform 2-D layouts."""

    def __init__(self, nx: int, ny: int, nz: int = 1,
                 lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
                 x_min: float = 0.0,
                 y_min: float = 0.0,
                 z_min: float = 0.0,
                 xf: np.ndarray | None = None,
                 yf: np.ndarray | None = None,
                 zf: np.ndarray | None = None):
        """
        Parameters
        ----------
        nx, ny, nz : int
            Number of cells in each direction. ``nz=1`` activates 2-D mode.
        lx, ly, lz : float
            Physical domain lengths.
        x_min, y_min, z_min : float
            Lower bounds of the domain when faces are generated internally.
        """
        self.nx, self.ny, self.nz = nx, ny, nz

        self.is_3d: bool = nz > 1

        # Cell-face (node) coordinates
        self.xf = np.linspace(
            x_min, x_min + lx, nx + 1) if xf is None else np.asarray(xf, dtype=float)
        self.yf = np.linspace(
            y_min, y_min + ly, ny + 1) if yf is None else np.asarray(yf, dtype=float)
        self.zf = np.linspace(
            z_min, z_min + lz, nz + 1) if zf is None else np.asarray(zf, dtype=float)

        if self.xf.shape != (nx + 1,):
            raise ValueError(f"xf must have shape ({nx + 1},)")
        if self.yf.shape != (ny + 1,):
            raise ValueError(f"yf must have shape ({ny + 1},)")
        if self.zf.shape != (nz + 1,):
            raise ValueError(f"zf must have shape ({nz + 1},)")
        if not np.all(np.diff(self.xf) > 0.0):
            raise ValueError("xf must be strictly increasing")
        if not np.all(np.diff(self.yf) > 0.0):
            raise ValueError("yf must be strictly increasing")
        if not np.all(np.diff(self.zf) > 0.0):
            raise ValueError("zf must be strictly increasing")

        self.x_min = float(self.xf[0])
        self.x_max = float(self.xf[-1])
        self.y_min = float(self.yf[0])
        self.y_max = float(self.yf[-1])
        self.z_min = float(self.zf[0])
        self.z_max = float(self.zf[-1])

        self.lx = self.x_max - self.x_min
        self.ly = self.y_max - self.y_min
        self.lz = self.z_max - self.z_min

        # Cell-centre coordinates
        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])

        self.dx_cells = np.diff(self.xf)
        self.dy_cells = np.diff(self.yf)
        self.dz_cells = np.diff(self.zf)
        self.dxc = np.diff(self.xc)
        self.dyc = np.diff(self.yc)

        self.dx_min = float(np.min(self.dx_cells))
        self.dy_min = float(np.min(self.dy_cells))
        self.dz_min = float(np.min(self.dz_cells))
        self.dx_max = float(np.max(self.dx_cells))
        self.dy_max = float(np.max(self.dy_cells))
        self.dz_max = float(np.max(self.dz_cells))

        # Backward-compatible scalar spacings used in legacy code paths.
        self.dx = self.dx_min
        self.dy = self.dy_min
        self.dz = self.dz_min if nz > 1 else self.lz
        self.mean_cell_area = float(
            np.mean(self.dx_cells) * np.mean(self.dy_cells))

        self.xc_with_ghost = np.empty(self.nx + 2, dtype=float)
        self.xc_with_ghost[1:-1] = self.xc
        self.xc_with_ghost[0] = 2.0 * self.x_min - self.xc[0]
        self.xc_with_ghost[-1] = 2.0 * self.x_max - self.xc[-1]

        self.yc_with_ghost = np.empty(self.ny + 2, dtype=float)
        self.yc_with_ghost[1:-1] = self.yc
        self.yc_with_ghost[0] = 2.0 * self.y_min - self.yc[0]
        self.yc_with_ghost[-1] = 2.0 * self.y_max - self.yc[-1]

        self.is_uniform = bool(
            np.allclose(self.dx_cells, self.dx_cells[0]) and
            np.allclose(self.dy_cells, self.dy_cells[0]) and
            np.allclose(self.dz_cells, self.dz_cells[0])
        )

    # ------------------------------------------------------------------
    # Array shape helpers
    # ------------------------------------------------------------------

    @property
    def p_shape(self) -> tuple:
        """Shape of pressure array."""
        return (self.nx, self.ny, self.nz) if self.is_3d else (self.nx, self.ny)

    @property
    def u_shape(self) -> tuple:
        """Shape of x-velocity array (staggered in x)."""
        return (self.nx + 1, self.ny, self.nz) if self.is_3d else (self.nx + 1, self.ny)

    @property
    def v_shape(self) -> tuple:
        """Shape of y-velocity array (staggered in y)."""
        return (self.nx, self.ny + 1, self.nz) if self.is_3d else (self.nx, self.ny + 1)

    @property
    def w_shape(self) -> tuple:
        """Shape of z-velocity array (staggered in z, 3-D only)."""
        if self.is_3d:
            return (self.nx, self.ny, self.nz + 1)
        return None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    def zeros_p(self) -> np.ndarray:
        return np.zeros(self.p_shape)

    def zeros_u(self) -> np.ndarray:
        return np.zeros(self.u_shape)

    def zeros_v(self) -> np.ndarray:
        return np.zeros(self.v_shape)

    def zeros_w(self) -> np.ndarray:
        if not self.is_3d:
            raise RuntimeError("w-velocity only exists in 3-D mode.")
        return np.zeros(self.w_shape)

    def to_metadata(self) -> dict:
        """Return serializable metadata describing the prepared grid."""
        return {
            "grid_type": "uniform" if self.is_uniform else "nonuniform",
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "lx": self.lx,
            "ly": self.ly,
            "lz": self.lz,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "dx_min": self.dx_min,
            "dy_min": self.dy_min,
            "dz_min": self.dz_min,
            "dx_max": self.dx_max,
            "dy_max": self.dy_max,
            "dz_max": self.dz_max,
            "is_3d": self.is_3d,
            "xf": self.xf.copy(),
            "yf": self.yf.copy(),
            "zf": self.zf.copy(),
            "xc": self.xc.copy(),
            "yc": self.yc.copy(),
            "zc": self.zc.copy(),
            "dx_cells": self.dx_cells.copy(),
            "dy_cells": self.dy_cells.copy(),
            "dz_cells": self.dz_cells.copy(),
            "is_uniform": self.is_uniform,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "CartesianGrid":
        """Rebuild a prepared grid from saved metadata."""
        xf = metadata.get("xf", None)
        yf = metadata.get("yf", None)
        zf = metadata.get("zf", None)
        return cls(
            nx=int(metadata["nx"]),
            ny=int(metadata["ny"]),
            nz=int(metadata.get("nz", 1)),
            lx=float(metadata["lx"]),
            ly=float(metadata["ly"]),
            lz=float(metadata.get("lz", 1.0)),
            x_min=float(metadata.get("x_min", 0.0)),
            y_min=float(metadata.get("y_min", 0.0)),
            z_min=float(metadata.get("z_min", 0.0)),
            xf=xf,
            yf=yf,
            zf=zf,
        )

    def __repr__(self) -> str:
        dim = "3D" if self.is_3d else "2D"
        grid_kind = "uniform" if self.is_uniform else "nonuniform"
        return (f"CartesianGrid({dim}, nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"type={grid_kind}, dx_min={self.dx_min:.4g}, dy_min={self.dy_min:.4g}, dz={self.dz:.4g})")


def stretched_faces_tanh(n: int, length: float, beta: float) -> np.ndarray:
    """Build monotonic face coordinates in [0, length] using tanh stretching.

    beta <= 0 returns a uniform distribution.
    beta > 0 clusters cells near both boundaries and coarsens near the center.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    xi = np.linspace(0.0, 1.0, n + 1)
    if beta <= 0.0:
        return length * xi

    eta = 2.0 * xi - 1.0
    s = 0.5 * (1.0 + np.tanh(beta * eta) / np.tanh(beta))
    return length * s


def _allocate_piecewise_cells(
    n: int,
    lengths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Allocate integer cell counts across piecewise segments."""
    lengths = np.asarray(lengths, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = lengths > 1e-14

    counts = np.zeros_like(lengths, dtype=int)
    num_positive = int(np.count_nonzero(positive))
    if num_positive == 0:
        raise ValueError("At least one segment must have positive length")

    if n <= num_positive:
        ranked = np.argsort(-(lengths * weights))
        counts[ranked[:n]] = 1
        return counts

    counts[positive] = 1
    remaining = n - num_positive
    effective = lengths[positive] * weights[positive]
    total = float(np.sum(effective))
    if total <= 0.0:
        counts[positive] += remaining // num_positive
        counts[np.flatnonzero(positive)[:remaining % num_positive]] += 1
        return counts

    raw = remaining * effective / total
    add = np.floor(raw).astype(int)
    counts[np.flatnonzero(positive)] += add

    shortfall = remaining - int(np.sum(add))
    if shortfall > 0:
        frac = raw - add
        order = np.argsort(-frac)
        positive_idx = np.flatnonzero(positive)
        for idx in order[:shortfall]:
            counts[positive_idx[idx]] += 1

    return counts


def _solve_geometric_ratio(n: int, first_width: float, total_length: float) -> float:
    """Solve for geometric ratio r in sum(first_width * r**k, k=0..n-1)=total_length."""
    if n <= 0:
        return 1.0
    if first_width <= 0.0:
        return 1.0

    target = float(total_length)
    mean_match = n * first_width
    if np.isclose(target, mean_match, rtol=1e-12, atol=1e-14):
        return 1.0

    def series_sum(r: float) -> float:
        if np.isclose(r, 1.0, rtol=0.0, atol=1e-14):
            return mean_match
        return first_width * (r**n - 1.0) / (r - 1.0)

    if target > mean_match:
        lo, hi = 1.0, 2.0
        while series_sum(hi) < target:
            hi *= 2.0
            if hi > 1e6:
                break
    else:
        lo, hi = 1e-8, 1.0

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if series_sum(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _faces_from_core_edge(
    n: int,
    length: float,
    core_edge_dx: float,
    core_at_upper_end: bool,
) -> np.ndarray:
    """Create faces on [0, length] with cell width matched at the core edge."""
    if n <= 0:
        return np.asarray([0.0, length], dtype=float)
    if length <= 0.0:
        return np.linspace(0.0, length, n + 1)

    ratio = _solve_geometric_ratio(n, core_edge_dx, length)
    widths = core_edge_dx * (ratio ** np.arange(n, dtype=float))

    if core_at_upper_end:
        cumulative = np.concatenate(([0.0], np.cumsum(widths)))
        faces = length - cumulative[::-1]
    else:
        faces = np.concatenate(([0.0], np.cumsum(widths)))

    faces[0] = 0.0
    faces[-1] = length
    return faces


def _resolve_center_uniform_interval(
    length: float,
    core_interval: tuple[float, float],
) -> tuple[float, float]:
    """Resolve [core_start, core_end] on [0, length] for center-uniform mode."""
    core_start = float(core_interval[0])
    core_end = float(core_interval[1])
    core_start = float(np.clip(core_start, 0.0, length))
    core_end = float(np.clip(core_end, 0.0, length))
    if not core_end > core_start:
        raise ValueError("core_interval must define a positive-width interval")
    return core_start, core_end


def stretched_faces_center_uniform(
    n: int,
    length: float,
    beta: float,
    core_interval: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    """Faces with a uniform central band and stretched outer regions.

    The middle band keeps constant spacing, while the left and right outer
    segments are smoothly stretched so cells grow away from the band toward
    the boundaries.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    core_start, core_end = _resolve_center_uniform_interval(
        length=length,
        core_interval=core_interval,
    )

    segment_lengths = np.array(
        [core_start, core_end - core_start, length - core_end],
        dtype=float,
    )
    core_weight = 1.0 + 0.25 * max(beta, 0.0)
    segment_weights = np.array([1.0, core_weight, 1.0], dtype=float)
    segment_counts = _allocate_piecewise_cells(
        n, segment_lengths, segment_weights)

    core_n = int(segment_counts[1])
    if (core_end - core_start) > 1e-14 and core_n <= 0:
        donor = int(np.argmax(segment_lengths))
        if segment_counts[donor] > 1:
            segment_counts[donor] -= 1
            segment_counts[1] += 1

    segments = []

    left_n = int(segment_counts[0])
    core_n = int(segment_counts[1])
    right_n = int(segment_counts[2])
    core_len = float(core_end - core_start)
    core_dx = core_len / max(core_n, 1)

    if left_n > 0:
        segments.append(
            _faces_from_core_edge(
                left_n, core_start, core_edge_dx=core_dx, core_at_upper_end=True
            )
        )

    if core_n > 0:
        segments.append(np.linspace(core_start, core_end, core_n + 1))

    if right_n > 0:
        segments.append(
            core_end
            + _faces_from_core_edge(
                right_n,
                length - core_end,
                core_edge_dx=core_dx,
                core_at_upper_end=False,
            )
        )

    if segments:
        faces = np.concatenate([segments[0]] + [segment[1:]
                               for segment in segments[1:]])
    else:
        faces = np.linspace(0.0, length, n + 1)

    faces[0] = 0.0
    faces[-1] = length
    return faces, core_start, core_end


def build_nonuniform_grid_metadata(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    beta_x: float,
    beta_y: float,
    x_min: float = 0.0,
    y_min: float = 0.0,
    uniform_x_start: float | None = None,
    uniform_x_end: float | None = None,
    uniform_y_start: float | None = None,
    uniform_y_end: float | None = None,
) -> dict:
    """Build serializable metadata for a 2-D non-uniform Cartesian grid.

    The nonuniform grid uses a constant-spacing central core with stretched
    outer regions toward the boundaries.
    """
    x_max = float(x_min + lx)
    y_max = float(y_min + ly)
    if uniform_x_start is None or uniform_x_end is None:
        raise ValueError(
            "Nonuniform grid generation requires uniform_x_start and uniform_x_end"
        )
    x_interval = (
        float(uniform_x_start) - float(x_min),
        float(uniform_x_end) - float(x_min),
    )
    xf, band_start_x, band_end_x = stretched_faces_center_uniform(
        nx,
        lx,
        beta_x,
        core_interval=x_interval,
    )

    mirrored_y = y_min < 0.0 < y_max
    if mirrored_y:
        if not np.isclose(y_min + y_max, 0.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                "center-uniform mirrored y-spacing requires y-domain symmetric about 0"
            )
        if ny % 2 != 0:
            raise ValueError(
                "center-uniform mirrored y-spacing requires even ny"
            )

        if uniform_y_start is None or uniform_y_end is None:
            raise ValueError(
                "Nonuniform grid generation requires uniform_y_start and uniform_y_end"
            )
        y0 = float(uniform_y_start)
        y1 = float(uniform_y_end)
        if not (y0 < 0.0 < y1 and np.isclose(abs(y0), abs(y1), atol=1e-10, rtol=0.0)):
            raise ValueError(
                "uniform_y_start/uniform_y_end must be symmetric around 0 (e.g. -a, a)"
            )
        y_core_interval = (0.0, abs(y1))

        upper_faces, y_core_start_local, y_core_end_local = stretched_faces_center_uniform(
            ny // 2,
            y_max,
            beta_y,
            core_interval=y_core_interval,
        )
        yf = np.concatenate((-upper_faces[::-1], upper_faces[1:]))
        band_start_y = -y_core_end_local
        band_end_y = y_core_end_local
    else:
        if uniform_y_start is None or uniform_y_end is None:
            raise ValueError(
                "Nonuniform grid generation requires uniform_y_start and uniform_y_end"
            )
        y_interval = (
            float(uniform_y_start) - float(y_min),
            float(uniform_y_end) - float(y_min),
        )
        yf, band_start_y, band_end_y = stretched_faces_center_uniform(
            ny,
            ly,
            beta_y,
            core_interval=y_interval,
        )

    xf = xf + float(x_min)
    if not mirrored_y:
        yf = yf + float(y_min)
    band_start_x += float(x_min)
    band_end_x += float(x_min)
    if not mirrored_y:
        band_start_y += float(y_min)
        band_end_y += float(y_min)

    grid = CartesianGrid(nx=nx, ny=ny, lx=lx, ly=ly,
                         x_min=x_min, y_min=y_min, xf=xf, yf=yf)
    metadata = grid.to_metadata()
    metadata["grid_type"] = "nonuniform"
    metadata["beta_x"] = float(beta_x)
    metadata["beta_y"] = float(beta_y)
    metadata["nonuniform_mode"] = "center-uniform"
    metadata["uniform_x_start"] = np.nan if uniform_x_start is None else float(
        uniform_x_start)
    metadata["uniform_x_end"] = np.nan if uniform_x_end is None else float(
        uniform_x_end)
    metadata["uniform_y_start"] = np.nan if uniform_y_start is None else float(
        uniform_y_start)
    metadata["uniform_y_end"] = np.nan if uniform_y_end is None else float(
        uniform_y_end)
    metadata["band_start_x"] = float(band_start_x)
    metadata["band_end_x"] = float(band_end_x)
    metadata["band_start_y"] = float(band_start_y)
    metadata["band_end_y"] = float(band_end_y)
    metadata["dx"] = grid.dx_cells.copy()
    metadata["dy"] = grid.dy_cells.copy()
    return metadata
