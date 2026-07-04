"""
Property averaging from GemPy octree solutions onto rectilinear FD grids.

This module maps geological units from a GemPy model — computed with adaptive
octree refinement — onto a rectilinear finite-difference simulation grid
(e.g. for SHEMAT-Suite-style geothermal process simulations).

Instead of assigning each FD cell the property of the unit at its cell center,
cells that are cut by a unit interface are populated with properly averaged
(upscaled) properties. The octree refinement provides the sub-cell resolution
needed to compute volume fractions and interface orientations cheaply: the
interpolation is only evaluated densely near interfaces, and the finest octree
level is expanded to a virtual fine grid from which the averaging is computed.

Supported averaging laws (for scalar properties such as thermal or hydraulic
conductivity):

* ``center``     — direct mapping of the unit at the cell center (no averaging)
* ``arithmetic`` — volume-weighted arithmetic mean (upper/Voigt bound,
  exact for flow parallel to a layered interface)
* ``harmonic``   — volume-weighted harmonic mean (lower/Reuss bound,
  exact for flow normal to a layered interface)
* ``geometric``  — volume-weighted geometric mean (common heuristic for
  disordered media)
* ``power``      — generalized power (Hölder) mean with exponent ``p``
  (p=1 arithmetic, p→0 geometric, p=-1 harmonic)

In addition, a full effective conductivity **tensor** can be determined per
cell: for a cell cut by an interface with unit normal ``n`` the classic
layered-medium (laminate) result applies, combining the harmonic mean normal
to the interface with the arithmetic mean parallel to it:

    K_eff = k_harm * (n ⊗ n) + k_arith * (I - n ⊗ n)

The interface normal is obtained from the gradient of GemPy's implicit scalar
field averaged over the cell. Since many FD codes (SHEMAT-Suite among them)
only support diagonal conductivity tensors aligned with the grid axes, the
tensor can be mapped to main axes in two ways:

* ``principal_axes_tensor`` — eigen-decomposition into principal values and
  directions (lossless, but the directions are generally not grid-aligned)
* ``diagonal_tensor``       — keep only the diagonal entries kxx, kyy, kzz in
  grid coordinates (what is effectively used when a simulator only accepts
  per-axis conductivities; discards off-diagonal coupling)

Example
-------
>>> import gempy as gp
>>> from gempy.modules.grids.fd_property_averaging import (
...     RectilinearGrid, upscale_octree_to_rectilinear)
>>> # model computed with octree refinement:
>>> # geo_model = gp.create_geomodel(..., refinement=6); gp.compute_model(geo_model)
>>> fd_grid = RectilinearGrid.from_resolution(geo_model.grid.extent, [20, 20, 10])
>>> upscaled = upscale_octree_to_rectilinear(geo_model.solutions, fd_grid)
>>> k_units = {1: 3.0, 2: 1.0}  # conductivity per unit id
>>> k_center = upscaled.map_property(k_units, method='center')
>>> k_arith = upscaled.map_property(k_units, method='arithmetic')
>>> K = upscaled.full_tensor(k_units)              # (nx, ny, nz, 3, 3)
>>> k_diag = upscaled.diagonal_tensor(k_units)     # (nx, ny, nz, 3)
"""

import dataclasses
from typing import Optional, Sequence, Union

import numpy as np


# region Rectilinear grid ====================================================

@dataclasses.dataclass
class RectilinearGrid:
    """Axis-aligned rectilinear grid defined by its cell edge coordinates.

    Spacing may vary per axis (as common in FD simulators such as
    SHEMAT-Suite), but cells are always axis-aligned boxes.
    """
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self):
        for name in ('x_edges', 'y_edges', 'z_edges'):
            edges = np.asarray(getattr(self, name), dtype='float64')
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError(f'{name} must be a 1D array with at least 2 entries')
            if not np.all(np.diff(edges) > 0):
                raise ValueError(f'{name} must be strictly increasing')
            setattr(self, name, edges)

    @classmethod
    def from_resolution(cls, extent: Sequence[float], resolution: Sequence[int]) -> 'RectilinearGrid':
        """Uniform grid from a GemPy-style extent [x0, x1, y0, y1, z0, z1]."""
        extent = np.asarray(extent, dtype='float64')
        resolution = np.asarray(resolution, dtype='int64')
        return cls(
            x_edges=np.linspace(extent[0], extent[1], resolution[0] + 1),
            y_edges=np.linspace(extent[2], extent[3], resolution[1] + 1),
            z_edges=np.linspace(extent[4], extent[5], resolution[2] + 1),
        )

    @classmethod
    def from_spacing(cls, origin: Sequence[float], dx: Sequence[float], dy: Sequence[float],
                     dz: Sequence[float]) -> 'RectilinearGrid':
        """Grid from an origin and per-axis cell size arrays (SHEMAT-Suite style delx/dely/delz)."""
        origin = np.asarray(origin, dtype='float64')
        return cls(
            x_edges=origin[0] + np.concatenate([[0.], np.cumsum(dx)]),
            y_edges=origin[1] + np.concatenate([[0.], np.cumsum(dy)]),
            z_edges=origin[2] + np.concatenate([[0.], np.cumsum(dz)]),
        )

    @property
    def resolution(self) -> np.ndarray:
        return np.array([self.x_edges.size - 1, self.y_edges.size - 1, self.z_edges.size - 1])

    @property
    def extent(self) -> np.ndarray:
        return np.array([self.x_edges[0], self.x_edges[-1],
                         self.y_edges[0], self.y_edges[-1],
                         self.z_edges[0], self.z_edges[-1]])

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def centers(self) -> np.ndarray:
        """Cell center coordinates, shape (n_cells, 3), x-index slowest ('ij' C-order)."""
        g = np.meshgrid(self.x_centers, self.y_centers, self.z_centers, indexing='ij')
        return np.vstack(tuple(map(np.ravel, g))).T


# endregion

# region Upscaling result ====================================================

_AVERAGING_METHODS = ('center', 'arithmetic', 'harmonic', 'geometric', 'power')


@dataclasses.dataclass
class UpscaledProperties:
    """Volume fractions and interface geometry of a geological model on a rectilinear grid.

    All arrays are C-ordered with shape (nx, ny, nz, ...).
    """
    grid: RectilinearGrid
    unit_ids: np.ndarray            #: sorted unique unit ids, shape (n_units,)
    fractions: np.ndarray           #: volume fraction per unit, shape (nx, ny, nz, n_units)
    center_unit_id: np.ndarray      #: unit id at the cell center, shape (nx, ny, nz)
    interface_normals: np.ndarray   #: unit vector of the mean scalar-field gradient per cell, shape (nx, ny, nz, 3); NaN where the gradient vanishes

    @property
    def mixed_cells(self) -> np.ndarray:
        """Boolean mask of cells cut by at least one interface, shape (nx, ny, nz)."""
        return np.max(self.fractions, axis=-1) < 1.0 - 1e-12

    def _values_per_unit(self, unit_values: Union[dict, Sequence[float]]) -> np.ndarray:
        """Normalize a {unit_id: value} dict or sequence (ordered as unit_ids) to an array."""
        if isinstance(unit_values, dict):
            missing = set(self.unit_ids.tolist()) - set(unit_values.keys())
            if missing:
                raise ValueError(f'No property value given for unit ids {sorted(missing)}')
            values = np.array([unit_values[unit_id] for unit_id in self.unit_ids], dtype='float64')
        else:
            values = np.asarray(unit_values, dtype='float64')
            if values.shape != self.unit_ids.shape:
                raise ValueError(f'Expected {self.unit_ids.size} values (one per unit id '
                                 f'{self.unit_ids.tolist()}), got shape {values.shape}')
        return values

    # ------------------------------------------------------------------ scalar averaging

    def map_property(self, unit_values: Union[dict, Sequence[float]], method: str = 'arithmetic',
                     p: Optional[float] = None) -> np.ndarray:
        """Map a scalar property per unit onto the grid, shape (nx, ny, nz).

        Args:
            unit_values: {unit_id: value} dict or sequence ordered as ``unit_ids``.
            method: one of 'center', 'arithmetic', 'harmonic', 'geometric', 'power'.
            p: exponent for the power (Hölder) mean; required if method='power'.
        """
        if method not in _AVERAGING_METHODS:
            raise ValueError(f'Unknown averaging method {method!r}, must be one of {_AVERAGING_METHODS}')
        values = self._values_per_unit(unit_values)

        if method == 'center':
            index = np.searchsorted(self.unit_ids, self.center_unit_id)
            return values[index]
        if method == 'arithmetic':
            return self.fractions @ values
        if method == 'harmonic':
            return 1.0 / (self.fractions @ (1.0 / values))
        if method == 'geometric':
            return np.exp(self.fractions @ np.log(values))
        # power mean
        if p is None:
            raise ValueError("method='power' requires the exponent p")
        if p == 0:
            return np.exp(self.fractions @ np.log(values))
        return (self.fractions @ values ** p) ** (1.0 / p)

    # ------------------------------------------------------------------ tensor determination

    def full_tensor(self, unit_values: Union[dict, Sequence[float]]) -> np.ndarray:
        """Full effective conductivity tensor per cell, shape (nx, ny, nz, 3, 3).

        Pure cells get an isotropic tensor k*I. Cells cut by an interface with
        unit normal n are treated as a local laminate: harmonic mean normal to
        the interface, arithmetic mean parallel to it,

            K = k_harm * (n ⊗ n) + k_arith * (I - n ⊗ n)

        which is exact for stationary flow through a layered medium and the
        standard upscaling rule for interface cells. In (rare) cells where an
        interface normal could not be determined the tensor falls back to the
        isotropic geometric mean.
        """
        k_arith = self.map_property(unit_values, 'arithmetic')
        k_harm = self.map_property(unit_values, 'harmonic')
        k_geo = self.map_property(unit_values, 'geometric')

        n = self.interface_normals
        has_normal = ~np.isnan(n[..., 0])
        mixed = self.mixed_cells & has_normal
        isotropic_fallback = self.mixed_cells & ~has_normal

        nnt = n[..., :, None] * n[..., None, :]  # n ⊗ n, NaN where undefined
        identity = np.eye(3)

        tensor = np.zeros(self.fractions.shape[:3] + (3, 3))
        # pure cells: k * I (arithmetic == harmonic == k there)
        tensor[...] = k_arith[..., None, None] * identity
        tensor[mixed] = (k_harm[mixed, None, None] * nnt[mixed]
                         + k_arith[mixed, None, None] * (identity - nnt[mixed]))
        tensor[isotropic_fallback] = k_geo[isotropic_fallback, None, None] * identity
        return tensor

    def principal_axes_tensor(self, unit_values: Union[dict, Sequence[float]]) -> tuple:
        """Eigen-decomposition of the full tensor into principal values and directions.

        Returns:
            (principal_values, principal_directions) with shapes
            (nx, ny, nz, 3) and (nx, ny, nz, 3, 3); ``principal_directions[..., :, i]``
            is the direction of ``principal_values[..., i]`` (ascending order).
        """
        tensor = self.full_tensor(unit_values)
        return np.linalg.eigh(tensor)

    def diagonal_tensor(self, unit_values: Union[dict, Sequence[float]]) -> np.ndarray:
        """Grid-axis-aligned diagonal (kxx, kyy, kzz) of the full tensor, shape (nx, ny, nz, 3).

        This is the mapping to main axes used when the target simulator only
        accepts per-axis conductivities: off-diagonal terms are discarded.
        """
        tensor = self.full_tensor(unit_values)
        return np.einsum('...ii->...i', tensor)


# endregion

# region Upscaling from octree solutions =====================================

def _to_numpy(array) -> np.ndarray:
    """Convert engine output (numpy array or torch tensor) to a numpy array."""
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def upscale_octree_to_rectilinear(solutions, grid: RectilinearGrid, level: Optional[int] = None,
                                  scalar_n: int = -1,
                                  model_extent: Optional[Sequence[float]] = None) -> UpscaledProperties:
    """Aggregate the finest octree level of a GemPy solution onto a rectilinear grid.

    The octree solution is expanded to a virtual dense grid at ``level`` (default:
    the finest computed level). Each fine voxel is assigned to the FD cell that
    contains its center; per-cell unit volume fractions and mean scalar-field
    gradients (interface normals) are accumulated from those voxels. The result
    is exact where FD cell faces coincide with fine voxel faces and accurate to
    the fine voxel size otherwise, so the FD grid may be any rectilinear grid
    within the model extent — it does not need to match the octree layout.

    Args:
        solutions: ``gempy_engine`` :obj:`Solutions` (i.e. ``geo_model.solutions``)
            computed on an octree grid.
        grid: target rectilinear FD grid; must lie within the model extent.
        level: octree level to expand (defaults to the finest available).
        scalar_n: index of the structural stack whose scalar field provides the
            interface normals (default: last).
        model_extent: world-coordinate extent [x0, x1, y0, y1, z0, z1] of the
            octree grid, i.e. ``geo_model.grid.extent``. Required whenever the
            model has an input transform (e.g. after ``update_transform``),
            because the engine then stores the octree extent in transformed
            space. Only translation/scaling transforms are supported — a
            rotated model is not rectilinear in world coordinates.
    """
    from gempy_engine.core.data.output.blocks_value_type import ValueType
    from gempy_engine.modules.octrees_topology.octrees_topology_interface import \
        get_regular_grid_value_for_level

    octrees = solutions.octrees_output
    if level is None:
        level = len(octrees) - 1

    ids_fine = _to_numpy(get_regular_grid_value_for_level(octrees, level=level, value_type=ValueType.ids))
    ids_fine = np.rint(ids_fine).astype('int64')
    scalar_fine = _to_numpy(get_regular_grid_value_for_level(octrees, level=level, value_type=ValueType.scalar,
                                                             scalar_n=scalar_n))

    if model_extent is not None:
        model_extent = np.asarray(model_extent, dtype='float64')
    else:
        # gempy_engine <= 2025.2 stores the root grid as `grid_centers`, newer versions as `grid`
        root_grid = octrees[0].grid_centers if hasattr(octrees[0], 'grid_centers') else octrees[0].grid
        model_extent = _to_numpy(root_grid.octree_grid.orthogonal_extent).astype('float64')
    fine_shape = np.asarray(ids_fine.shape)

    tol = 1e-6 * np.max(np.abs(model_extent))  # engine pads the extent by 1e-6
    outside = (grid.extent[[0, 2, 4]] < model_extent[[0, 2, 4]] - tol).any() or \
              (grid.extent[[1, 3, 5]] > model_extent[[1, 3, 5]] + tol).any()
    if outside:
        raise ValueError(f'FD grid extent {grid.extent} exceeds the model extent {model_extent}. '
                         f'If the model uses an input transform, pass the world-coordinate extent '
                         f'explicitly: model_extent=geo_model.grid.extent')

    # Fine voxel center coordinates and spacing per axis
    fine_coords, fine_spacing = [], []
    for axis in range(3):
        lo, hi = model_extent[2 * axis], model_extent[2 * axis + 1]
        d = (hi - lo) / fine_shape[axis]
        fine_coords.append(lo + d * (np.arange(fine_shape[axis]) + 0.5))
        fine_spacing.append(d)

    # Interface normals from the scalar field gradient on the fine grid
    grad_fine = np.stack(np.gradient(scalar_fine, *fine_spacing), axis=-1)

    # Assign each fine voxel (by center) to an FD cell; -1 marks voxels outside the FD grid
    fd_index_1d = []
    for axis, edges in enumerate((grid.x_edges, grid.y_edges, grid.z_edges)):
        idx = np.searchsorted(edges, fine_coords[axis], side='right') - 1
        idx[(fine_coords[axis] < edges[0]) | (fine_coords[axis] > edges[-1])] = -1
        idx[idx == edges.size - 1] = edges.size - 2  # centers exactly on the last edge
        fd_index_1d.append(idx)

    nx, ny, nz = grid.resolution
    ix, iy, iz = np.meshgrid(*fd_index_1d, indexing='ij')
    inside = (ix >= 0) & (iy >= 0) & (iz >= 0)
    cell_flat = (ix * ny + iy) * nz + iz  # C-order flat index, valid where inside

    unit_ids = np.unique(ids_fine)
    unit_index_fine = np.searchsorted(unit_ids, ids_fine)
    n_cells, n_units = nx * ny * nz, unit_ids.size

    # Volume fractions: count fine voxels per (cell, unit). Fine voxels are uniform
    # in size, so counts are directly proportional to volume.
    counts = np.zeros((n_cells, n_units))
    np.add.at(counts, (cell_flat[inside], unit_index_fine[inside]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    empty = totals[:, 0] == 0
    if empty.any():
        raise ValueError(f'{empty.sum()} FD cells contain no fine octree voxel. Increase the '
                         f'octree refinement or coarsen the FD grid.')
    fractions = (counts / totals).reshape(nx, ny, nz, n_units)

    # Interface normals: mean scalar-field gradient per cell, normalized
    grad_sum = np.zeros((n_cells, 3))
    np.add.at(grad_sum, cell_flat[inside], grad_fine[inside])
    norm = np.linalg.norm(grad_sum, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normals = grad_sum / norm[:, None]
    normals[norm < 1e-30] = np.nan
    normals = normals.reshape(nx, ny, nz, 3)

    # Cell-center mapping: unit id of the fine voxel containing each FD cell center
    center_index_1d = [
        np.clip(((centers - model_extent[2 * axis]) / fine_spacing[axis]).astype('int64'),
                0, fine_shape[axis] - 1)
        for axis, centers in enumerate((grid.x_centers, grid.y_centers, grid.z_centers))
    ]
    cx, cy, cz = np.meshgrid(*center_index_1d, indexing='ij')
    center_unit_id = ids_fine[cx, cy, cz]

    return UpscaledProperties(
        grid=grid,
        unit_ids=unit_ids,
        fractions=fractions,
        center_unit_id=center_unit_id,
        interface_normals=normals,
    )

# endregion
