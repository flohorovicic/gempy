﻿from typing import Union

import numpy as np

from gempy import GeoModel, Grid
from gempy.core.grid import GridTypes
from gempy.core.grid_modules.topography import Topography
from gempy.modules.grids.create_topography import create_random_topography


def set_section_grid(grid: Grid, section_dict: dict):
    if grid.sections is None:
        grid.create_section_grid(section_dict=section_dict)
    else:
        grid.sections.set_sections(section_dict,
                                   regular_grid=grid.regular_grid)

    set_active_grid(grid, [GridTypes.SECTIONS])
    return grid.sections


def set_topography_from_random(grid: Grid, fractal_dimension: float = 2.0, d_z: Union[float, None] = None,
                               topography_resolution: Union[list, None] = None):
    if topography_resolution is None:
        topography_resolution = grid.regular_grid.resolution

    random_topography: np.ndarray = create_random_topography(
        extent=grid.regular_grid.extent,
        resolution=topography_resolution,
        dz=d_z,
        fractal_dimension=fractal_dimension
    )
    
    grid.topography = Topography(
        regular_grid=grid.regular_grid,
        values_2d=random_topography
    )

    set_active_grid(grid, [GridTypes.TOPOGRAPHY])
    return grid.topography


def set_topography_from_gdal():
    raise NotImplementedError("This is not implemented yet")


def set_topography_from_array():
    raise NotImplementedError("This is not implemented yet")


def set_active_grid(grid: Grid, grid_type: list[GridTypes], reset: bool = False):
    if reset is True:
        grid.deactivate_all_grids()
    for grid_type in grid_type:
        grid.active_grids[grid_type.value] = True

    print(f'Active grids: {grid.grid_types[grid.active_grids]}')

    return grid