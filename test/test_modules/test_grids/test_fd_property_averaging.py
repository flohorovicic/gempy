import numpy as np
import pytest

from test.conftest import TEST_SPEED, TestSpeed
import gempy as gp
from gempy.modules.grids.fd_property_averaging import RectilinearGrid, upscale_octree_to_rectilinear

pytestmark = pytest.mark.skipif(TEST_SPEED.value < TestSpeed.SECONDS.value, reason="Global test speed below this test value.")

EXTENT = [0, 1000, 0, 1000, 0, 500]


@pytest.fixture(scope='module')
def flat_layer_model() -> gp.data.GeoModel:
    """Single flat horizontal interface at z=275 computed on an octree grid."""
    geo_model = gp.create_geomodel(
        project_name='flat_layer_upscaling',
        extent=EXTENT,
        refinement=5,
        structural_frame=gp.data.StructuralFrame.initialize_default_structure()
    )
    gp.add_surface_points(geo_model, x=[200, 800], y=[300, 700], z=[275, 275],
                          elements_names=['surface1', 'surface1'])
    gp.add_orientations(geo_model, x=[500], y=[500], z=[275],
                        elements_names=['surface1'], pole_vector=[[0, 0, 1]])
    # Input transform: the engine octree extent is then in transformed space, so the
    # upscaling calls below must pass the world extent explicitly (the common case).
    geo_model.update_transform(gp.data.GlobalAnisotropy.NONE)
    gp.compute_model(geo_model)
    return geo_model


def test_volume_fractions_and_mixed_cells(flat_layer_model):
    fd_grid = RectilinearGrid.from_resolution(EXTENT, [10, 10, 5])
    upscaled = upscale_octree_to_rectilinear(flat_layer_model.solutions, fd_grid, model_extent=EXTENT)

    assert upscaled.fractions.shape == (10, 10, 5, upscaled.unit_ids.size)
    np.testing.assert_allclose(upscaled.fractions.sum(axis=-1), 1.0)
    # The flat interface at z=275 cuts exactly the third z-layer of cells (z in [200, 300])
    assert upscaled.mixed_cells.sum() == 10 * 10
    assert upscaled.mixed_cells[:, :, 2].all()


def test_averaging_laws_and_bounds(flat_layer_model):
    fd_grid = RectilinearGrid.from_resolution(EXTENT, [10, 10, 5])
    upscaled = upscale_octree_to_rectilinear(flat_layer_model.solutions, fd_grid, model_extent=EXTENT)
    k_units = {int(unit_id): value for unit_id, value in zip(upscaled.unit_ids, [4.0, 1.0])}

    k_center = upscaled.map_property(k_units, 'center')
    k_arith = upscaled.map_property(k_units, 'arithmetic')
    k_harm = upscaled.map_property(k_units, 'harmonic')
    k_geo = upscaled.map_property(k_units, 'geometric')

    assert np.isin(k_center, list(k_units.values())).all()
    # Fundamental bounds: harmonic <= geometric <= arithmetic
    assert np.all(k_harm <= k_geo + 1e-12)
    assert np.all(k_geo <= k_arith + 1e-12)
    # Power mean limits reproduce arithmetic and harmonic
    np.testing.assert_allclose(upscaled.map_property(k_units, 'power', p=1.0), k_arith)
    np.testing.assert_allclose(upscaled.map_property(k_units, 'power', p=-1.0), k_harm)
    np.testing.assert_allclose(upscaled.map_property(k_units, 'power', p=0.0), k_geo)


def test_full_tensor_flat_interface(flat_layer_model):
    fd_grid = RectilinearGrid.from_resolution(EXTENT, [10, 10, 5])
    upscaled = upscale_octree_to_rectilinear(flat_layer_model.solutions, fd_grid, model_extent=EXTENT)
    k_units = {int(unit_id): value for unit_id, value in zip(upscaled.unit_ids, [4.0, 1.0])}

    tensor = upscaled.full_tensor(k_units)
    k_arith = upscaled.map_property(k_units, 'arithmetic')
    k_harm = upscaled.map_property(k_units, 'harmonic')

    # Horizontal interface -> normal along z -> laminate tensor diag(k_a, k_a, k_h)
    mixed_cell = (5, 5, 2)
    np.testing.assert_allclose(np.abs(upscaled.interface_normals[mixed_cell]), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(
        tensor[mixed_cell],
        np.diag([k_arith[mixed_cell], k_arith[mixed_cell], k_harm[mixed_cell]]),
        atol=1e-10
    )
    # Pure cells stay isotropic
    pure_cell = (5, 5, 0)
    np.testing.assert_allclose(tensor[pure_cell], np.eye(3) * k_arith[pure_cell])

    principal_values, principal_directions = upscaled.principal_axes_tensor(k_units)
    np.testing.assert_allclose(
        np.sort(principal_values[mixed_cell]),
        np.sort([k_harm[mixed_cell], k_arith[mixed_cell], k_arith[mixed_cell]]),
        atol=1e-10
    )
    diagonal = upscaled.diagonal_tensor(k_units)
    np.testing.assert_allclose(diagonal, np.einsum('...ii->...i', tensor))


def test_non_uniform_rectilinear_grid(flat_layer_model):
    fd_grid = RectilinearGrid.from_spacing(
        origin=[0, 0, 0],
        dx=[100] * 10,
        dy=[200] * 5,
        dz=[50, 50, 100, 100, 100, 50, 50],
    )
    upscaled = upscale_octree_to_rectilinear(flat_layer_model.solutions, fd_grid, model_extent=EXTENT)
    assert tuple(upscaled.grid.resolution) == (10, 5, 7)
    np.testing.assert_allclose(upscaled.fractions.sum(axis=-1), 1.0)
    assert upscaled.mixed_cells.any()


def test_grid_outside_extent_raises(flat_layer_model):
    fd_grid = RectilinearGrid.from_resolution([0, 2000, 0, 1000, 0, 500], [10, 10, 5])
    with pytest.raises(ValueError, match='exceeds the model extent'):
        upscale_octree_to_rectilinear(flat_layer_model.solutions, fd_grid, model_extent=EXTENT)


def test_transformed_model_requires_explicit_extent(flat_layer_model):
    # The fixture model has an input transform: without model_extent the engine's
    # transformed-space extent does not contain the world-coordinate FD grid.
    fd_grid = RectilinearGrid.from_resolution(EXTENT, [10, 10, 5])
    with pytest.raises(ValueError, match='model_extent=geo_model.grid.extent'):
        upscale_octree_to_rectilinear(flat_layer_model.solutions, fd_grid)
