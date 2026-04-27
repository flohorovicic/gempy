"""
Grid Structures Comparison
==========================

This tutorial compares the different grid types available in GemPy for
evaluating and extracting geological models. Understanding when to use
each grid type is essential for efficient geological modeling.

Grid Types Overview:
- **Dense Grid**: Uniform 3D grid with fixed spacing
- **Octree Grid**: Adaptive hierarchical grid with variable resolution
- **Custom Grid**: Arbitrary XYZ coordinates for specific evaluations
- **Topography**: 2D surface representing terrain elevation
- **Sections**: 2D vertical cross-sections through the model
- **Centered Grid**: Clusters around center points (for geophysics)
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt

import gempy as gp
from gempy.core.data import Grid
from gempy.core.data.grid_modules import RegularGrid, CustomGrid
from gempy.core.data.enumerators import ExampleModel

np.random.seed(42)

# %%
# Part 1: Understanding the Grid Container
# -----------------------------------------
#
# The ``Grid`` class is a container that manages multiple grid types through
# a composition pattern. Each grid type can be activated/deactivated independently,
# and the main ``Grid.values`` array concatenates all active grids.

# %%
# Let's start by creating an empty grid and examining its structure:

grid = Grid()

print("Grid Types available:")
print(grid.GridTypes)
print(f"\nActive grids: {grid.active_grids}")
print(f"Grid values shape: {grid.values.shape}")

# %%
# Part 2: Dense (Regular) Grid
# ----------------------------
#
# A dense grid creates a uniform 3D grid with fixed spacing in all directions.
# This is the most common grid type for standard geological modeling.
#
# **Characteristics:**
#
# - Uniform resolution: all voxels have the same size
# - Fixed point count: nx × ny × nz
# - Simple to understand and visualize
# - Memory usage scales with resolution

extent = np.array([0, 1000, 0, 1000, 0, 500])
resolution = np.array([20, 20, 10])  # 4000 points total

dense_grid = RegularGrid(extent, resolution)

print("Dense Grid Properties:")
print(f"  Extent: {extent}")
print(f"  Resolution: {resolution}")
print(f"  Total points: {np.prod(resolution)} ({resolution[0]}×{resolution[1]}×{resolution[2]})")
print(f"  Cell spacing: dx={dense_grid.dx:.1f}, dy={dense_grid.dy:.1f}, dz={dense_grid.dz:.1f}")
print(f"  Values shape: {dense_grid.values.shape}")

# %%
# Let's visualize the dense grid points in a slice:

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# XY slice (top view)
z_slice = resolution[2] // 2
xy_points = dense_grid.values.reshape(resolution[0], resolution[1], resolution[2], 3)[:, :, z_slice, :]
axes[0].scatter(xy_points[:, :, 0].flatten(), xy_points[:, :, 1].flatten(), s=5, alpha=0.6)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title(f'XY Slice (Z={z_slice})')
axes[0].set_aspect('equal')

# XZ slice (front view)
y_slice = resolution[1] // 2
xz_points = dense_grid.values.reshape(resolution[0], resolution[1], resolution[2], 3)[:, y_slice, :, :]
axes[1].scatter(xz_points[:, :, 0].flatten(), xz_points[:, :, 2].flatten(), s=5, alpha=0.6)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title(f'XZ Slice (Y={y_slice})')

# YZ slice (side view)
x_slice = resolution[0] // 2
yz_points = dense_grid.values.reshape(resolution[0], resolution[1], resolution[2], 3)[x_slice, :, :, :]
axes[2].scatter(yz_points[:, :, 1].flatten(), yz_points[:, :, 2].flatten(), s=5, alpha=0.6)
axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_title(f'YZ Slice (X={x_slice})')

plt.tight_layout()
plt.suptitle('Dense Grid Structure', y=1.02)
plt.show()

# %%
# Part 3: Octree Grid
# -------------------
#
# An octree grid uses hierarchical subdivision to provide adaptive resolution.
# Areas of interest can have finer resolution while less important areas remain coarse.
#
# **Characteristics:**
#
# - Adaptive resolution based on refinement levels
# - Memory efficient for large models
# - Better for complex geometries
# - Resolution doubles with each level: final_res = base_res × 2^(levels-1)
#
# In GemPy, octree grids are created through the Grid initialization:

# Create octree grid with 3 refinement levels
octree_grid = Grid.init_octree_grid(
    extent=extent,
    octree_levels=3
)

print("Octree Grid Properties:")
print(f"  Octree levels: 3")
print(f"  Base resolution: {octree_grid.regular_grid.resolution}")
print(f"  Active grids: {octree_grid.active_grids}")
print(f"  Grid values shape: {octree_grid.values.shape}")

# %%
# Part 4: Custom Grid
# -------------------
#
# A custom grid allows you to specify arbitrary XYZ coordinates where the
# geological model should be evaluated. This is the most flexible grid type.
#
# **Characteristics:**
#
# - Completely unstructured - any XYZ coordinates
# - No resolution or extent constraints
# - Ideal for specific point evaluations
# - Most memory efficient for sparse sampling
#
# **Common use cases:**
#
# - Borehole/drillhole locations
# - Scattered measurement points
# - Well path analysis
# - Specific point queries

# Define custom points - e.g., borehole locations
borehole_coords = np.array([
    [200, 500, 400],   # Borehole 1 - top
    [200, 500, 300],   # Borehole 1 - middle
    [200, 500, 100],   # Borehole 1 - bottom
    [500, 500, 450],   # Borehole 2 - top
    [500, 500, 250],   # Borehole 2 - middle
    [500, 500, 50],    # Borehole 2 - bottom
    [800, 500, 400],   # Borehole 3 - top
    [800, 500, 200],   # Borehole 3 - middle
    [800, 500, 50],    # Borehole 3 - bottom
])

custom_grid = CustomGrid(values=borehole_coords)

print("Custom Grid Properties:")
print(f"  Number of points: {custom_grid.length}")
print(f"  Values shape: {custom_grid.values.shape}")
print(f"  Points:\n{custom_grid.values}")

# %%
# Visualize custom grid points:

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot borehole paths
for i in range(0, len(borehole_coords), 3):
    borehole = borehole_coords[i:i+3]
    ax.plot(borehole[:, 0], borehole[:, 1], borehole[:, 2], 'o-', markersize=8, linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Custom Grid: Borehole Locations')
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_zlim(extent[4], extent[5])
plt.show()

# %%
# Part 5: Using Multiple Grid Types Together
# ------------------------------------------
#
# GemPy allows activating multiple grid types simultaneously. All active
# grids are concatenated into a single ``Grid.values`` array for computation.

# Create a grid with multiple types
multi_grid = Grid(extent=extent, resolution=[10, 10, 5])  # Start with dense grid

# Add custom grid
gp.set_custom_grid(multi_grid, borehole_coords)

# Add random topography
gp.set_topography_from_random(
    grid=multi_grid,
    fractal_dimension=1.5,
    d_z=np.array([400, 500]),
    topography_resolution=np.array([20, 20])
)

# Add cross-sections
gp.set_section_grid(
    grid=multi_grid,
    section_dict={
        'section_NS': ([500, 0], [500, 1000], [50, 25]),      # North-South section
        'section_EW': ([0, 500], [1000, 500], [50, 25]),      # East-West section
        'section_diag': ([0, 0], [1000, 1000], [70, 25]),     # Diagonal section
    }
)

print("Multi-Grid Configuration:")
print(f"  Active grids: {multi_grid.active_grids}")
print(f"  Total grid values shape: {multi_grid.values.shape}")
print(f"  Grid lengths (cumulative indices): {multi_grid.length}")

# %%
# Part 6: Practical Comparison with a Geological Model
# ----------------------------------------------------
#
# Let's create a simple geological model and evaluate it using different grid types.

# Generate an anticline example model
geo_model = gp.generate_example_model(
    example_model=ExampleModel.ANTICLINE,
    compute_model=False
)

print("Geological Model Created:")
print(f"  Project: {geo_model.meta.name}")
print(f"  Extent: {geo_model.grid.regular_grid.extent}")
print(f"  Initial grid type: {geo_model.grid.active_grids}")

# %%
# Evaluate at custom points using ``compute_model_at``:

# Define evaluation points along a vertical profile
profile_points = np.array([
    [500, 500, z] for z in np.linspace(100, 900, 20)
])

# Compute model at these specific points
result_at_points = gp.compute_model_at(
    gempy_model=geo_model,
    at=profile_points
)

print("\nModel evaluation at custom points:")
print(f"  Number of evaluation points: {len(profile_points)}")
print(f"  Lithology IDs at points: {result_at_points}")

# %%
# Visualize the vertical profile results:

fig, ax = plt.subplots(figsize=(8, 6))

colors = plt.cm.viridis(np.linspace(0, 1, int(result_at_points.max()) + 1))
for i, (point, lith_id) in enumerate(zip(profile_points, result_at_points)):
    ax.scatter(point[2], i, c=[colors[int(lith_id)]], s=100, edgecolors='black')

ax.set_xlabel('Z Coordinate')
ax.set_ylabel('Point Index')
ax.set_title('Lithology Along Vertical Profile (X=500, Y=500)')
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Lithology ID')
plt.show()

# %%
# Part 7: Grid Type Comparison Summary
# ------------------------------------
#
# Here's a comparison table of the different grid types:
#
# +----------------+------------------+------------------+-------------------+
# | Aspect         | Dense            | Octree           | Custom            |
# +================+==================+==================+===================+
# | Structure      | Uniform Cartesian| Hierarchical     | Unstructured      |
# +----------------+------------------+------------------+-------------------+
# | Point count    | Fixed (nx×ny×nz) | Variable         | Arbitrary         |
# +----------------+------------------+------------------+-------------------+
# | Spacing        | Uniform          | Variable         | None              |
# +----------------+------------------+------------------+-------------------+
# | Memory         | High             | Medium           | Low (sparse)      |
# +----------------+------------------+------------------+-------------------+
# | Speed          | Medium           | Fast (adaptive)  | Fastest (few pts) |
# +----------------+------------------+------------------+-------------------+
# | Best for       | Standard models  | Large/complex    | Specific points   |
# +----------------+------------------+------------------+-------------------+

# %%
# Additional Grid Types
# ~~~~~~~~~~~~~~~~~~~~~
#
# **Topography Grid:**
#
# - 2D surface representing terrain elevation
# - Used for masking voxels above ground
# - Integrates real-world DEM data
#
# **Sections Grid:**
#
# - 2D vertical cross-sections
# - Defined by start/end points and resolution
# - Useful for profile-based analysis
#
# **Centered Grid:**
#
# - Points clustered around center locations
# - Geometric spacing from center
# - Designed for geophysics (gravity, magnetics)

# %%
# Part 8: When to Use Each Grid Type
# ----------------------------------
#
# **Use Dense Grid when:**
#
# - Building standard geological models
# - Uniform importance across the domain
# - Exact uniform spacing is required
# - Visualization of complete 3D volume
#
# **Use Octree Grid when:**
#
# - Working with large models
# - Complex geometries with features at different scales
# - Memory is limited
# - Computation speed is critical
#
# **Use Custom Grid when:**
#
# - Evaluating at specific borehole locations
# - Scattered measurement points
# - Well path analysis
# - Quick queries at specific coordinates
#
# **Use Multiple Grids when:**
#
# - Need both volume visualization and specific point evaluations
# - Combining geological model with cross-sections
# - Integrating topography with subsurface model

# %%
# Conclusion
# ----------
#
# GemPy's flexible grid system allows you to choose the most appropriate
# representation for your geological modeling needs. The composition pattern
# enables combining multiple grid types in a single model for comprehensive
# analysis.

# sphinx_gallery_thumbnail_number = 2
