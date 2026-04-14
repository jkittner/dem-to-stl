from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from scipy.spatial import Delaunay

from .models import OutputShape
from .stl_writer import write_binary_stl


def _tri(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast triangle vertices to STL-friendly float32 arrays.

    Parameters:
        p0: First vertex ``[x_mm, y_mm, z_mm]``.
        p1: Second vertex ``[x_mm, y_mm, z_mm]``.
        p2: Third vertex ``[x_mm, y_mm, z_mm]``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Typed triangle vertices.
    """

    return p0.astype(np.float32), p1.astype(np.float32), p2.astype(np.float32)


def _shape_mask(
    output_shape: OutputShape,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    width_mm: float,
    height_mm: float,
    shape_margin_mm: float = 0.0,
) -> np.ndarray:
    """Evaluate whether XY sample locations lie inside the requested shape.

    Parameters:
        output_shape: Shape family (square/circle/hexagon).
        x_mm: X coordinates in millimeters.
        y_mm: Y coordinates in millimeters.
        width_mm: Full output width in millimeters.
        height_mm: Full output height in millimeters.
        shape_margin_mm: Inward margin applied before testing membership.
            Lower values keep samples closer to the perimeter.
            Higher values pull samples inward and reduce edge artifacts.

    Returns:
        np.ndarray: Boolean mask with ``True`` for points inside shape.
    """

    if output_shape == OutputShape.SQUARE:
        return (
            (x_mm >= shape_margin_mm)
            & (x_mm <= (width_mm - shape_margin_mm))
            & (y_mm >= shape_margin_mm)
            & (y_mm <= (height_mm - shape_margin_mm))
        )

    cx = width_mm / 2.0
    cy = height_mm / 2.0
    dx = x_mm - cx
    dy = y_mm - cy
    r = (min(width_mm, height_mm) / 2.0) - shape_margin_mm

    if output_shape == OutputShape.CIRCLE:
        return (dx * dx + dy * dy) <= (r * r)

    # Flat-top regular hexagon.
    adx = np.abs(dx)
    ady = np.abs(dy)
    return (ady <= (math.sqrt(3.0) * r / 2.0)) & ((math.sqrt(3.0) * adx + ady) <= math.sqrt(3.0) * r)


def _shape_contains_points(
    output_shape: OutputShape,
    points_xy: np.ndarray,
    width_mm: float,
    height_mm: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """Point-in-shape test for arbitrary XY point arrays.

    Parameters:
        output_shape: Shape family.
        points_xy: ``(N, 2)`` XY points in millimeters.
        width_mm: Output width in millimeters.
        height_mm: Output height in millimeters.
        eps: Boundary tolerance.
            Lower values are stricter near edges.
            Higher values include more boundary-adjacent points.

    Returns:
        np.ndarray: Boolean inclusion flags for each input point.
    """

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    if output_shape == OutputShape.SQUARE:
        return (x >= -eps) & (x <= width_mm + eps) & (y >= -eps) & (y <= height_mm + eps)

    cx = width_mm / 2.0
    cy = height_mm / 2.0
    dx = x - cx
    dy = y - cy
    r = min(width_mm, height_mm) / 2.0

    if output_shape == OutputShape.CIRCLE:
        return (dx * dx + dy * dy) <= ((r + eps) * (r + eps))

    adx = np.abs(dx)
    ady = np.abs(dy)
    return (ady <= (math.sqrt(3.0) * r / 2.0 + eps)) & ((math.sqrt(3.0) * adx + ady) <= (math.sqrt(3.0) * r + eps))


def _boundary_polygon(output_shape: OutputShape, width_mm: float, height_mm: float, spacing_mm: float) -> np.ndarray:
    """Build analytic boundary vertices for the output footprint.

    Parameters:
        output_shape: Shape family.
        width_mm: Output width in millimeters.
        height_mm: Output height in millimeters.
        spacing_mm: Baseline mesh spacing in millimeters.
            Lower values produce denser circular/edge discretization.
            Higher values reduce boundary vertex count.

    Returns:
        np.ndarray: ``(N, 2)`` boundary vertices in counterclockwise order.
    """

    cx = width_mm / 2.0
    cy = height_mm / 2.0
    shape_margin_mm = max(0.0, spacing_mm * 0.35)
    inset_width = max(spacing_mm, width_mm - 2.0 * shape_margin_mm)
    inset_height = max(spacing_mm, height_mm - 2.0 * shape_margin_mm)
    r = (min(inset_width, inset_height) / 2.0)

    if output_shape == OutputShape.SQUARE:
        return np.array(
            [
                [shape_margin_mm, shape_margin_mm],
                [shape_margin_mm + inset_width, shape_margin_mm],
                [shape_margin_mm + inset_width, shape_margin_mm + inset_height],
                [shape_margin_mm, shape_margin_mm + inset_height],
            ],
            dtype=np.float64,
        )

    if output_shape == OutputShape.HEXAGON:
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)]).astype(np.float64)

    circumference = 2.0 * math.pi * r
    n = max(96, int(math.ceil(circumference / max(0.5, spacing_mm))))
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)]).astype(np.float64)


def _sample_edges(vertices: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Densify polygon edges with approximately uniform samples.

    Parameters:
        vertices: Ordered polygon vertices ``(N, 2)``.
        spacing_mm: Target edge spacing in millimeters.
            Lower values increase edge fidelity and triangle count.
            Higher values reduce boundary sampling density.

    Returns:
        np.ndarray: Edge samples in traversal order.
    """

    points: list[np.ndarray] = []
    n = len(vertices)
    step = max(0.25, spacing_mm / 2.0)
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        count = max(1, int(math.ceil(length / step)))
        for k in range(count):
            t = k / count
            points.append(p0 + seg * t)
    return np.array(points, dtype=np.float64)


def _build_2d_points(
    output_shape: OutputShape,
    poly: np.ndarray,
    width_mm: float,
    height_mm: float,
    spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create initial 2D triangulation points for top-surface meshing.

    Parameters:
        output_shape: Shape family for masking/support strategy.
        poly: Boundary polygon points.
        width_mm: Output width in millimeters.
        height_mm: Output height in millimeters.
        spacing_mm: Baseline point spacing in millimeters.
            Lower values produce denser interior sampling and more triangles.
            Higher values reduce density and runtime.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - all unique points used for Delaunay.
            - sampled boundary points used to preserve silhouette.
    """

    # Use edge-sampled perimeters for all shapes; square/hex remain exact because their
    # boundary points are collinear along the straight edges.
    boundary = _sample_edges(poly, spacing_mm)

    support_ring = np.empty((0, 2), dtype=np.float64)
    if output_shape in (OutputShape.SQUARE, OutputShape.HEXAGON):
        center = np.mean(poly, axis=0)
        support_inset = max(0.15, spacing_mm * 0.25)
        scale = 0.0
        distances = np.linalg.norm(poly - center, axis=1)
        max_radius = max(1e-6, float(np.max(distances)))
        scale = max(0.0, 1.0 - support_inset / max_radius)
        support_poly = center + (poly - center) * scale
        support_ring = _sample_edges(support_poly, spacing_mm)

    xs = np.arange(spacing_mm, width_mm, spacing_mm, dtype=np.float64)
    ys = np.arange(spacing_mm, height_mm, spacing_mm, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        interior = np.empty((0, 2), dtype=np.float64)
    else:
        xg, yg = np.meshgrid(xs, ys)
        shape_margin_mm = max(0.0, spacing_mm * 0.35)
        mask = _shape_mask(output_shape, xg, yg, width_mm, height_mm, shape_margin_mm=shape_margin_mm)

        # Keep interior samples away from the boundary so the silhouette is driven
        # by explicit boundary vertices (avoids multi-corner artifacts).
        tol = max(1e-6, spacing_mm * 1.1)
        if output_shape == OutputShape.SQUARE:
            interior_mask = (
                (xg > (shape_margin_mm + tol))
                & (xg < (width_mm - shape_margin_mm - tol))
                & (yg > (shape_margin_mm + tol))
                & (yg < (height_mm - shape_margin_mm - tol))
            )
        else:
            cx = width_mm / 2.0
            cy = height_mm / 2.0
            dx = xg - cx
            dy = yg - cy
            r = (min(width_mm, height_mm) / 2.0) - shape_margin_mm
            if output_shape == OutputShape.CIRCLE:
                interior_mask = (dx * dx + dy * dy) < ((r - tol) * (r - tol))
            else:
                adx = np.abs(dx)
                ady = np.abs(dy)
                interior_mask = (
                    (ady <= (math.sqrt(3.0) * r / 2.0 - tol))
                    & ((math.sqrt(3.0) * adx + ady) <= (math.sqrt(3.0) * r - tol))
                )

        mask = mask & interior_mask
        interior = np.column_stack([xg[mask], yg[mask]]).astype(np.float64)

    center = np.array([[width_mm / 2.0, height_mm / 2.0]], dtype=np.float64)
    all_points = np.vstack([boundary, support_ring, interior, center]) if interior.size else np.vstack([boundary, support_ring, center])

    uniq: dict[tuple[int, int], tuple[float, float]] = {}
    for x, y in all_points:
        key = (int(round(x * 1000)), int(round(y * 1000)))
        uniq[key] = (float(x), float(y))

    pts = np.array(list(uniq.values()), dtype=np.float64)
    return pts, boundary


def _is_ccw_xy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Return ``True`` if triangle ``(a, b, c)`` is counterclockwise in XY."""

    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0.0


def _triangle_min_angles_deg(tri_pts: np.ndarray) -> np.ndarray:
    """Compute minimum interior angle for each 2D triangle.

    Parameters:
        tri_pts: Triangle vertices shaped ``(T, 3, 2)`` in XY millimeters.

    Returns:
        np.ndarray: Minimum angle (degrees) per triangle.
            Lower values indicate skinnier triangles.
    """

    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]

    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)

    eps = 1e-12
    a = np.maximum(a, eps)
    b = np.maximum(b, eps)
    c = np.maximum(c, eps)

    cos_A = np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)
    cos_B = np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0)
    cos_C = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)

    A = np.degrees(np.arccos(cos_A))
    B = np.degrees(np.arccos(cos_B))
    C = np.degrees(np.arccos(cos_C))
    return np.minimum(np.minimum(A, B), C)


def _longest_edge_midpoints(pts_xy: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Generate midpoint insertion candidates on each triangle longest edge.

    Parameters:
        pts_xy: Global XY point array.
        simplices: Triangle index array into ``pts_xy``.

    Returns:
        np.ndarray: Candidate midpoint coordinates.
    """

    tri = pts_xy[simplices]
    e01 = np.linalg.norm(tri[:, 0, :] - tri[:, 1, :], axis=1)
    e12 = np.linalg.norm(tri[:, 1, :] - tri[:, 2, :], axis=1)
    e20 = np.linalg.norm(tri[:, 2, :] - tri[:, 0, :], axis=1)
    stack = np.stack([e01, e12, e20], axis=1)
    idx = np.argmax(stack, axis=1)

    mids = np.empty((len(simplices), 2), dtype=np.float64)
    for i, which in enumerate(idx):
        a, b, c = simplices[i]
        if which == 0:
            u, v = a, b
        elif which == 1:
            u, v = b, c
        else:
            u, v = c, a
        mids[i] = (pts_xy[u] + pts_xy[v]) * 0.5
    return mids


def _ridge_aligned_points(
    tri_xy: np.ndarray,
    tri_z: np.ndarray,
    mesh_spacing_mm: float,
    anisotropic_strength: float,
) -> np.ndarray:
    """Create anisotropic candidates aligned with local ridge direction.

    Parameters:
        tri_xy: Triangle XY vertices ``(T, 3, 2)`` in millimeters.
        tri_z: Triangle elevations ``(T, 3)`` in millimeters.
        mesh_spacing_mm: Baseline spacing used to bound insertion step.
            Lower values allow tighter anisotropic displacements.
            Higher values spread displacements farther.
        anisotropic_strength: Directional displacement multiplier.
            Lower values behave closer to isotropic centroid insertion.
            Higher values bias stronger ridge-direction elongation.

    Returns:
        np.ndarray: Candidate points offset from triangle centroids along
        local contour/ridge direction.
    """

    p0 = tri_xy[:, 0, :]
    p1 = tri_xy[:, 1, :]
    p2 = tri_xy[:, 2, :]

    z0 = tri_z[:, 0]
    z1 = tri_z[:, 1]
    z2 = tri_z[:, 2]

    ax = p1[:, 0] - p0[:, 0]
    ay = p1[:, 1] - p0[:, 1]
    az = z1 - z0
    bx = p2[:, 0] - p0[:, 0]
    by = p2[:, 1] - p0[:, 1]
    bz = z2 - z0

    # Triangle plane normal n = (a x b), gradient = (-nx/nz, -ny/nz)
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx

    safe_nz = np.where(np.abs(nz) < 1e-12, 1e-12, nz)
    grad_x = -nx / safe_nz
    grad_y = -ny / safe_nz

    # Ridge/contour direction is perpendicular to gradient.
    ridge_x = -grad_y
    ridge_y = grad_x
    ridge_norm = np.sqrt(ridge_x * ridge_x + ridge_y * ridge_y)
    valid = ridge_norm > 1e-10
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    ridge_x = ridge_x[valid] / ridge_norm[valid]
    ridge_y = ridge_y[valid] / ridge_norm[valid]

    tri_xy_valid = tri_xy[valid]
    centroids = np.mean(tri_xy_valid, axis=1)

    e01 = np.linalg.norm(tri_xy_valid[:, 0, :] - tri_xy_valid[:, 1, :], axis=1)
    e12 = np.linalg.norm(tri_xy_valid[:, 1, :] - tri_xy_valid[:, 2, :], axis=1)
    e20 = np.linalg.norm(tri_xy_valid[:, 2, :] - tri_xy_valid[:, 0, :], axis=1)
    local_scale = np.maximum(np.maximum(e01, e12), e20)

    step = local_scale * (0.18 + 0.30 * anisotropic_strength)
    step = np.clip(step, mesh_spacing_mm * 0.2, mesh_spacing_mm * 1.5)

    p_plus = np.column_stack([centroids[:, 0] + ridge_x * step, centroids[:, 1] + ridge_y * step])
    p_minus = np.column_stack([centroids[:, 0] - ridge_x * step, centroids[:, 1] - ridge_y * step])
    return np.vstack([p_plus, p_minus]).astype(np.float64)


def _top_simplices_for_shape(
    pts_xy: np.ndarray,
    output_shape: OutputShape,
    output_width_mm: float,
    output_height_mm: float,
) -> np.ndarray:
    """Run Delaunay and keep only valid, non-degenerate inside triangles.

    Parameters:
        pts_xy: Input XY points for triangulation.
        output_shape: Shape filter for keeping simplices.
        output_width_mm: Shape width in millimeters.
        output_height_mm: Shape height in millimeters.

    Returns:
        np.ndarray: Filtered simplex indices into ``pts_xy``.
    """

    delaunay = Delaunay(pts_xy)
    simplices = delaunay.simplices
    n_pts = len(pts_xy)
    valid_idx = np.all((simplices >= 0) & (simplices < n_pts), axis=1)
    simplices = simplices[valid_idx]

    p0 = pts_xy[simplices[:, 0]]
    p1 = pts_xy[simplices[:, 1]]
    p2 = pts_xy[simplices[:, 2]]
    doubled_area = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    area_ok = doubled_area > 1e-8

    centroids = (p0 + p1 + p2) / 3.0
    inside = _shape_contains_points(
        output_shape=output_shape,
        points_xy=centroids,
        width_mm=output_width_mm,
        height_mm=output_height_mm,
    )

    keep = area_ok & inside
    return simplices[keep]


def _elevation_for_points(
    da: xr.DataArray,
    points_xy_mm: np.ndarray,
    output_width_mm: float,
    output_height_mm: float,
    bbox_west: float,
    bbox_east: float,
    bbox_south: float,
    bbox_north: float,
) -> np.ndarray:
    """Sample DEM elevations at arbitrary XY model coordinates.

    Parameters:
        da: DEM raster as an ``xarray.DataArray`` with lon/lat axes.
        points_xy_mm: XY points in model millimeters.
        output_width_mm: Model width in millimeters for XY->lon mapping.
        output_height_mm: Model height in millimeters for XY->lat mapping.
        bbox_west: West longitude of requested extent.
        bbox_east: East longitude of requested extent.
        bbox_south: South latitude of requested extent.
        bbox_north: North latitude of requested extent.

    Returns:
        np.ndarray: Elevation values in meters, with nodata holes patched via
        nearest-neighbor fallback (remaining NaNs replaced by minimum elevation).
    """

    x_src = da["x"].values
    y_src = da["y"].values

    if x_src[0] > x_src[-1]:
        da = da.sortby("x")
    if y_src[0] > y_src[-1]:
        da = da.sortby("y")

    x_sorted = da["x"].values.astype(np.float64)
    y_sorted = da["y"].values.astype(np.float64)

    if x_sorted.size >= 2:
        dx = float(np.median(np.abs(np.diff(x_sorted))))
    else:
        dx = 0.0
    if y_sorted.size >= 2:
        dy = float(np.median(np.abs(np.diff(y_sorted))))
    else:
        dy = 0.0

    # Sample slightly inside raster bounds to avoid boundary nodata/corner artifacts.
    safe_west = float(x_sorted.min() + dx * 0.5)
    safe_east = float(x_sorted.max() - dx * 0.5)
    safe_south = float(y_sorted.min() + dy * 0.5)
    safe_north = float(y_sorted.max() - dy * 0.5)

    if safe_east <= safe_west:
        safe_west = float(x_sorted.min())
        safe_east = float(x_sorted.max())
    if safe_north <= safe_south:
        safe_south = float(y_sorted.min())
        safe_north = float(y_sorted.max())

    lons = safe_west + (points_xy_mm[:, 0] / output_width_mm) * (safe_east - safe_west)
    lats = safe_north - (points_xy_mm[:, 1] / output_height_mm) * (safe_north - safe_south)

    sampled_linear = da.interp(
        x=xr.DataArray(lons, dims="points"),
        y=xr.DataArray(lats, dims="points"),
        method="linear",
    ).values.astype(np.float64)

    if np.any(np.isnan(sampled_linear)):
        sampled_nearest = da.interp(
            x=xr.DataArray(lons, dims="points"),
            y=xr.DataArray(lats, dims="points"),
            method="nearest",
        ).values.astype(np.float64)
        sampled_linear = np.where(np.isnan(sampled_linear), sampled_nearest, sampled_linear)

    if np.all(np.isnan(sampled_linear)):
        raise ValueError("Downloaded DEM contains only nodata.")

    min_elev = float(np.nanmin(sampled_linear))
    return np.where(np.isnan(sampled_linear), min_elev, sampled_linear)


def geotiff_to_triangles(
    geotiff_path: Path,
    output_shape: OutputShape,
    output_width_mm: float,
    output_height_mm: float,
    mesh_spacing_mm: float,
    adaptive_triangulation: bool,
    adaptive_relief_threshold_mm: float,
    adaptive_max_new_points: int,
    adaptive_iterations: int,
    adaptive_min_angle_deg: float,
    adaptive_anisotropic_refinement: bool,
    adaptive_anisotropic_strength: float,
    base_height_mm: float,
    vertical_exaggeration: float,
    ground_width_m: float,
    ground_height_m: float,
    bbox_west: float,
    bbox_east: float,
    bbox_south: float,
    bbox_north: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert a DEM GeoTIFF into watertight top/base/side STL triangles.

    Parameters:
        geotiff_path: Input DEM GeoTIFF path.
        output_shape: Target XY footprint shape.
        output_width_mm: Model width in millimeters.
            Higher values enlarge model and reduce apparent Z steepness.
        output_height_mm: Model height in millimeters.
            Higher values enlarge model and reduce apparent Z steepness.
        mesh_spacing_mm: Baseline XY spacing in millimeters.
            Lower values increase base mesh density and detail.
            Higher values decrease density and runtime.
        adaptive_triangulation: Enables iterative refinement.
            ``True`` adds points where relief/quality thresholds are violated.
        adaptive_relief_threshold_mm: Relief trigger in millimeters.
            Lower values refine more terrain areas.
            Higher values focus only on stronger terrain transitions.
        adaptive_max_new_points: Total adaptive insertion budget.
            Lower values cap detail/runtimes aggressively.
            Higher values permit denser local refinement.
        adaptive_iterations: Number of refinement rounds.
            Lower values stop earlier; higher values continue re-evaluating.
        adaptive_min_angle_deg: Minimum triangle angle target.
            Lower values tolerate skinnier triangles.
            Higher values produce better quality but more insertions.
        adaptive_anisotropic_refinement: Enables ridge-direction insertion.
            ``True`` adds contour-aligned points in high-relief triangles.
        adaptive_anisotropic_strength: Magnitude of ridge-direction offsets.
            Lower values are subtle; higher values strengthen anisotropy.
        base_height_mm: Constant base thickness in millimeters.
            Higher values make thicker printable base.
        vertical_exaggeration: Vertical scaling multiplier.
            Lower values flatten relief; higher values exaggerate relief.
        ground_width_m: Ground width in meters represented by model width.
        ground_height_m: Ground height in meters represented by model height.
        bbox_west: West longitude of sampled DEM extent.
        bbox_east: East longitude of sampled DEM extent.
        bbox_south: South latitude of sampled DEM extent.
        bbox_north: North latitude of sampled DEM extent.

    Returns:
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]: Triangle list ready
        for binary STL serialization.

    Raises:
        ValueError: If triangulation cannot produce a valid non-empty mesh.
    """

    da = rioxarray.open_rasterio(geotiff_path, masked=True).squeeze(drop=True)
    if "band" in da.dims:
        da = da.isel(band=0)

    poly = _boundary_polygon(output_shape, output_width_mm, output_height_mm, mesh_spacing_mm)
    pts_xy, _ = _build_2d_points(
        output_shape=output_shape,
        poly=poly,
        width_mm=output_width_mm,
        height_mm=output_height_mm,
        spacing_mm=mesh_spacing_mm,
    )

    if len(pts_xy) < 3:
        raise ValueError("Not enough points to build mesh.")

    elev_m = _elevation_for_points(
        da=da,
        points_xy_mm=pts_xy,
        output_width_mm=output_width_mm,
        output_height_mm=output_height_mm,
        bbox_west=bbox_west,
        bbox_east=bbox_east,
        bbox_south=bbox_south,
        bbox_north=bbox_north,
    )

    min_elev_m = float(np.min(elev_m))
    meters_per_mm_x = max(1e-9, ground_width_m / output_width_mm)
    meters_per_mm_y = max(1e-9, ground_height_m / output_height_mm)
    meters_per_mm = max(meters_per_mm_x, meters_per_mm_y)
    z_top = base_height_mm + ((elev_m - min_elev_m) / meters_per_mm) * vertical_exaggeration

    top_simplices = _top_simplices_for_shape(
        pts_xy=pts_xy,
        output_shape=output_shape,
        output_width_mm=output_width_mm,
        output_height_mm=output_height_mm,
    )

    if adaptive_triangulation and len(top_simplices) > 0:
        total_added = 0
        for _ in range(adaptive_iterations):
            if len(top_simplices) == 0:
                break

            tri_pts = pts_xy[top_simplices]
            tri_z = z_top[top_simplices]
            relief_mm = np.max(tri_z, axis=1) - np.min(tri_z, axis=1)
            min_angles = _triangle_min_angles_deg(tri_pts)

            relief_bad = relief_mm >= adaptive_relief_threshold_mm
            angle_bad = min_angles < adaptive_min_angle_deg

            if not np.any(relief_bad | angle_bad):
                break

            new_point_candidates: list[np.ndarray] = []
            if np.any(relief_bad):
                new_point_candidates.append(np.mean(tri_pts[relief_bad], axis=1))
                if adaptive_anisotropic_refinement:
                    ridge_points = _ridge_aligned_points(
                        tri_xy=tri_pts[relief_bad],
                        tri_z=tri_z[relief_bad],
                        mesh_spacing_mm=mesh_spacing_mm,
                        anisotropic_strength=adaptive_anisotropic_strength,
                    )
                    if ridge_points.size > 0:
                        new_point_candidates.append(ridge_points)
            if np.any(angle_bad):
                new_point_candidates.append(_longest_edge_midpoints(pts_xy, top_simplices[angle_bad]))

            if not new_point_candidates:
                break

            merged = np.vstack(new_point_candidates)
            inside = _shape_contains_points(
                output_shape=output_shape,
                points_xy=merged,
                width_mm=output_width_mm,
                height_mm=output_height_mm,
                eps=1e-6,
            )
            merged = merged[inside]
            if merged.size == 0:
                break

            uniq: dict[tuple[int, int], tuple[float, float]] = {}
            for x, y in merged:
                k = (int(round(float(x) * 1000)), int(round(float(y) * 1000)))
                uniq[k] = (float(x), float(y))
            new_pts = np.array(list(uniq.values()), dtype=np.float64)

            if new_pts.size == 0:
                break

            remaining_budget = adaptive_max_new_points - total_added
            if remaining_budget <= 0:
                break
            if len(new_pts) > remaining_budget:
                new_pts = new_pts[:remaining_budget]

            new_elev_m = _elevation_for_points(
                da=da,
                points_xy_mm=new_pts,
                output_width_mm=output_width_mm,
                output_height_mm=output_height_mm,
                bbox_west=bbox_west,
                bbox_east=bbox_east,
                bbox_south=bbox_south,
                bbox_north=bbox_north,
            )
            new_z = base_height_mm + ((new_elev_m - min_elev_m) / meters_per_mm) * vertical_exaggeration

            pts_xy = np.vstack([pts_xy, new_pts])
            z_top = np.concatenate([z_top, new_z])
            total_added += len(new_pts)

            top_simplices = _top_simplices_for_shape(
                pts_xy=pts_xy,
                output_shape=output_shape,
                output_width_mm=output_width_mm,
                output_height_mm=output_height_mm,
            )

    tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for simplex in top_simplices:
        a, b, c = simplex
        pa = np.array([pts_xy[a, 0], pts_xy[a, 1], z_top[a]])
        pb = np.array([pts_xy[b, 0], pts_xy[b, 1], z_top[b]])
        pc = np.array([pts_xy[c, 0], pts_xy[c, 1], z_top[c]])

        ba = np.array([pts_xy[a, 0], pts_xy[a, 1], 0.0])
        bb = np.array([pts_xy[b, 0], pts_xy[b, 1], 0.0])
        bc = np.array([pts_xy[c, 0], pts_xy[c, 1], 0.0])

        if _is_ccw_xy(pa, pb, pc):
            tris.append(_tri(pa, pb, pc))
            tris.append(_tri(ba, bc, bb))
        else:
            tris.append(_tri(pa, pc, pb))
            tris.append(_tri(ba, bb, bc))

    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in top_simplices:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_counts[e] = edge_counts.get(e, 0) + 1

    boundary_edges = [e for e, cnt in edge_counts.items() if cnt == 1]

    for u, v in boundary_edges:
        x0, y0 = pts_xy[u]
        x1, y1 = pts_xy[v]
        z0 = float(z_top[u])
        z1 = float(z_top[v])

        t0 = np.array([x0, y0, z0])
        t1 = np.array([x1, y1, z1])
        b0 = np.array([x0, y0, 0.0])
        b1 = np.array([x1, y1, 0.0])

        tris.append(_tri(b0, t1, t0))
        tris.append(_tri(b0, b1, t1))

    if not tris:
        raise ValueError("No triangles were generated. Check extent and spacing inputs.")

    return tris


def convert_geotiff_to_stl(
    geotiff_path: Path,
    output_path: Path,
    output_shape: OutputShape,
    output_width_mm: float,
    output_height_mm: float,
    mesh_spacing_mm: float,
    base_height_mm: float,
    vertical_exaggeration: float,
    ground_width_m: float,
    ground_height_m: float,
    bbox_west: float,
    bbox_east: float,
    bbox_south: float,
    bbox_north: float,
) -> int:
    """Legacy convenience helper that writes STL directly from a GeoTIFF.

    Parameters:
        geotiff_path: Input DEM GeoTIFF.
        output_path: Destination STL path.
        output_shape: Output footprint shape.
        output_width_mm: Model width in millimeters.
        output_height_mm: Model height in millimeters.
        mesh_spacing_mm: Baseline meshing spacing.
            Lower values increase detail and file size.
            Higher values reduce detail and runtime.
        base_height_mm: Base thickness in millimeters.
        vertical_exaggeration: Vertical scaling multiplier.
            Lower values flatten terrain; higher values accentuate relief.
        ground_width_m: Ground width represented by model X span.
        ground_height_m: Ground height represented by model Y span.
        bbox_west: Extent west longitude.
        bbox_east: Extent east longitude.
        bbox_south: Extent south latitude.
        bbox_north: Extent north latitude.

    Returns:
        int: Number of top-surface triangles written (legacy convention).
    """

    tris = geotiff_to_triangles(
        geotiff_path=geotiff_path,
        output_shape=output_shape,
        output_width_mm=output_width_mm,
        output_height_mm=output_height_mm,
        mesh_spacing_mm=mesh_spacing_mm,
        adaptive_triangulation=False,
        adaptive_relief_threshold_mm=1.5,
        adaptive_max_new_points=15000,
        adaptive_iterations=3,
        adaptive_min_angle_deg=28.0,
        adaptive_anisotropic_refinement=False,
        adaptive_anisotropic_strength=0.7,
        base_height_mm=base_height_mm,
        vertical_exaggeration=vertical_exaggeration,
        ground_width_m=ground_width_m,
        ground_height_m=ground_height_m,
        bbox_west=bbox_west,
        bbox_east=bbox_east,
        bbox_south=bbox_south,
        bbox_north=bbox_north,
    )
    return write_binary_stl(output_path, tris)
