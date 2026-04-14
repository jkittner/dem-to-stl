from __future__ import annotations

import math
from pathlib import Path

from pyproj import Geod

from .converter import geotiff_to_triangles
from .earth_engine import fetch_merit_dem_geotiff
from .models import BoundingBox, DEMToSTLRequest, STLResult
from .stl_writer import build_binary_stl_bytes


_WGS84 = Geod(ellps="WGS84")


def _bbox_from_center_radius(lat: float, lon: float, radius_m: float) -> BoundingBox:
    """Approximate a WGS84 bounding box from center + radius.

    Parameters:
        lat: Center latitude in degrees.
        lon: Center longitude in degrees.
        radius_m: Radius in meters.
            Higher values produce a wider geographic crop.

    Returns:
        BoundingBox: Axis-aligned bounds approximating the circular extent.
    """

    lat_delta = radius_m / 111320.0
    lon_delta = radius_m / (111320.0 * max(0.01, math.cos(math.radians(lat))))
    return BoundingBox(
        north=lat + lat_delta,
        south=lat - lat_delta,
        east=lon + lon_delta,
        west=lon - lon_delta,
    )


def _ground_extent_m(bbox: BoundingBox) -> tuple[float, float]:
    """Compute physical ground width/height for a geographic bounding box.

    Parameters:
        bbox: Geographic extent in degrees.

    Returns:
        tuple[float, float]: ``(width_m, height_m)`` measured with WGS84
        geodesic distance at bbox center lines.
    """

    center_lat = (bbox.north + bbox.south) / 2.0
    center_lon = (bbox.east + bbox.west) / 2.0

    _, _, width_m = _WGS84.inv(bbox.west, center_lat, bbox.east, center_lat)
    _, _, height_m = _WGS84.inv(center_lon, bbox.south, center_lon, bbox.north)
    return width_m, height_m


def generate_stl_bytes(
    request: DEMToSTLRequest,
    write_to_file: bool = False,
    output_path: Path | None = None,
) -> STLResult:
    """Generate terrain STL with optional file write and in-memory bytes.

    Parameters:
        request: Full generation configuration.
            Geometry and adaptive settings directly control mesh density,
            detail distribution, and model scaling.
        write_to_file: If ``True``, writes STL to disk.
            If ``False``, returns STL only in ``STLResult.stl_bytes``.
        output_path: Optional path override for file output.
            When provided, it takes precedence over ``request.output_path``.

    Returns:
        STLResult: Paths, cache info, triangle count, native DEM scale,
        and optional STL bytes.

    Raises:
        ValueError: If validation fails or file output is requested without a
            valid output path.
    """

    request.validate()

    if request.corners_bbox is not None:
        bbox = request.corners_bbox
    else:
        center = request.center_radius
        assert center is not None
        bbox = _bbox_from_center_radius(center.latitude, center.longitude, center.radius_m)

    ground_width_m, ground_height_m = _ground_extent_m(bbox)

    geotiff_path, cache_hit, native_dem_scale_m = fetch_merit_dem_geotiff(
        request=request,
        bbox=bbox,
    )

    triangles_data = geotiff_to_triangles(
        geotiff_path=geotiff_path,
        output_shape=request.output_shape,
        output_width_mm=request.output_width_mm,
        output_height_mm=request.output_height_mm,
        mesh_spacing_mm=request.mesh_spacing_mm,
        adaptive_triangulation=request.adaptive_triangulation,
        adaptive_relief_threshold_mm=request.adaptive_relief_threshold_mm,
        adaptive_max_new_points=request.adaptive_max_new_points,
        adaptive_iterations=request.adaptive_iterations,
        adaptive_min_angle_deg=request.adaptive_min_angle_deg,
        adaptive_anisotropic_refinement=request.adaptive_anisotropic_refinement,
        adaptive_anisotropic_strength=request.adaptive_anisotropic_strength,
        base_height_mm=request.base_height_mm,
        vertical_exaggeration=request.vertical_exaggeration,
        ground_width_m=ground_width_m,
        ground_height_m=ground_height_m,
        bbox_west=bbox.west,
        bbox_east=bbox.east,
        bbox_south=bbox.south,
        bbox_north=bbox.north,
    )

    stl_bytes = build_binary_stl_bytes(triangles_data)

    final_output_path = output_path or request.output_path
    if write_to_file:
        if final_output_path is None:
            raise ValueError("write_to_file=True requires output_path or request.output_path.")
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_output_path.write_bytes(stl_bytes)
    else:
        final_output_path = None

    return STLResult(
        output_path=final_output_path,
        geotiff_path=geotiff_path,
        cache_hit=cache_hit,
        triangles=len(triangles_data),
        bbox=bbox,
        dem_native_scale_m=native_dem_scale_m,
        stl_bytes=stl_bytes,
    )


def generate_stl(request: DEMToSTLRequest) -> STLResult:
    """Convenience wrapper that always writes the STL to disk.

    Parameters:
        request: Full generation configuration. ``request.output_path`` must
            be set for successful file output.

    Returns:
        STLResult: Generation metadata including on-disk ``output_path``.
    """

    return generate_stl_bytes(request=request, write_to_file=True)
