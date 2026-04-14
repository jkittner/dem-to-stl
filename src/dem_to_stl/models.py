from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class OutputShape(str, Enum):
    """Supported 2D footprints for the generated STL top surface.

    Values map to analytic shapes used during triangulation:
    - ``CIRCLE``: smooth radial boundary.
    - ``HEXAGON``: regular flat-top hexagon.
    - ``SQUARE``: axis-aligned square/rectangle footprint.
    """

    CIRCLE = "circular"
    HEXAGON = "hexagonal"
    SQUARE = "square"


@dataclass(frozen=True)
class BoundingBox:
    """Geographic axis-aligned bounds in WGS84 degrees.

    Parameters:
        north: Northern latitude limit in degrees.
            Higher values move the extent north.
        south: Southern latitude limit in degrees.
            Lower values move the extent south.
        east: Eastern longitude limit in degrees.
            Higher values move the extent east.
        west: Western longitude limit in degrees.
            Lower values move the extent west.

    Notes:
        Increasing ``north - south`` increases north/south terrain coverage.
        Increasing ``east - west`` increases east/west terrain coverage.
    """

    north: float
    south: float
    east: float
    west: float

    def validate(self) -> None:
        """Validate coordinate ranges and ordering constraints.

        Raises:
            ValueError: If any coordinate is out of range or if the bounds
                do not describe a positive-area rectangle.
        """

        if not (-90.0 <= self.south <= 90.0 and -90.0 <= self.north <= 90.0):
            raise ValueError("Latitude must be in [-90, 90].")
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError("Longitude must be in [-180, 180].")
        if self.north <= self.south:
            raise ValueError("north must be greater than south.")
        if self.east <= self.west:
            raise ValueError("east must be greater than west.")


@dataclass(frozen=True)
class CenterRadius:
    """Circular geographic selection mode in WGS84.

    Parameters:
        latitude: Center latitude in degrees.
        longitude: Center longitude in degrees.
        radius_m: Radius in meters.
            Lower values produce a tighter local terrain crop.
            Higher values include a larger surrounding area.
    """

    latitude: float
    longitude: float
    radius_m: float

    def validate(self) -> None:
        """Validate center coordinates and radius.

        Raises:
            ValueError: If latitude/longitude are out of range or radius is
                not strictly positive.
        """

        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError("Center latitude must be in [-90, 90].")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError("Center longitude must be in [-180, 180].")
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be greater than 0.")


@dataclass
class DEMToSTLRequest:
    """Full request configuration for DEM-to-STL generation.

    Parameters:
        output_path: Destination STL path when writing to disk.
            ``None`` is valid when using in-memory generation.
        output_shape: Target XY footprint shape.
            Choose ``HEXAGON`` or ``CIRCLE`` for symmetric silhouettes;
            ``SQUARE`` preserves rectangular framing.
        output_width_mm: Final model width in millimeters.
            Higher values enlarge XY size and reduce apparent steepness.
        output_height_mm: Final model height in millimeters.
            Higher values enlarge XY size and reduce apparent steepness.
        vertical_exaggeration: Multiplicative Z scaling factor.
            Lower values flatten relief; higher values emphasize terrain.
        mesh_spacing_mm: Baseline sample spacing in millimeters.
            Lower values increase detail and triangle count.
            Higher values reduce detail and triangle count.
        adaptive_triangulation: Enables adaptive insertion passes.
            ``True`` increases local detail where triangles fail quality tests.
        adaptive_relief_threshold_mm: Relief trigger per triangle in mm.
            Lower values refine more broadly.
            Higher values refine only strong relief transitions.
        adaptive_max_new_points: Global cap on adaptive insertions.
            Lower values limit density and runtime.
            Higher values allow more local refinement.
        adaptive_iterations: Number of adaptive refinement passes.
            Lower values stop early; higher values push finer adaptation.
        adaptive_min_angle_deg: Minimum triangle angle target in degrees.
            Lower values allow skinnier triangles.
            Higher values push better-shaped triangles (more refinement).
        adaptive_anisotropic_refinement: Enables ridge-direction-aware points.
            ``True`` biases refinement along contour/ridge direction.
        adaptive_anisotropic_strength: Strength of anisotropic displacement.
            Lower values behave closer to isotropic refinement.
            Higher values insert points farther along ridge direction.
        base_height_mm: Constant base thickness under terrain in mm.
            Higher values produce a thicker physical base plate.
        corners_bbox: Rectangular extent mode (mutually exclusive with
            ``center_radius``).
        center_radius: Circular extent mode (mutually exclusive with
            ``corners_bbox``; requires circular output shape).
        earth_engine_project: Google Earth Engine project identifier used
            for authentication and quota context.
        dem_dataset_id: Earth Engine DEM image collection/image id.
            Changing this changes source raster and resolution behavior.
        cache_dir: Local folder for downloaded GeoTIFF cache and metadata.
    """

    output_path: Optional[Path] = None
    output_shape: OutputShape = OutputShape.SQUARE
    output_width_mm: float = 100.0
    output_height_mm: float = 100.0
    vertical_exaggeration: float = 2.0
    mesh_spacing_mm: float = 1.0
    adaptive_triangulation: bool = False
    adaptive_relief_threshold_mm: float = 1.5
    adaptive_max_new_points: int = 15000
    adaptive_iterations: int = 3
    adaptive_min_angle_deg: float = 28.0
    adaptive_anisotropic_refinement: bool = False
    adaptive_anisotropic_strength: float = 0.7
    base_height_mm: float = 10.0
    corners_bbox: Optional[BoundingBox] = None
    center_radius: Optional[CenterRadius] = None
    earth_engine_project: str = "focus-nucleus-413610"
    dem_dataset_id: str = "MERIT/DEM/v1_0_3"
    cache_dir: Path = Path(".cache/dem_to_stl")

    def validate(self) -> None:
        """Validate request ranges and mutually exclusive extent modes.

        Raises:
            ValueError: If numeric constraints are violated or extent mode is
                ambiguous/invalid.
        """

        if self.output_width_mm <= 0 or self.output_height_mm <= 0:
            raise ValueError("output_width_mm and output_height_mm must be greater than 0.")
        if self.vertical_exaggeration <= 0:
            raise ValueError("vertical_exaggeration must be greater than 0.")
        if self.mesh_spacing_mm <= 0:
            raise ValueError("mesh_spacing_mm must be greater than 0.")
        if self.adaptive_relief_threshold_mm <= 0:
            raise ValueError("adaptive_relief_threshold_mm must be greater than 0.")
        if self.adaptive_max_new_points <= 0:
            raise ValueError("adaptive_max_new_points must be greater than 0.")
        if self.adaptive_iterations <= 0:
            raise ValueError("adaptive_iterations must be greater than 0.")
        if not (5.0 <= self.adaptive_min_angle_deg <= 40.0):
            raise ValueError("adaptive_min_angle_deg must be between 5 and 40 degrees.")
        if not (0.0 <= self.adaptive_anisotropic_strength <= 2.0):
            raise ValueError("adaptive_anisotropic_strength must be in [0, 2].")
        if self.base_height_mm < 0:
            raise ValueError("base_height_mm must be greater than or equal to 0.")

        has_bbox = self.corners_bbox is not None
        has_center = self.center_radius is not None

        if has_bbox == has_center:
            raise ValueError("Provide exactly one extent mode: corners_bbox or center_radius.")

        if has_bbox:
            self.corners_bbox.validate()

        if has_center:
            self.center_radius.validate()
            if self.output_shape != OutputShape.CIRCLE:
                raise ValueError("center_radius mode is only supported for circular output.")


@dataclass(frozen=True)
class STLResult:
    """Result payload from an STL generation call.

    Attributes:
        output_path: Path of written STL, or ``None`` for in-memory mode.
        geotiff_path: Cached GeoTIFF source used for meshing.
        cache_hit: ``True`` when source GeoTIFF came from local cache.
        triangles: Final triangle count written to STL.
        bbox: Effective geographic bounds used for DEM retrieval.
        dem_native_scale_m: Native nominal DEM pixel size in meters.
        stl_bytes: In-memory STL bytes when requested by API.
    """

    output_path: Optional[Path]
    geotiff_path: Path
    cache_hit: bool
    triangles: int
    bbox: BoundingBox
    dem_native_scale_m: float
    stl_bytes: Optional[bytes] = None
