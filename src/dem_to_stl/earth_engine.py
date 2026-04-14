from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests

from .cache import geotiff_cache_key, geotiff_paths, metadata_for_bbox, write_metadata
from .models import BoundingBox, DEMToSTLRequest


def fetch_merit_dem_geotiff(
    request: DEMToSTLRequest,
    bbox: BoundingBox,
 ) -> tuple[Path, bool, float]:
    """Fetch a DEM GeoTIFF from Earth Engine at native dataset resolution.

    Parameters:
        request: Generation request providing Earth Engine settings.
            ``earth_engine_project`` selects the EE project context.
            ``dem_dataset_id`` selects DEM source (affects resolution/terrain).
            ``cache_dir`` controls cache location.
        bbox: Geographic extent to download.
            Larger extents increase download size and processing time.

    Returns:
        tuple[Path, bool, float]:
            - GeoTIFF path in local cache.
            - cache-hit flag.
            - native nominal DEM scale in meters.

    Raises:
        requests.HTTPError: If Earth Engine download URL fetch fails.
    """

    import ee

    ee.Initialize(project=request.earth_engine_project)
    image = ee.Image(request.dem_dataset_id)
    native_scale_m = float(image.projection().nominalScale().getInfo())

    key = geotiff_cache_key(
        bbox=bbox,
        dem_scale_m=native_scale_m,
        dataset_id=request.dem_dataset_id,
    )
    tif_path, meta_path = geotiff_paths(request.cache_dir, key)

    if tif_path.exists() and meta_path.exists():
        return tif_path, True, native_scale_m

    tif_path.parent.mkdir(parents=True, exist_ok=True)

    region = ee.Geometry.Rectangle(
        [bbox.west, bbox.south, bbox.east, bbox.north],
        proj="EPSG:4326",
        geodesic=False,
    )
    clipped = image.clip(region)

    url = clipped.getDownloadURL(
        {
            "region": region,
            "scale": native_scale_m,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
    )

    response = requests.get(url, timeout=180)
    response.raise_for_status()
    tif_path.write_bytes(response.content)

    write_metadata(
        meta_path,
        {
            "cache_key": key,
            "dataset_id": request.dem_dataset_id,
            "earth_engine_project": request.earth_engine_project,
            "bbox": metadata_for_bbox(bbox),
            "dem_scale_m": native_scale_m,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "geotiff_path": str(tif_path),
            "content_bytes": len(response.content),
        },
    )

    return tif_path, False, native_scale_m
