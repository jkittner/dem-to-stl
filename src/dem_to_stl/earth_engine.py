from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import ee
import requests

from .cache import geotiff_cache_key
from .cache import geotiff_paths
from .cache import metadata_for_bbox
from .cache import write_metadata
from .models import BoundingBox
from .models import DEMToSTLRequest


def _elevation_band_candidates(dataset_id: str) -> list[str]:
    """Return likely elevation band names for a supported dataset."""

    if dataset_id == 'JAXA/ALOS/AW3D30/V4_1':
        return ['DSM', 'DEM', 'dem']
    if dataset_id == 'MERIT/DEM/v1_0_3':
        return ['dem', 'DEM']
    return ['DEM', 'dem', 'DSM']


def _select_elevation_band(image: Any, dataset_id: str) -> Any:
    """Select the first matching elevation band from an EE image-like object."""

    band_names = list(image.bandNames().getInfo())
    for candidate in _elevation_band_candidates(dataset_id):
        if candidate in band_names:
            return image.select(candidate)

    if len(band_names) == 1:
        return image.select(band_names[0])

    raise ValueError(
        f'Could not determine elevation band for {dataset_id}; '
        f'available bands: {band_names}.',
    )


def _resolve_dem_image(dataset_id: str) -> tuple[Any, str]:
    """Resolve dataset id to a single Earth Engine image.

    ImageCollections are mosaiced first, then forced back onto the projection
    of one source tile so the resulting raster is one continuous image.
    """

    try:
        image = _select_elevation_band(ee.Image(dataset_id), dataset_id)
        _ = image.projection().nominalScale().getInfo()
        return image, 'IMAGE'
    except Exception:
        pass

    try:
        collection = ee.ImageCollection(dataset_id)
        first = _select_elevation_band(ee.Image(collection.first()), dataset_id)
        selected_band = first.bandNames().getInfo()[0]
        collection = collection.select(selected_band)
        native_proj = first.projection()
        mosaic = collection.mosaic().setDefaultProjection(native_proj)
        return mosaic, 'IMAGE_COLLECTION'
    except Exception as exc:
        raise ValueError(
            f"Unsupported or inaccessible Earth Engine dataset id: {dataset_id}. "
            'Provide an ee.Image or ee.ImageCollection id.',
        ) from exc


def _get_dem_native_scale(dataset_id: str) -> float:
    """Get native scale in meters for known DEM datasets."""
    scales = {
        'MERIT/DEM/v1_0_3': 90.0,
        'COPERNICUS/DEM/GLO30': 30.0,
        'JAXA/ALOS/AW3D30/V4_1': 30.0,
    }
    return scales.get(dataset_id, 30.0)


def fetch_dem_geotiff(
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
    ee.Initialize(project=request.earth_engine_project)
    print(f"Resolving DEM dataset: {request.dem_dataset_id}")
    image, asset_type = _resolve_dem_image(request.dem_dataset_id)
    print(f"Asset type: {asset_type}")
    native_scale_m = _get_dem_native_scale(request.dem_dataset_id)
    print(f"Native scale: {native_scale_m} m")

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
        proj='EPSG:4326',
        geodesic=False,
    )
    clipped = image.clip(region)

    # Export on the image's native grid to avoid reprojection artifacts
    # introduced by scale+EPSG:4326 resampling.
    proj_info = image.projection().getInfo()
    native_crs = proj_info.get('crs')
    native_transform = proj_info.get('transform')

    print('Getting download URL for clipped image...')
    download_params = {
        'region': region,
        'format': 'GEO_TIFF',
    }
    if native_crs and native_transform:
        download_params['crs'] = native_crs
        download_params['crs_transform'] = native_transform
    else:
        # Fallback when projection metadata is not available.
        download_params['scale'] = native_scale_m
        download_params['crs'] = 'EPSG:4326'

    url = clipped.getDownloadURL(download_params)
    print(f"Download URL obtained: {url}...")

    response = requests.get(url, timeout=180)
    response.raise_for_status()
    print(f"Downloaded {len(response.content)} bytes")
    tif_path.write_bytes(response.content)

    write_metadata(
        meta_path,
        {
            'cache_key': key,
            'dataset_id': request.dem_dataset_id,
            'dataset_asset_type': asset_type,
            'earth_engine_project': request.earth_engine_project,
            'bbox': metadata_for_bbox(bbox),
            'dem_scale_m': native_scale_m,
            'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'geotiff_path': str(tif_path),
            'content_bytes': len(response.content),
        },
    )

    return tif_path, False, native_scale_m
