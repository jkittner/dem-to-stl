import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BoundingBox


def _rounded(v: float) -> float:
    """Round floating values to stable precision for deterministic cache keys."""

    return round(v, 6)


def geotiff_cache_key(bbox: BoundingBox, dem_scale_m: float, dataset_id: str) -> str:
    """Build deterministic cache key for DEM download artifacts.

    Parameters:
        bbox: Geographic extent used for DEM request.
        dem_scale_m: Download scale in meters.
            Lower values represent finer native pixels.
            Higher values represent coarser sampling.
        dataset_id: DEM source identifier.
            Changing this always produces a new key.

    Returns:
        str: SHA-256 hex digest uniquely representing fetch inputs.
    """

    payload = {
        'north': _rounded(bbox.north),
        'south': _rounded(bbox.south),
        'east': _rounded(bbox.east),
        'west': _rounded(bbox.west),
        'dem_scale_m': round(dem_scale_m, 6),
        'dataset_id': dataset_id,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True,
        ).encode('utf-8'),
    ).hexdigest()
    return digest


def geotiff_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    """Resolve sharded cache paths for GeoTIFF and metadata.

    Parameters:
        cache_dir: Cache root directory.
        key: Cache key digest.

    Returns:
        tuple[Path, Path]: ``(tif_path, metadata_json_path)``.
    """

    shard = key[:2]
    root = cache_dir / shard
    return root / f"{key}.tif", root / f"{key}.json"


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Persist cache metadata JSON with indentation for inspection.

    Parameters:
        path: Destination JSON path.
        metadata: Serializable metadata payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def metadata_for_bbox(bbox: BoundingBox) -> dict[str, float]:
    """Convert bounding box dataclass to plain metadata dictionary."""

    return asdict(bbox)
