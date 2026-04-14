from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .api import generate_stl
from .models import BoundingBox, DEMToSTLRequest, OutputShape


@dataclass
class JobRecord:
    """In-memory state for one asynchronous STL generation request."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    request: dict[str, Any]
    output_path: Path | None = None
    geotiff_path: str | None = None
    triangles: int | None = None
    cache_hit: bool | None = None
    dem_native_scale_m: float | None = None
    duration_seconds: float | None = None
    stl_size_bytes: int | None = None
    error: str | None = None


class GenerateJobRequest(BaseModel):
    """User-facing payload for terrain STL generation.

    All dimensions are in millimeters unless otherwise noted.
    """

    center_lat: float = Field(..., ge=-90.0, le=90.0)
    center_lon: float = Field(..., ge=-180.0, le=180.0)
    radius_m: float = Field(8000.0, gt=100.0)

    output_shape: OutputShape = OutputShape.HEXAGON
    output_width_mm: float = Field(100.0, gt=10.0)
    output_height_mm: float = Field(100.0, gt=10.0)
    vertical_exaggeration: float = Field(2.0, gt=0.0)
    mesh_spacing_mm: float = Field(0.4, gt=0.05)
    base_height_mm: float = Field(2.0, ge=0.0)

    adaptive_triangulation: bool = True
    adaptive_relief_threshold_mm: float = Field(0.2, gt=0.0)
    adaptive_max_new_points: int = Field(40000, gt=0)
    adaptive_iterations: int = Field(8, gt=0)
    adaptive_min_angle_deg: float = Field(26.0, ge=5.0, le=40.0)
    adaptive_anisotropic_refinement: bool = True
    adaptive_anisotropic_strength: float = Field(0.9, ge=0.0, le=2.0)

    earth_engine_project: str = "focus-nucleus-413610"
    dem_dataset_id: str = "MERIT/DEM/v1_0_3"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bbox_from_center_radius(lat: float, lon: float, radius_m: float) -> BoundingBox:
    """Convert a center + radius to an axis-aligned bbox in WGS84 degrees."""

    lat_delta = radius_m / 111320.0
    lon_delta = radius_m / (111320.0 * max(0.01, math.cos(math.radians(lat))))
    return BoundingBox(
        north=lat + lat_delta,
        south=lat - lat_delta,
        east=lon + lon_delta,
        west=lon - lon_delta,
    )


def _parameter_help() -> dict[str, str]:
    return {
        "center_lat": "Center latitude in degrees. Lower values move south; higher values move north.",
        "center_lon": "Center longitude in degrees. Lower values move west; higher values move east.",
        "radius_m": "Selection radius in meters. Lower values select a smaller area (usually faster). Higher values select a larger area (usually slower).",
        "output_shape": "XY footprint shape for the model: circular, hexagonal, or square.",
        "output_width_mm": "Final model width in millimeters. Higher values enlarge X size and make slopes appear less steep.",
        "output_height_mm": "Final model height in millimeters. Higher values enlarge Y size and make slopes appear less steep.",
        "vertical_exaggeration": "Multiplicative Z scaling factor. Lower values flatten relief; higher values emphasize relief.",
        "mesh_spacing_mm": "Baseline sample spacing in millimeters. Lower values increase detail and triangle count; higher values reduce detail and triangle count.",
        "base_height_mm": "Constant base thickness in millimeters. Higher values create a thicker physical base.",
        "adaptive_triangulation": "Enable adaptive refinement passes that add points where relief or triangle-angle quality requires it.",
        "adaptive_relief_threshold_mm": "Per-triangle relief trigger in millimeters. Lower values refine more broadly; higher values refine only stronger terrain transitions.",
        "adaptive_iterations": "Number of adaptive refinement passes. Lower values stop earlier; higher values continue refinement longer.",
        "adaptive_max_new_points": "Global cap on adaptive point insertions. Lower values limit density/runtime; higher values allow denser local refinement.",
        "adaptive_min_angle_deg": "Minimum triangle-angle target in degrees. Lower values allow skinnier triangles; higher values improve triangle shape (often adding more points).",
        "adaptive_anisotropic_refinement": "Enable ridge-direction-aware refinement so inserted points can align along terrain structure.",
        "adaptive_anisotropic_strength": "Strength of directional refinement. Lower values are subtle; higher values align refinement more strongly with ridge direction.",
        "earth_engine_project": "Google Earth Engine project id used for authentication and quota context.",
        "dem_dataset_id": "Earth Engine DEM dataset id. Changing this changes source raster characteristics and resolution behavior.",
    }


def _dir_size_bytes(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    files = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            files += 1
            total += int(file_path.stat().st_size)
    return total, files


def _serialize_job(record: JobRecord) -> dict[str, Any]:
    stl_size_bytes = record.stl_size_bytes
    output_path = record.output_path
    if output_path is None:
        inferred = (Path.cwd() / "output" / "web_jobs" / f"{record.job_id}.stl").resolve()
        if inferred.exists():
            output_path = inferred
            record.output_path = inferred

    if stl_size_bytes is None and output_path is not None and output_path.exists():
        stl_size_bytes = int(output_path.stat().st_size)
        record.stl_size_bytes = stl_size_bytes

    response: dict[str, Any] = {
        "job_id": record.job_id,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "request": record.request,
        "error": record.error,
        "triangles": record.triangles,
        "cache_hit": record.cache_hit,
        "dem_native_scale_m": record.dem_native_scale_m,
        "duration_seconds": record.duration_seconds,
        "stl_size_bytes": stl_size_bytes,
        "geotiff_path": record.geotiff_path,
    }
    if record.status == "done" and output_path is not None and output_path.exists():
        response["stl_url"] = f"/api/jobs/{record.job_id}/stl"
    return response


def create_app() -> FastAPI:
    """Create the FastAPI app serving both API and static frontend."""

    app = FastAPI(title="DEM to STL Web", version="0.1.0")

    static_dir = Path(__file__).resolve().parent / "web_static"
    output_dir = Path.cwd() / "output" / "web_jobs"
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "jobs.sqlite3"
    cache_dir = DEMToSTLRequest().cache_dir

    jobs: dict[str, JobRecord] = {}

    def init_db() -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    output_path TEXT,
                    geotiff_path TEXT,
                    triangles INTEGER,
                    cache_hit INTEGER,
                    dem_native_scale_m REAL,
                    duration_seconds REAL,
                    stl_size_bytes INTEGER,
                    error TEXT
                )
                """
            )

    def upsert_job(record: JobRecord) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, created_at, updated_at, request_json,
                    output_path, geotiff_path, triangles, cache_hit,
                    dem_native_scale_m, duration_seconds, stl_size_bytes, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    request_json=excluded.request_json,
                    output_path=excluded.output_path,
                    geotiff_path=excluded.geotiff_path,
                    triangles=excluded.triangles,
                    cache_hit=excluded.cache_hit,
                    dem_native_scale_m=excluded.dem_native_scale_m,
                    duration_seconds=excluded.duration_seconds,
                    stl_size_bytes=excluded.stl_size_bytes,
                    error=excluded.error
                """,
                (
                    record.job_id,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.request, separators=(",", ":")),
                    str(record.output_path) if record.output_path is not None else None,
                    record.geotiff_path,
                    record.triangles,
                    1 if record.cache_hit else (0 if record.cache_hit is not None else None),
                    record.dem_native_scale_m,
                    record.duration_seconds,
                    record.stl_size_bytes,
                    record.error,
                ),
            )

    def delete_job_from_db(job_id: str) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def load_jobs_from_db() -> None:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    job_id, status, created_at, updated_at, request_json,
                    output_path, geotiff_path, triangles, cache_hit,
                    dem_native_scale_m, duration_seconds, stl_size_bytes, error
                FROM jobs
                """
            ).fetchall()

        for row in rows:
            request_dict = json.loads(row["request_json"]) if row["request_json"] else {}
            output_path_val = row["output_path"]
            jobs[row["job_id"]] = JobRecord(
                job_id=row["job_id"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                request=request_dict,
                output_path=Path(output_path_val) if output_path_val else None,
                geotiff_path=row["geotiff_path"],
                triangles=row["triangles"],
                cache_hit=(None if row["cache_hit"] is None else bool(row["cache_hit"])),
                dem_native_scale_m=row["dem_native_scale_m"],
                duration_seconds=row["duration_seconds"],
                stl_size_bytes=row["stl_size_bytes"],
                error=row["error"],
            )

    init_db()
    load_jobs_from_db()

    async def run_job(job_id: str, payload: GenerateJobRequest) -> None:
        record = jobs[job_id]
        record.status = "running"
        record.updated_at = _utc_now_iso()
        upsert_job(record)
        started = time.perf_counter()

        try:
            bbox = _bbox_from_center_radius(payload.center_lat, payload.center_lon, payload.radius_m)
            output_path = (output_dir / f"{job_id}.stl").resolve()

            request = DEMToSTLRequest(
                output_path=output_path,
                corners_bbox=bbox,
                output_shape=payload.output_shape,
                output_width_mm=payload.output_width_mm,
                output_height_mm=payload.output_height_mm,
                vertical_exaggeration=payload.vertical_exaggeration,
                mesh_spacing_mm=payload.mesh_spacing_mm,
                adaptive_triangulation=payload.adaptive_triangulation,
                adaptive_relief_threshold_mm=payload.adaptive_relief_threshold_mm,
                adaptive_max_new_points=payload.adaptive_max_new_points,
                adaptive_iterations=payload.adaptive_iterations,
                adaptive_min_angle_deg=payload.adaptive_min_angle_deg,
                adaptive_anisotropic_refinement=payload.adaptive_anisotropic_refinement,
                adaptive_anisotropic_strength=payload.adaptive_anisotropic_strength,
                base_height_mm=payload.base_height_mm,
                earth_engine_project=payload.earth_engine_project,
                dem_dataset_id=payload.dem_dataset_id,
            )

            result = await asyncio.to_thread(generate_stl, request)
            record.status = "done"
            record.output_path = result.output_path
            record.geotiff_path = str(result.geotiff_path)
            record.triangles = result.triangles
            record.cache_hit = result.cache_hit
            record.dem_native_scale_m = result.dem_native_scale_m
            if result.output_path is not None and result.output_path.exists():
                record.stl_size_bytes = int(result.output_path.stat().st_size)
            record.duration_seconds = time.perf_counter() - started
            record.updated_at = _utc_now_iso()
            upsert_job(record)
        except Exception as exc:  # noqa: BLE001
            record.status = "error"
            record.error = str(exc)
            record.duration_seconds = time.perf_counter() - started
            record.updated_at = _utc_now_iso()
            upsert_job(record)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/api/jobs")
    async def create_job(payload: GenerateJobRequest) -> dict[str, str]:
        job_id = uuid4().hex
        now = _utc_now_iso()
        tracked_output = (output_dir / f"{job_id}.stl").resolve()
        jobs[job_id] = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request=payload.model_dump(),
            output_path=tracked_output,
        )
        upsert_job(jobs[job_id])
        asyncio.create_task(run_job(job_id, payload))
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        ordered = sorted(jobs.values(), key=lambda rec: rec.created_at, reverse=True)
        return [_serialize_job(record) for record in ordered]

    @app.get("/api/parameter-help")
    async def parameter_help() -> dict[str, str]:
        return _parameter_help()

    @app.get("/api/cache/stats")
    async def cache_stats() -> dict[str, int | str]:
        size_bytes, file_count = _dir_size_bytes(cache_dir)
        return {
            "cache_dir": str(cache_dir),
            "size_bytes": size_bytes,
            "file_count": file_count,
        }

    @app.delete("/api/cache")
    async def clear_cache() -> dict[str, int | str]:
        deleted_files = 0
        deleted_bytes = 0
        skipped_files = 0
        if cache_dir.exists():
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file():
                    file_size = int(file_path.stat().st_size)
                    try:
                        file_path.unlink(missing_ok=True)
                        deleted_bytes += file_size
                        deleted_files += 1
                    except PermissionError:
                        # Skip files still opened by another process.
                        skipped_files += 1
                    except OSError:
                        skipped_files += 1
            # Remove empty folders bottom-up.
            for folder in sorted([p for p in cache_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
                try:
                    folder.rmdir()
                except OSError:
                    pass
        cache_dir.mkdir(parents=True, exist_ok=True)
        return {
            "cache_dir": str(cache_dir),
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
            "skipped_files": skipped_files,
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return _serialize_job(record)

    @app.get("/api/jobs/{job_id}/stl")
    async def get_job_stl(job_id: str) -> FileResponse:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.status != "done" or record.output_path is None:
            raise HTTPException(status_code=409, detail="STL not ready")
        if not record.output_path.exists():
            raise HTTPException(status_code=404, detail="STL file missing")
        return FileResponse(
            record.output_path,
            media_type="model/stl",
            filename=record.output_path.name,
        )

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, str | bool]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")

        file_deleted = False
        if record.output_path is not None and record.output_path.exists():
            try:
                record.output_path.unlink(missing_ok=True)
                file_deleted = True
            except PermissionError:
                return {
                    "job_id": job_id,
                    "deleted": False,
                    "file_deleted": False,
                    "reason": "STL file is locked by another process.",
                }
            except OSError:
                return {
                    "job_id": job_id,
                    "deleted": False,
                    "file_deleted": False,
                    "reason": "STL file could not be deleted.",
                }

        jobs.pop(job_id, None)
        delete_job_from_db(job_id)
        return {
            "job_id": job_id,
            "deleted": True,
            "file_deleted": file_deleted,
            "reason": "Job and associated STL deleted.",
        }

    @app.delete("/api/jobs/{job_id}/stl")
    async def delete_job_stl(job_id: str) -> dict[str, str | bool]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if record.output_path is None:
            return {"job_id": job_id, "deleted": False, "reason": "No output file tracked for this job."}

        existed = record.output_path.exists()
        if existed:
            try:
                record.output_path.unlink(missing_ok=True)
            except PermissionError:
                return {"job_id": job_id, "deleted": False, "reason": "STL file is locked by another process."}
            except OSError:
                return {"job_id": job_id, "deleted": False, "reason": "STL file could not be deleted."}

        record.stl_size_bytes = None
        record.updated_at = _utc_now_iso()
        upsert_job(record)

        return {"job_id": job_id, "deleted": existed, "reason": "Deleted" if existed else "File already missing"}

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()


def run_web() -> None:
    """Run the FastAPI web server for DEM-to-STL UI."""

    import uvicorn

    uvicorn.run("dem_to_stl.web_app:app", host="127.0.0.1", port=8000, reload=False)
