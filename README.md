# STL Mountains

Terrain to STL generator with a FastAPI backend and browser UI.

Package layout uses `src/` (`src/dem_to_stl`).

## Features

- DEM to STL generation from Earth Engine (MERIT DEM by default)
- Interactive web UI (map selection + 3D STL preview)
- Background generation jobs with status polling
- Persistent job history in SQLite (`output/web_jobs/jobs.sqlite3`)
- DEM cache with stats and clear operation (locked files are skipped)

## Quick Start

```powershell
.\venv\Scripts\python.exe -m pip install -e .
.\venv\Scripts\python.exe -m uvicorn dem_to_stl.web_app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Web Workflow

1. Set center, radius, shape, and model parameters.
2. Move center via marker drag or map double-click.
3. Generate STL and monitor job status.
4. Display or download results from Generated Models.

## Defaults

- Web UI defaults are tuned for detailed output (adaptive refinement enabled).
- Python dataclass defaults in [src/dem_to_stl/models.py](src/dem_to_stl/models.py) are more conservative and intended for programmatic control.
- Parameter descriptions used by the UI help tooltips are served by `GET /api/parameter-help`.

## Main Endpoints

- `POST /api/jobs`: create generation job
- `GET /api/jobs`: list job history
- `GET /api/jobs/{job_id}`: get job status/details
- `GET /api/jobs/{job_id}/stl`: download STL
- `DELETE /api/jobs/{job_id}`: delete job entry and STL file
- `GET /api/cache/stats`: cache size/count
- `DELETE /api/cache`: clear cache (best effort, skips locked files)
