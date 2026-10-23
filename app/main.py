from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT_DIR, save_settings, settings
from .database import connect, execute, fetch_all, fetch_one, get_video, init_db, utc_now
from .media import detect_watermark_region, get_thumbnail, source_path
from .scanner import scan_source
from .worker import worker


STATIC_DIR = ROOT_DIR / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    init_db()
    execute(
        "UPDATE jobs SET status='queued', updated_at=? WHERE status='processing'",
        (utc_now(),),
    )
    yield
    if worker.running:
        await worker.pause()


app = FastAPI(title="Watermark Batch Console", version="0.1.0", lifespan=lifespan)


class SettingsUpdate(BaseModel):
    source_dir: str | None = None
    output_dir: str | None = None
    provider: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    concurrency: int | None = Field(None, ge=1, le=8)
    max_retries: int | None = Field(None, ge=0, le=10)
    poll_interval_seconds: float | None = Field(None, ge=0.5, le=60)
    request_timeout_seconds: float | None = Field(None, ge=10, le=3600)
    mask_x: float | None = Field(None, ge=0, le=1)
    mask_y: float | None = Field(None, ge=0, le=1)
    mask_width: float | None = Field(None, gt=0, le=1)
    mask_height: float | None = Field(None, gt=0, le=1)
    mask_feather: int | None = Field(None, ge=0, le=100)
    local_mask_grow: int | None = Field(None, ge=0, le=6)
    local_patch_size: int | None = Field(None, ge=3, le=11)


class RegionUpdate(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class QueueRequest(BaseModel):
    video_ids: list[int] = Field(default_factory=list)
    category: str | None = None
    status: str | None = "pending"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "worker_running": worker.running, "provider": settings.provider}


@app.get("/api/settings")
def read_settings() -> dict[str, Any]:
    return settings.public_dict()


@app.put("/api/settings")
async def update_settings(payload: SettingsUpdate) -> dict[str, Any]:
    restart_worker = worker.running
    if restart_worker:
        await worker.pause()
    values = payload.model_dump(exclude_none=True)
    api_key = values.pop("api_key", None)
    if api_key is not None:
        os.environ["WATERMARK_API_KEY"] = api_key
    for key, value in values.items():
        setattr(settings, key, value)
    save_settings(settings)
    if restart_worker:
        await worker.start()
    return settings.public_dict()


@app.post("/api/scan")
async def scan() -> dict[str, int]:
    try:
        return await asyncio.to_thread(scan_source, settings.source_dir)
    except (FileNotFoundError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    totals = fetch_one(
        "SELECT COUNT(*) AS total, COALESCE(SUM(size_bytes), 0) AS bytes FROM videos"
    ) or {"total": 0, "bytes": 0}
    statuses = {
        row["status"]: row["count"]
        for row in fetch_all(
            """
            SELECT COALESCE(j.status, 'pending') AS status, COUNT(*) AS count
            FROM videos v LEFT JOIN jobs j ON j.video_id=v.id
            GROUP BY COALESCE(j.status, 'pending')
            """
        )
    }
    categories = fetch_all(
        "SELECT category, COUNT(*) AS count FROM videos GROUP BY category ORDER BY count DESC"
    )
    return {
        "total": totals["total"], "bytes": totals["bytes"],
        "statuses": statuses, "categories": categories,
        "worker_running": worker.running,
    }


@app.get("/api/videos")
def list_videos(
    search: str = "",
    category: str = "",
    status: str = "",
    limit: int = Query(36, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    conditions: list[str] = []
    parameters: list[Any] = []
    if search:
        conditions.append("(v.filename LIKE ? OR v.title LIKE ? OR v.material_id LIKE ?)")
        token = f"%{search}%"
        parameters.extend([token, token, token])
    if category:
        conditions.append("v.category = ?")
        parameters.append(category)
    if status:
        conditions.append("COALESCE(j.status, 'pending') = ?")
        parameters.append(status)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    total = fetch_one(
        f"SELECT COUNT(*) AS count FROM videos v LEFT JOIN jobs j ON j.video_id=v.id {where}",
        tuple(parameters),
    )["count"]
    rows = fetch_all(
        f"""
        SELECT v.*, COALESCE(j.status, 'pending') AS status,
               COALESCE(j.progress, 0) AS progress, j.error, j.output_path
        FROM videos v LEFT JOIN jobs j ON j.video_id=v.id
        {where} ORDER BY v.id LIMIT ? OFFSET ?
        """,
        tuple(parameters + [limit, offset]),
    )
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@app.get("/api/videos/{video_id}")
def video_detail(video_id: int) -> dict[str, Any]:
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.get("/api/videos/{video_id}/thumbnail")
async def thumbnail(video_id: int) -> FileResponse:
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    path = source_path(video["relative_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    try:
        target = await asyncio.to_thread(get_thumbnail, video_id, path)
        return FileResponse(target, media_type="image/jpeg")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/videos/{video_id}/media")
def video_media(video_id: int) -> FileResponse:
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    path = source_path(video["relative_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    return FileResponse(path, media_type="video/mp4")


@app.put("/api/videos/{video_id}/region")
def update_region(video_id: int, payload: RegionUpdate) -> dict[str, Any]:
    if payload.x + payload.width > 1 or payload.y + payload.height > 1:
        raise HTTPException(status_code=422, detail="Mask region exceeds frame bounds")
    if not get_video(video_id):
        raise HTTPException(status_code=404, detail="Video not found")
    execute(
        """
        UPDATE videos SET mask_x=?, mask_y=?, mask_width=?, mask_height=?, updated_at=?
        WHERE id=?
        """,
        (payload.x, payload.y, payload.width, payload.height, utc_now(), video_id),
    )
    execute(
        """
        UPDATE videos SET mask_score=1, mask_method='manual', mask_review_required=0
        WHERE id=?
        """,
        (video_id,),
    )
    return get_video(video_id)


@app.post("/api/videos/{video_id}/detect-region")
async def detect_region(video_id: int) -> dict[str, Any]:
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    path = source_path(video["relative_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    region = await asyncio.to_thread(detect_watermark_region, video_id, path)
    execute(
        """
        UPDATE videos SET mask_x=?, mask_y=?, mask_width=?, mask_height=?,
                          mask_score=?, mask_method=?, mask_review_required=?, updated_at=?
        WHERE id=?
        """,
        (
            region["x"], region["y"], region["width"], region["height"],
            region["score"], region["method"], int(region["review_required"]),
            utc_now(), video_id,
        ),
    )
    result = get_video(video_id)
    result["detection_score"] = region["score"]
    return result


@app.post("/api/jobs/queue")
def queue_jobs(payload: QueueRequest) -> dict[str, int]:
    if payload.video_ids:
        placeholders = ",".join("?" for _ in payload.video_ids)
        rows = fetch_all(f"SELECT id FROM videos WHERE id IN ({placeholders})", tuple(payload.video_ids))
    else:
        conditions = []
        parameters: list[Any] = []
        if payload.category:
            conditions.append("v.category=?")
            parameters.append(payload.category)
        if payload.status:
            conditions.append("COALESCE(j.status, 'pending')=?")
            parameters.append(payload.status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = fetch_all(
            f"SELECT v.id FROM videos v LEFT JOIN jobs j ON j.video_id=v.id {where}",
            tuple(parameters),
        )
    now = utc_now()
    with connect() as connection:
        for row in rows:
            connection.execute(
                """
                INSERT INTO jobs(video_id, status, progress, attempt, created_at, updated_at)
                VALUES (?, 'queued', 0, 0, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status='queued', progress=0, attempt=0, error=NULL,
                    output_path=NULL, updated_at=excluded.updated_at
                """,
                (row["id"], now, now),
            )
    return {"queued": len(rows)}


@app.post("/api/jobs/start")
async def start_jobs() -> dict[str, bool]:
    await worker.start()
    return {"running": worker.running}


@app.post("/api/jobs/pause")
async def pause_jobs() -> dict[str, bool]:
    await worker.pause()
    return {"running": worker.running}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
