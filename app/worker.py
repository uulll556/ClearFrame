from __future__ import annotations

import asyncio
from pathlib import Path

from .config import settings
from .database import connect, execute, fetch_one, get_video, set_job_status, utc_now
from .media import create_mask, detect_watermark_region, source_path
from .providers import get_provider


class JobWorker:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task] = []
        self.running = False

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.tasks = [
            asyncio.create_task(self._runner(index), name=f"watermark-worker-{index}")
            for index in range(max(1, settings.concurrency))
        ]

    async def pause(self) -> None:
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        execute(
            "UPDATE jobs SET status='queued', updated_at=? WHERE status='processing'",
            (utc_now(),),
        )

    async def _runner(self, index: int) -> None:
        del index
        while self.running:
            job = self._claim_next_job()
            if not job:
                await asyncio.sleep(1)
                continue
            await self._process(job)

    def _claim_next_job(self) -> dict | None:
        with connect() as connection:
            row = connection.execute(
                """
                UPDATE jobs
                SET status='processing', attempt=attempt+1, error=NULL, updated_at=?
                WHERE id=(
                    SELECT id FROM jobs WHERE status='queued' ORDER BY created_at, id LIMIT 1
                ) AND status='queued'
                RETURNING *
                """,
                (utc_now(),),
            ).fetchone()
            return dict(row) if row else None

    async def _process(self, job: dict) -> None:
        video = get_video(job["video_id"])
        if not video:
            set_job_status(job["id"], "failed", error="Video record not found")
            return
        media_path = source_path(video["relative_path"])
        output_path = Path(settings.output_dir) / video["category"] / video["filename"]
        try:
            if not media_path.exists():
                raise FileNotFoundError(f"Source file not found: {media_path}")
            if video.get("mask_x") is None:
                region = await asyncio.to_thread(
                    detect_watermark_region, video["id"], media_path
                )
                execute(
                    """
                    UPDATE videos SET mask_x=?, mask_y=?, mask_width=?, mask_height=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        region["x"], region["y"], region["width"], region["height"],
                        utc_now(), video["id"],
                    ),
                )
                video = get_video(video["id"])
            mask_path = await asyncio.to_thread(create_mask, video, media_path)
            provider = get_provider()

            async def progress(value: float, provider_job_id: str | None) -> None:
                set_job_status(
                    job["id"], "processing", progress=value,
                    provider_job_id=provider_job_id,
                )

            await provider.process(
                media_path,
                mask_path,
                output_path,
                {
                    "video_id": video["id"],
                    "material_id": video["material_id"],
                    "relative_path": video["relative_path"],
                },
                progress,
            )
            set_job_status(
                job["id"], "completed", progress=1.0,
                output_path=str(output_path), error="",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            current = fetch_one("SELECT attempt FROM jobs WHERE id=?", (job["id"],)) or {}
            retry = int(current.get("attempt", 1)) <= settings.max_retries
            set_job_status(job["id"], "queued" if retry else "failed", error=str(error))


worker = JobWorker()
