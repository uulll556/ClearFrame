from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .config import settings


ProgressCallback = Callable[[float, str | None], Awaitable[None]]


class ProviderError(RuntimeError):
    pass


class MockProvider:
    async def process(
        self,
        video_path: Path,
        mask_path: Path,
        output_path: Path,
        metadata: dict,
        progress: ProgressCallback,
    ) -> str | None:
        del mask_path, metadata
        await progress(0.2, None)
        await asyncio.sleep(0.4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, video_path, output_path)
        await progress(1.0, None)
        return None


class LocalInpaintProvider:
    async def process(
        self,
        video_path: Path,
        mask_path: Path,
        output_path: Path,
        metadata: dict,
        progress: ProgressCallback,
    ) -> str | None:
        del metadata
        try:
            from .local_inpaint import remove_watermark
        except ImportError as error:
            raise ProviderError(
                "Local provider needs opencv-python and pypatchmatch: "
                "python -m pip install -r requirements.txt"
            ) from error
        await remove_watermark(
            video_path,
            mask_path,
            output_path,
            progress,
            grow=max(0, int(settings.local_mask_grow)),
            patch_size=max(3, int(settings.local_patch_size)),
        )
        return None


class ProPainterProvider:
    async def process(
        self,
        video_path: Path,
        mask_path: Path,
        output_path: Path,
        metadata: dict,
        progress: ProgressCallback,
    ) -> str | None:
        del metadata
        try:
            from .propainter import remove_watermark
        except ImportError as error:
            raise ProviderError(
                "ProPainter provider needs PyTorch with CUDA; see README."
            ) from error
        await remove_watermark(
            video_path,
            mask_path,
            output_path,
            progress,
            grow=max(0, int(settings.local_mask_grow)),
        )
        return None


class GenericApiProvider:
    async def process(
        self,
        video_path: Path,
        mask_path: Path,
        output_path: Path,
        metadata: dict,
        progress: ProgressCallback,
    ) -> str | None:
        api_url = settings.effective_api_url
        if not api_url:
            raise ProviderError("WATERMARK_API_URL is not configured")
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"

        timeout = httpx.Timeout(settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            await progress(0.08, None)
            with video_path.open("rb") as video_handle, mask_path.open("rb") as mask_handle:
                response = await client.post(
                    api_url,
                    headers=headers,
                    data={
                        "metadata": json.dumps(metadata, ensure_ascii=False),
                        "prompt": "Remove the watermark and reconstruct the background consistently across frames.",
                    },
                    files={
                        "video": (video_path.name, video_handle, "video/mp4"),
                        "mask": (mask_path.name, mask_handle, "image/png"),
                    },
                )
            response.raise_for_status()

            if response.headers.get("content-type", "").startswith("video/"):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)
                await progress(1.0, None)
                return None

            payload = response.json()
            provider_job_id = str(payload.get("job_id") or payload.get("id") or "") or None
            output_url = payload.get("output_url") or payload.get("url")
            status_url = payload.get("status_url")
            await progress(0.2, provider_job_id)

            while not output_url and status_url:
                await asyncio.sleep(settings.poll_interval_seconds)
                status_response = await client.get(status_url, headers=headers)
                status_response.raise_for_status()
                status_payload = status_response.json()
                status = str(status_payload.get("status", "")).lower()
                if status in {"failed", "error", "cancelled"}:
                    raise ProviderError(status_payload.get("error") or f"Remote status: {status}")
                output_url = status_payload.get("output_url") or status_payload.get("url")
                remote_progress = float(status_payload.get("progress", 0) or 0)
                await progress(min(0.9, 0.2 + remote_progress * 0.7), provider_job_id)

            if not output_url:
                raise ProviderError("API response has no output_url or status_url")
            download = await client.get(output_url, headers=headers)
            download.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(output_path.suffix + ".part")
            temporary.write_bytes(download.content)
            temporary.replace(output_path)
            await progress(1.0, provider_job_id)
            return provider_job_id


def get_provider():
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "local":
        return LocalInpaintProvider()
    if settings.provider == "propainter":
        return ProPainterProvider()
    if settings.provider == "generic":
        return GenericApiProvider()
    raise ProviderError(f"Unsupported API adapter: {settings.provider}")
