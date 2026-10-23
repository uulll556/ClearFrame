from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import ROOT_DIR, settings


CACHE_DIR = ROOT_DIR / "cache"
THUMB_DIR = CACHE_DIR / "thumbnails"
MASK_DIR = CACHE_DIR / "masks"


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def source_path(relative_path: str) -> Path:
    return Path(settings.source_dir) / relative_path


def get_video_size(path: Path) -> tuple[int, int]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        return int(width), int(height)
    finally:
        reader.close()


def get_thumbnail(video_id: int, path: Path, second: float = 1.0) -> Path:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha1(f"{path}:{path.stat().st_mtime_ns}".encode()).hexdigest()[:12]
    target = THUMB_DIR / f"{video_id}-{signature}.jpg"
    if target.exists():
        return target
    subprocess.run(
        [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error",
            "-ss", str(max(second, 0)), "-i", str(path), "-frames:v", "1",
            "-vf", "scale=640:-2", "-q:v", "3", "-y", str(target),
        ],
        check=True,
    )
    return target


def detect_watermark_region(video_id: int, path: Path) -> dict[str, float]:
    import numpy as np
    from skimage.color import rgb2gray
    from skimage.feature import canny, match_template

    image = Image.open(get_thumbnail(video_id, path)).convert("RGB")
    edges = canny(rgb2gray(np.asarray(image) / 255.0), sigma=1.2)
    font_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    if not font_path.exists():
        return {
            "x": settings.mask_x, "y": settings.mask_y,
            "width": settings.mask_width, "height": settings.mask_height,
            "score": 0.0,
        }
    text = "\u65b0\u7247\u573a\u6b63\u7248\u7d20\u6750"
    best: tuple[float, float, float, float, float] | None = None
    for ratio in (0.30, 0.34, 0.38, 0.42, 0.46):
        target_width = int(image.width * ratio)
        font = ImageFont.truetype(str(font_path), max(12, int(target_width / 7.2)))
        box = font.getbbox(text)
        template = Image.new("L", (box[2] - box[0] + 8, box[3] - box[1] + 8), 0)
        ImageDraw.Draw(template).text(
            (4 - box[0], 4 - box[1]), text, font=font, fill=255
        )
        template_edges = canny(np.asarray(template) / 255.0, sigma=0.8)
        result = match_template(edges, template_edges, pad_input=False)
        y, x = np.unravel_index(np.argmax(result), result.shape)
        candidate = (
            float(result[y, x]), float(x / image.width), float(y / image.height),
            float(template.width / image.width), float(template.height / image.height),
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None or best[0] < 0.07:
        return {
            "x": settings.mask_x, "y": settings.mask_y,
            "width": settings.mask_width, "height": settings.mask_height,
            "score": best[0] if best else 0.0,
        }
    score, x, y, width, height = best
    padded_x = max(0.0, x - 0.055)
    padded_y = max(0.0, y - 0.03)
    return {
        "x": padded_x,
        "y": padded_y,
        "width": min(1.0 - padded_x, width + 0.105),
        "height": min(1.0 - padded_y, height + 0.06),
        "score": score,
    }


def create_mask(video: dict, path: Path) -> Path:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    width = video.get("width")
    height = video.get("height")
    if not width or not height:
        width, height = get_video_size(path)

    region = (
        video.get("mask_x") if video.get("mask_x") is not None else settings.mask_x,
        video.get("mask_y") if video.get("mask_y") is not None else settings.mask_y,
        video.get("mask_width") if video.get("mask_width") is not None else settings.mask_width,
        video.get("mask_height") if video.get("mask_height") is not None else settings.mask_height,
    )
    x, y, region_width, region_height = region
    padding = max(2, round(min(width, height) * 0.006))
    box = (
        max(0, round(x * width) - padding),
        max(0, round(y * height) - padding),
        min(width, round((x + region_width) * width) + padding),
        min(height, round((y + region_height) * height) + padding),
    )
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=padding * 2, fill=255)
    if settings.mask_feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(settings.mask_feather))
    target = MASK_DIR / f"{video['id']}.png"
    mask.save(target)
    return target
