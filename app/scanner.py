from __future__ import annotations

import csv
from pathlib import Path

from .database import upsert_videos


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CSV_NAME = "\u7d20\u6750\u6e05\u5355.csv"


def parse_duration(value: str) -> float:
    if not value:
        return 0
    try:
        hours, minutes, seconds = (float(part) for part in value.split(":"))
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError):
        return 0


def scan_source(source_dir: str) -> dict[str, int]:
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory is unavailable: {source_dir}")

    imported = 0
    missing = 0
    csv_path = root / CSV_NAME
    indexed_paths: set[str] = set()
    items: list[dict] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                category = (row.get("\u5206\u7c7b") or "Uncategorized").strip()
                filename = (row.get("\u6587\u4ef6\u540d") or "").strip()
                if not filename:
                    continue
                relative_path = str(Path(category) / filename)
                media_path = root / relative_path
                exists = media_path.exists()
                size_bytes = media_path.stat().st_size if exists else int(
                    float(row.get("\u5927\u5c0fMB") or 0) * 1024 * 1024
                )
                items.append({
                    "relative_path": relative_path,
                    "category": category,
                    "filename": filename,
                    "title": (row.get("\u6807\u9898") or "").strip(),
                    "duration_seconds": parse_duration(row.get("\u65f6\u957f") or ""),
                    "size_bytes": size_bytes,
                    "material_id": (row.get("\u7d20\u6750ID") or "").strip(),
                    "source_exists": exists,
                })
                indexed_paths.add(relative_path.lower())
                imported += 1
                missing += int(not exists)

    discovered = 0
    if imported == 0:
        for media_path in root.rglob("*"):
            if not media_path.is_file() or media_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            relative_path = str(media_path.relative_to(root))
            items.append({
                "relative_path": relative_path,
                "category": media_path.parent.name,
                "filename": media_path.name,
                "title": media_path.stem,
                "size_bytes": media_path.stat().st_size,
                "source_exists": True,
            })
            discovered += 1
    upsert_videos(items)
    return {"imported": imported, "discovered": discovered, "missing": missing}
