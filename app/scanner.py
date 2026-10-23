from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .database import sync_videos


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

    media_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    files_by_name: dict[str, list[Path]] = defaultdict(list)
    for media_path in media_files:
        files_by_name[media_path.name.casefold()].append(media_path)

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
                expected_path = root / category / filename
                matches = files_by_name.get(filename.casefold(), [])
                media_path = expected_path if expected_path.is_file() else None
                if media_path is None and len(matches) == 1:
                    media_path = matches[0]
                exists = media_path is not None
                relative_path = str(
                    media_path.relative_to(root) if media_path else Path(category) / filename
                )
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
                indexed_paths.add(relative_path.casefold())
                imported += 1
                missing += int(not exists)

    discovered = 0
    for media_path in media_files:
        relative = media_path.relative_to(root)
        relative_path = str(relative)
        if relative_path.casefold() in indexed_paths:
            continue
        items.append({
            "relative_path": relative_path,
            "category": relative.parts[0] if len(relative.parts) > 1 else "Uncategorized",
            "filename": media_path.name,
            "title": media_path.stem,
            "size_bytes": media_path.stat().st_size,
            "source_exists": True,
        })
        indexed_paths.add(relative_path.casefold())
        discovered += 1
    sync_videos(items)
    return {"imported": imported, "discovered": discovered, "missing": missing}
