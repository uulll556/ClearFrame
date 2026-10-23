from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import DATA_DIR


DB_PATH = DATA_DIR / "watermark.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relative_path TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                filename TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                duration_seconds REAL NOT NULL DEFAULT 0,
                width INTEGER,
                height INTEGER,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                material_id TEXT NOT NULL DEFAULT '',
                source_exists INTEGER NOT NULL DEFAULT 1,
                mask_x REAL,
                mask_y REAL,
                mask_width REAL,
                mask_height REAL,
                mask_score REAL,
                mask_method TEXT,
                mask_review_required INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                progress REAL NOT NULL DEFAULT 0,
                attempt INTEGER NOT NULL DEFAULT 0,
                provider_job_id TEXT,
                output_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(video_id),
                FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_videos_category ON videos(category);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            """
        )
        video_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(videos)").fetchall()
        }
        migrations = {
            "mask_score": "ALTER TABLE videos ADD COLUMN mask_score REAL",
            "mask_method": "ALTER TABLE videos ADD COLUMN mask_method TEXT",
            "mask_review_required": (
                "ALTER TABLE videos ADD COLUMN mask_review_required INTEGER NOT NULL DEFAULT 1"
            ),
        }
        for column, statement in migrations.items():
            if column not in video_columns:
                connection.execute(statement)
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_single_frame', mask_review_required=1
            WHERE mask_x IS NOT NULL AND mask_method IS NULL
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_oversized', mask_review_required=1
            WHERE mask_method='multi_frame_template'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_x_alignment', mask_review_required=1
            WHERE mask_method='multi_frame_adaptive'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_large_padding', mask_review_required=1
            WHERE mask_method='multi_frame_logo_anchor'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_max_padding', mask_review_required=1
            WHERE mask_method='multi_frame_size_aware'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_false_scale', mask_review_required=1
            WHERE mask_method='multi_frame_size_aware_v2'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_extra_large_padding', mask_review_required=1
            WHERE mask_method='multi_frame_complete_scale'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_scale_alignment', mask_review_required=1
            WHERE mask_method='multi_frame_complete_scale_v2'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_logo_recovery', mask_review_required=1
            WHERE mask_method='multi_frame_complete_scale_v3'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_logo_edge_padding', mask_review_required=1
            WHERE mask_method='multi_frame_complete_scale_v4'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_logo_edge_padding_v2', mask_review_required=1
            WHERE mask_method='multi_frame_complete_scale_v5'
            """
        )
        connection.execute(
            """
            UPDATE videos
            SET mask_method='legacy_logo_edge_vertical', mask_review_required=1
            WHERE mask_method='multi_frame_complete_scale_v6'
            """
        )


def fetch_one(query: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(query, parameters).fetchone()
        return dict(row) if row else None


def fetch_all(query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as connection:
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def execute(query: str, parameters: tuple[Any, ...] = ()) -> int:
    with connect() as connection:
        cursor = connection.execute(query, parameters)
        return int(cursor.lastrowid or 0)


def upsert_video(item: dict[str, Any]) -> None:
    upsert_videos([item])


def _upsert_videos(connection: sqlite3.Connection, items: list[dict[str, Any]]) -> None:
    now = utc_now()
    values = [(
        item["relative_path"], item["category"], item["filename"],
        item.get("title", ""), item.get("duration_seconds", 0),
        item.get("size_bytes", 0), item.get("material_id", ""),
        int(item.get("source_exists", True)), now, now,
    ) for item in items]
    connection.executemany(
        """
        INSERT INTO videos (
            relative_path, category, filename, title, duration_seconds,
            size_bytes, material_id, source_exists, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(relative_path) DO UPDATE SET
            category=excluded.category, filename=excluded.filename,
            title=excluded.title, duration_seconds=excluded.duration_seconds,
            size_bytes=excluded.size_bytes, material_id=excluded.material_id,
            source_exists=excluded.source_exists, updated_at=excluded.updated_at
        """,
        values,
    )


def upsert_videos(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    with connect() as connection:
        _upsert_videos(connection, items)


def sync_videos(items: list[dict[str, Any]]) -> None:
    with connect() as connection:
        if items:
            _upsert_videos(connection, items)
        connection.execute(
            "CREATE TEMP TABLE current_scan_paths (relative_path TEXT PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO current_scan_paths (relative_path) VALUES (?)",
            [(item["relative_path"],) for item in items],
        )
        connection.execute(
            """
            DELETE FROM videos
            WHERE relative_path NOT IN (SELECT relative_path FROM current_scan_paths)
            """
        )


def get_video(video_id: int) -> dict[str, Any] | None:
    return fetch_one(
        """
        SELECT v.*, COALESCE(j.status, 'pending') AS status,
               COALESCE(j.progress, 0) AS progress, j.error, j.output_path
        FROM videos v LEFT JOIN jobs j ON j.video_id = v.id
        WHERE v.id = ?
        """,
        (video_id,),
    )


def set_job_status(
    job_id: int,
    status: str,
    *,
    progress: float | None = None,
    error: str | None = None,
    output_path: str | None = None,
    provider_job_id: str | None = None,
) -> None:
    assignments = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, utc_now()]
    for column, value in (
        ("progress", progress), ("error", error),
        ("output_path", output_path), ("provider_job_id", provider_job_id),
    ):
        if value is not None:
            assignments.append(f"{column} = ?")
            values.append(value)
    values.append(job_id)
    execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", tuple(values))
