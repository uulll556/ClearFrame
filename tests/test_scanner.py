from pathlib import Path

from app import database
from app.scanner import parse_duration, scan_source


def test_parse_duration() -> None:
    assert parse_duration("00:01:05") == 65
    assert parse_duration("") == 0
    assert parse_duration("invalid") == 0


def test_scan_source_uses_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    category = "\u683c\u6597-\u62f3\u51fb"
    filename = "clip.mp4"
    media_dir = tmp_path / category
    media_dir.mkdir()
    (media_dir / filename).write_bytes(b"video")
    csv_text = (
        "\u5e8f\u53f7,\u5206\u7c7b,\u6587\u4ef6\u540d,\u6807\u9898,\u65f6\u957f,\u5206\u8fa8\u7387,\u5927\u5c0fMB,\u7d20\u6750ID\n"
        f"1,{category},{filename},Example,00:00:07,,0.01,asset-1\n"
    )
    (tmp_path / "\u7d20\u6750\u6e05\u5355.csv").write_text(csv_text, encoding="utf-8")

    result = scan_source(str(tmp_path))
    row = database.fetch_one("SELECT * FROM videos")

    assert result == {"imported": 1, "discovered": 0, "missing": 0}
    assert row["category"] == category
    assert row["duration_seconds"] == 7
    assert row["source_exists"] == 1


def test_scan_source_resolves_csv_file_in_nested_directory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    category = "\u81ea\u7136\u98ce\u5149"
    subcategory = "\u68ee\u6797\u690d\u88ab"
    filename = "nested.mp4"
    media_dir = tmp_path / category / subcategory
    media_dir.mkdir(parents=True)
    (media_dir / filename).write_bytes(b"video")
    csv_text = (
        "\u5e8f\u53f7,\u5206\u7c7b,\u6587\u4ef6\u540d,\u6807\u9898,\u65f6\u957f,\u5927\u5c0fMB,\u7d20\u6750ID\n"
        f"1,{category},{filename},Nested,00:00:05,0.01,asset-2\n"
    )
    (tmp_path / "\u7d20\u6750\u6e05\u5355.csv").write_text(csv_text, encoding="utf-8")

    result = scan_source(str(tmp_path))
    row = database.fetch_one("SELECT * FROM videos")

    assert result == {"imported": 1, "discovered": 0, "missing": 0}
    assert row["relative_path"] == str(Path(category) / subcategory / filename)
    assert row["category"] == category


def test_scan_source_discovers_unlisted_files_and_removes_stale_rows(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    database.upsert_video({
        "relative_path": "old/clip.mp4",
        "category": "old",
        "filename": "clip.mp4",
    })
    category = tmp_path / "new"
    category.mkdir()
    (category / "extra.mp4").write_bytes(b"video")

    result = scan_source(str(tmp_path))
    rows = database.fetch_all("SELECT relative_path FROM videos")

    assert result == {"imported": 0, "discovered": 1, "missing": 0}
    assert rows == [{"relative_path": str(Path("new") / "extra.mp4")}]
