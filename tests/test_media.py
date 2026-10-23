from pathlib import Path

from PIL import Image

from app import media


def test_create_mask_uses_normalized_region(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(media, "MASK_DIR", tmp_path)
    video = {
        "id": 7, "width": 100, "height": 50,
        "mask_x": 0.2, "mask_y": 0.3,
        "mask_width": 0.4, "mask_height": 0.2,
    }
    result = media.create_mask(video, tmp_path / "unused.mp4")
    mask = Image.open(result)

    assert mask.size == (100, 50)
    assert mask.getbbox() is not None
    left, top, right, bottom = mask.getbbox()
    assert left <= 20
    assert top <= 15
    assert right >= 60
    assert bottom >= 25
