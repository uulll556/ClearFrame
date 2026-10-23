from pathlib import Path

from PIL import Image

from app import local_inpaint, providers
from app.config import settings


def test_local_provider_selected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "provider", "local")
    assert isinstance(providers.get_provider(), providers.LocalInpaintProvider)


def test_mask_bbox_reads_rect(tmp_path: Path) -> None:
    mask = Image.new("L", (200, 100), 0)
    mask.paste(255, (40, 30, 120, 60))
    path = tmp_path / "mask.png"
    mask.save(path)

    assert local_inpaint._mask_bbox(path) == (40, 30, 120, 60)
