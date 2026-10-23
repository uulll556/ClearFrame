from pathlib import Path

from PIL import Image, ImageDraw

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


def test_multi_frame_detection_ignores_moving_background() -> None:
    font_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    if not font_path.exists():
        return
    width, height = 480, 270
    watermark = media._build_watermark_template(width, 0.36)
    frames = []
    for index in range(5):
        image = Image.new("RGB", (width, height), (25 + index * 18, 45, 70))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (index * 55, 25 + index * 8, index * 55 + 100, 170),
            fill=(180, 65 + index * 20, 40),
        )
        image.paste((245, 245, 245), (145, 180), watermark)
        frames.append(image)

    region = media.detect_region_from_images(frames)

    assert region["method"] == "multi_frame_complete_scale_v7"
    assert abs(region["x"] - 145 / width) < 0.04
    assert 0.55 < region["y"] < 0.75
    assert region["width"] < 0.48
    assert region["height"] < 0.13
    assert region["score"] > 0.1


def test_adaptive_region_scales_padding_with_text_height() -> None:
    small = media._adaptive_mask_region(480, 270, 165, 176, 134, 27)
    large = media._adaptive_mask_region(480, 270, 145, 80, 169, 31)
    extra_large = media._adaptive_mask_region(480, 270, 120, 79, 224, 36)

    assert small[2] < large[2]
    assert small[3] < large[3]
    assert small[2] < 0.34
    assert small[3] < 0.10
    assert large[3] > 0.128
    assert round(extra_large[2] * 480) == 240
    assert round(extra_large[3] * 270) == 48


def test_complete_scale_preference_rejects_partial_text_match() -> None:
    assert media._candidate_rank(0.120, 0.36) > media._candidate_rank(0.122, 0.24)


def test_aligned_candidate_recovers_missing_logo_without_expanding_right() -> None:
    candidates = [
        (media._candidate_rank(0.147, 0.32), 0.147, 0.32, 152, 174, 182, 30),
        (media._candidate_rank(0.133, 0.40), 0.133, 0.40, 124, 171, 224, 36),
        (media._candidate_rank(0.108, 0.44), 0.108, 0.44, 125, 170, 246, 37),
    ]

    selected, recover_logo_anchor = media._choose_complete_candidate(candidates, 480, 270)

    assert selected is not None
    assert selected[2] == 0.32
    assert recover_logo_anchor is True

    normal = media._adaptive_mask_region(480, 270, 152, 174, 182, 30)
    recovered = media._adaptive_mask_region(480, 270, 152, 174, 182, 30, True)

    assert round(recovered[0] * 480) == round(normal[0] * 480) - 21
    assert round((recovered[0] + recovered[2]) * 480) == round(
        (normal[0] + normal[2]) * 480
    ) + 5
    assert round(recovered[3] * 270) == round(normal[3] * 270) + 4
