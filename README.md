# ClearFrame Watermark Batch Console

Local batch orchestration for licensed video assets. The application scans the
network share, detects the watermark location, builds a per-video mask, submits
the video to a remote inpainting API, and archives the returned file.

> Only process footage that you own or are licensed to modify.

## Dataset findings

- 3,332 videos in 19 categories
- About 11.45 GB total, 3.52 MB average, 104.61 MB maximum
- The watermark text is consistent, but its vertical position differs by video
- The watermark remains fixed within an individual video

Because the position varies between videos, one global rectangle is not enough.
ClearFrame uses lightweight local template matching on a preview frame. This is
coordinate detection only; video restoration remains entirely in the remote API.

## Pipeline

1. Read the supplied CSV manifest and reconcile files on the network share.
2. Generate a cached preview frame with the bundled FFmpeg binary.
3. Match the known watermark text and save a normalized mask region.
4. Allow an operator to review or adjust the region in the preview drawer.
5. Queue selected videos in SQLite with retry and resume state.
6. Upload `video + PNG mask + metadata` to the configured remote API.
7. Poll the remote job and download the result into the category output folder.

## Start

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

or `.\start.ps1` for the default port. When port 8765 is already taken,
use `.\start.ps1 -Port 8766`.

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The initial source
directory is already configured for the supplied network share and can be
changed from the API settings page.

The default provider is `mock`; it copies the input to the output folder so the
complete queue can be tested without API charges. For real removal switch to
`propainter` (GPU) or `local` (CPU).

## Local repair provider

Set the provider to `local` to remove the watermark without a remote service.
It uses OpenCV plus PatchMatch exemplar inpainting:

1. Sample up to 48 frames and take the temporal median of the local contrast
   residual. A static watermark survives the median, moving backgrounds do not,
   so the repair mask follows the lettering instead of blanking a rectangle.
2. Keep the speech-bubble logo by filling the glyph mask and adding the bubble
   ellipse to its left.
3. Crop the watermark region in every frame, inpaint it with PatchMatch, and
   re-encode at the source resolution, frame rate and audio.

Tune `local_mask_grow` (pixels of mask dilation, default 2) and
`local_patch_size` (PatchMatch radius, default 5) on the settings page when a
watermark has a strong glow or sits on a thin structure. Local repair is
CPU-bound at roughly 1-2 seconds per frame; queue a few clips first to check
quality.

Known limitation: a watermark that covers a thin, high-contrast object (for
example a sword blade) cannot always be reconstructed perfectly, because the
hidden pixels never appear in any frame.

## ProPainter provider (GPU)

Set the provider to `propainter` to run ProPainter (ICCV 2023, flow-guided video
inpainting) locally. Unlike per-frame repair it propagates content between
frames, so the repaired area stays temporally consistent.

Requirements:

- NVIDIA GPU. The CUDA build of PyTorch is required:
  ```powershell
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
  python -m pip install einops matplotlib av
  ```
- The ProPainter source in `tools/ProPainter` and its three checkpoints in
  `tools/ProPainter/weights` (`ProPainter.pth`, `raft-things.pth`,
  `recurrent_flow_completion.pth`, about 200 MB). The source is published at
  [sczhou/ProPainter](https://huggingface.co/spaces/sczhou/ProPainter) and the
  weights at [camenduru/ProPainter](https://huggingface.co/camenduru/ProPainter).

Only the watermark band is repainted, not the whole frame. Clips up to 720p cut
the band out and process it at native resolution (`WATERMARK_PROPAINTER_BAND=1`),
which keeps the repair sharp; 1080p sources use a whole-frame pass capped at
960x540 (`WATERMARK_PROPAINTER_BAND=0`) because a native band loses the scene
context that a thin high-contrast object needs. The frames are pasted back with a
tight feather, leftover glyph pixels get a small Telea pass, and a local contrast
pass restores micro-detail inside the patch so it does not read as a blurred
rectangle. Tune the pass with `WATERMARK_PROPAINTER_MAX_WIDTH`/`_MAX_HEIGHT`,
`WATERMARK_PROPAINTER_SUBVIDEO` (default 8) and `WATERMARK_PROPAINTER_NEIGHBOR`
(default 4); both shrink memory use on a card that also drives the desktop.

The mask is detected adaptively (per-pixel median over sample frames, or a
morphological opening when the overlay is a solid bright block), so glyphs over
fire, cloth or a blade are all covered. Text drawn directly on top of a very
bright thin object can only be smoothed, not rebuilt — the pixels underneath are
gone.

ProPainter is research code under the S-Lab license; check it before commercial
use.

## Generic remote API contract

Set the provider to `generic`, then configure:

```powershell
$env:WATERMARK_API_KEY = "your-key"
$env:WATERMARK_API_URL = "https://provider.example/v1/video-inpaint"
```

ClearFrame sends a multipart request:

| Field | Type | Description |
| --- | --- | --- |
| `video` | file | Source video |
| `mask` | PNG file | White repair area on a black background |
| `metadata` | JSON string | Video ID, material ID and relative path |
| `prompt` | string | Temporal reconstruction instruction |

Supported responses:

```json
{"output_url": "https://cdn.example/result.mp4"}
```

or an asynchronous job:

```json
{
  "job_id": "job_123",
  "status_url": "https://provider.example/v1/jobs/job_123"
}
```

The status endpoint should return `status`, optional numeric `progress`, and an
`output_url` when complete. A direct `video/*` response is also accepted.

## Production recommendations

- Start with concurrency `2`; increase only after confirming provider limits.
- Run a 30-50 video quality sample before submitting the entire library.
- Keep the original share read-only and write results to a separate volume.
- Prefer API providers that preserve source resolution, frame rate and audio.
- Add webhook support when the provider offers it to reduce polling traffic.
- Review low-confidence or visually complex clips before batch submission.

## Tests

```powershell
python -m pytest -q
```

## Project layout

- `app/main.py` - FastAPI routes and static application host
- `app/scanner.py` - CSV/network share indexing
- `app/media.py` - preview, watermark detection and mask generation
- `app/local_inpaint.py` - local PatchMatch repair engine (mask extraction, inpaint, encode)
- `app/propainter.py` - ProPainter GPU engine (mask, subprocess runner, re-encode)
- `app/providers.py` - mock, propainter, local and generic remote API adapters
- `app/worker.py` - resumable concurrent job runner
- `app/static/` - browser console
- `tools/ProPainter/` - vendored ProPainter source and weights (optional)
- `data/watermark.db` - local queue state, created at runtime
- `cache/` - generated previews and masks
- `output/` - returned videos grouped by source category
