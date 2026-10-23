from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"


@dataclass(slots=True)
class Settings:
    source_dir: str = ""
    output_dir: str = str(ROOT_DIR / "output")
    provider: str = "mock"
    api_url: str = ""
    concurrency: int = 2
    max_retries: int = 2
    poll_interval_seconds: float = 4.0
    request_timeout_seconds: float = 180.0
    mask_x: float = 0.34
    mask_y: float = 0.70
    mask_width: float = 0.38
    mask_height: float = 0.13
    mask_feather: int = 6

    @property
    def api_key(self) -> str:
        return os.getenv("WATERMARK_API_KEY", "")

    @property
    def effective_api_url(self) -> str:
        return self.api_url or os.getenv("WATERMARK_API_URL", "")

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key_configured"] = bool(self.api_key)
        data["api_url"] = self.effective_api_url
        return data


def load_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = Settings()
    if SETTINGS_FILE.exists():
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(Settings)}
        for key, value in saved.items():
            if key in allowed:
                setattr(current, key, value)
    if os.getenv("WATERMARK_PROVIDER"):
        current.provider = os.environ["WATERMARK_PROVIDER"]
    return current


def save_settings(current: Settings) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(
        json.dumps(asdict(current), ensure_ascii=False, indent=2), encoding="utf-8"
    )


settings = load_settings()
