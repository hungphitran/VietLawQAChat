"""Load app config (YAML) + .env. LLM values are env-overridable."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # pull HF_TOKEN / LLM_API_KEY / LLM_* from .env

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "app" / "app.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else Path(os.getenv("APP_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path) as f:
        return yaml.safe_load(f)
