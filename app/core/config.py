from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROMPTS_DIR = APP_DIR / "prompts"
CONFIG_DIR = BASE_DIR / "config"
FRONTEND_DIR = BASE_DIR / "frontend"
SANDBOX_DIR = BASE_DIR / "sandbox"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Chat With Your Data")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))

MODELS_FILE = CONFIG_DIR / "models.json"
PROMPT_MODES_FILE = CONFIG_DIR / "prompt_modes.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
