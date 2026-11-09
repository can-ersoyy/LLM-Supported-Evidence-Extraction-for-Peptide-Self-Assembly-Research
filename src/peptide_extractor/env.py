from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(root / ".env")
