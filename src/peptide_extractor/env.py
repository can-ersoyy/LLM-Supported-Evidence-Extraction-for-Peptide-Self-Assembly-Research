from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

root = Path(__file__).resolve().parent.parent.parent

def load_env() -> None:
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(find_dotenv(usecwd=True))
