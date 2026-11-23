from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Global flag that can be set directly in Python code (e.g., in pipeline.py)
LOG_TO_FILE: bool = False

# Mapping from common step synonyms to standard folder names
STEP_NAME_MAP: Dict[str, str] = {
    "search": "search",
    "fetch": "fetch",
    "fetch_fulltext": "fetch",
    "fetch_paper_fulltexts": "fetch",
    "relevance": "relevance",
    "screening": "relevance",
    "extraction": "extraction",
    "evaluation": "evaluation",
    "deduplicate": "deduplicate",
    "find_abstracts": "find_abstracts",
    "clean_empty": "clean_empty",
    "visualization": "visualization",
    "visualize": "visualization",
}

# Registry to track active FileHandlers for re-entrancy / nesting:
# maps canonical log file path -> (handler, ref_count)
_active_handlers: Dict[str, Tuple[logging.Handler, int]] = {}


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_default_log_dir() -> Path:
    """Return the base log directory, defaulting to <project_root>/docs/logs."""
    env_dir = os.getenv("PEPTIDE_EXTRACTOR_LOG_DIR") or os.getenv("LOG_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return get_project_root() / "data" / "logs"


def normalize_step_name(step: str) -> str:
    """Normalize a pipeline step name to its standard log folder name."""
    cleaned = step.strip().lower()
    return STEP_NAME_MAP.get(cleaned, cleaned)


def is_file_logging_enabled(
    step: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    override: Optional[bool] = None,
) -> bool:
    """Determine whether file logging should be active.

    Order of precedence:
    1. Direct override argument if provided (True / False).
    2. Step-specific configuration setting in config dict (if provided).
    3. Global configuration setting in config dict (if provided).
    4. Module-level LOG_TO_FILE variable.
    5. Environment variables: PEPTIDE_EXTRACTOR_LOG_TO_FILE, LOG_TO_FILE, SAVE_LOGS.
    """
    if override is not None:
        return bool(override)

    if config is not None:
        if step:
            step_cfg = config.get(step, {})
            if isinstance(step_cfg, dict):
                for key in ("log_to_file", "save_logs", "file_logging"):
                    if key in step_cfg:
                        return bool(step_cfg[key])
        for key in ("log_to_file", "save_logs", "file_logging"):
            if key in config:
                return bool(config[key])

    if LOG_TO_FILE:
        return True

    for env_var in (
        "PEPTIDE_EXTRACTOR_LOG_TO_FILE",
        "LOG_TO_FILE",
        "SAVE_LOGS",
        "FILE_LOGGING",
    ):
        val = os.getenv(env_var)
        if val is not None:
            return val.strip().lower() in ("1", "true", "yes", "on")

    return False


NOISY_LIBRARY_LOGGERS: Tuple[str, ...] = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "google",
    "google_genai",
    "google.genai",
    "google.auth",
    "google.api_core",
    "grpc",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_google_genai",
    "langchain_ollama",
    "langchain_openrouter",
    "openai",
    "anthropic",
    "azure",
    "azure.core",
    "sqlalchemy",
    "matplotlib",
    "PIL",
    "spacy",
    "thinc",
    "fitz",
    "pymupdf",
    "ollama",
)


def silence_library_loggers(level: int = logging.ERROR) -> None:
    """Suppress INFO/WARNING noise from third-party libraries (e.g. HTTP clients, LLM SDKs)."""
    for lib_name in NOISY_LIBRARY_LOGGERS:
        lib_logger = logging.getLogger(lib_name)
        if lib_logger.level == logging.NOTSET or lib_logger.level < level:
            lib_logger.setLevel(level)


@contextmanager
def step_logger(
    step: str,
    run_identifier: Union[int, str],
    enabled: Optional[bool] = None,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
) -> Generator[Optional[logging.Handler], None, None]:
    """Context manager to attach a FileHandler for a pipeline step's logs.

    Logs will be written to:
        {log_dir}/[normalized_step]/{run_identifier}.log

    Re-entrant: If an outer context already attached a FileHandler for the exact
    same file, the reference count is incremented and the handler will not be
    removed or closed until the outermost context exits.
    """
    if enabled is None:
        enabled = is_file_logging_enabled(step=step)

    if not enabled:
        yield None
        return

    silence_library_loggers()

    step_name = normalize_step_name(step)
    base_dir = Path(log_dir).resolve() if log_dir else get_default_log_dir()
    target_dir = base_dir / step_name
    target_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = target_dir / f"{run_identifier}.log"
    canonical_key = str(log_file_path.resolve())

    root_logger = logging.getLogger()

    # Ensure root logger passes through messages at the requested level
    if root_logger.level == logging.NOTSET or root_logger.level > level:
        root_logger.setLevel(level)

    if canonical_key in _active_handlers:
        existing_handler, ref_count = _active_handlers[canonical_key]
        _active_handlers[canonical_key] = (existing_handler, ref_count + 1)
        try:
            yield existing_handler
        finally:
            curr_handler, curr_count = _active_handlers.get(
                canonical_key, (existing_handler, 1)
            )
            if curr_count <= 1:
                _active_handlers.pop(canonical_key, None)
                try:
                    root_logger.removeHandler(curr_handler)
                except Exception:
                    pass
                curr_handler.flush()
                curr_handler.close()
            else:
                _active_handlers[canonical_key] = (curr_handler, curr_count - 1)
        return

    handler = logging.FileHandler(str(log_file_path), mode="a", encoding="utf-8")
    handler.setLevel(level)
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    _active_handlers[canonical_key] = (handler, 1)

    try:
        yield handler
    finally:
        curr_handler, curr_count = _active_handlers.get(
            canonical_key, (handler, 1)
        )
        if curr_count <= 1:
            _active_handlers.pop(canonical_key, None)
            try:
                root_logger.removeHandler(curr_handler)
            except Exception:
                pass
            curr_handler.flush()
            curr_handler.close()
        else:
            _active_handlers[canonical_key] = (curr_handler, curr_count - 1)
