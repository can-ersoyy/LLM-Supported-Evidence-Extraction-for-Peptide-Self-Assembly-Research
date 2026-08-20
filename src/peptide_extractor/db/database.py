import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.engine import Engine

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# Default SQLite database path in project's data directory
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "papers.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Create and return a SQLModel/SQLAlchemy engine."""
    url = db_url or DATABASE_URL

    # Ensure parent directory exists for SQLite databases
    if url.startswith("sqlite:///"):
        sqlite_file_path = url.replace("sqlite:///", "")
        if sqlite_file_path and sqlite_file_path != ":memory:":
            Path(sqlite_file_path).parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=echo, connect_args=connect_args)


# Shared default engine instance
engine = get_engine()


def init_db(db_engine: Optional[Engine] = None) -> None:
    """Initialize database tables for all registered SQLModel models."""
    # Ensure all models are imported and registered with SQLModel.metadata
    from peptide_extractor.db.paper_content import PaperContent  # noqa: F401
    from peptide_extractor.db.paper_metadata import PaperMetadata  # noqa: F401

    target_engine = db_engine or engine
    SQLModel.metadata.create_all(target_engine)


@contextmanager
def get_session_context(db_engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    target_engine = db_engine or engine
    with Session(target_engine) as session:
        yield session


def get_session(db_engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Generator for database sessions (useful for dependency injection or manual loops)."""
    target_engine = db_engine or engine
    with Session(target_engine) as session:
        yield session
