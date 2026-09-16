import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Type, TypeVar, Union, overload

from dotenv import find_dotenv, load_dotenv
from sqlmodel import Session, SQLModel, create_engine, select
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
    from peptide_extractor.db.paper_relevance import PaperRelevance  # noqa: F401

    target_engine = db_engine or engine
    SQLModel.metadata.create_all(target_engine)


@contextmanager
def get_session_context(
    db_engine: Optional[Engine] = None, expire_on_commit: bool = False
) -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    target_engine = db_engine or engine
    with Session(target_engine, expire_on_commit=expire_on_commit) as session:
        yield session


def get_session(
    db_engine: Optional[Engine] = None, expire_on_commit: bool = False
) -> Generator[Session, None, None]:
    """Generator for database sessions (useful for dependency injection or manual loops)."""
    target_engine = db_engine or engine
    with Session(target_engine, expire_on_commit=expire_on_commit) as session:
        yield session


T = TypeVar("T", bound=SQLModel)


def _resolve_model(model_or_name: Union[Type[SQLModel], str]) -> Type[SQLModel]:
    """Resolve a model class from a SQLModel class or string table name."""
    if not isinstance(model_or_name, str):
        if isinstance(model_or_name, type) and issubclass(model_or_name, SQLModel):
            return model_or_name
        raise ValueError(f"Expected SQLModel subclass or string, got {type(model_or_name)}")

    table_name = model_or_name.strip().lower()
    from peptide_extractor.db.paper_metadata import PaperMetadata
    from peptide_extractor.db.paper_content import PaperContent
    from peptide_extractor.db.paper_relevance import PaperRelevance

    registry = {
        "paper_metadata": PaperMetadata,
        "paper_content": PaperContent,
        "paper_relevance": PaperRelevance,
    }
    try:
        from peptide_extractor.db.peptide_entry import PeptideEntry
        if hasattr(PeptideEntry, "__tablename__"):
            registry["peptide_entry"] = PeptideEntry
    except ImportError:
        pass

    if table_name not in registry:
        raise ValueError(
            f"Unknown table name '{model_or_name}'. Available tables: {list(registry.keys())}"
        )
    return registry[table_name]


@overload
def get_records(
    model: Type[T],
    session: Optional[Session] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    **filters: Any,
) -> List[T]: ...


@overload
def get_records(
    model: str,
    session: Optional[Session] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    **filters: Any,
) -> List[SQLModel]: ...


def get_records(
    model: Union[Type[T], str],
    session: Optional[Session] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    **filters: Any,
) -> List[Any]:
    """Retrieve records from a database table matching arbitrary column values.

    Args:
        model: SQLModel class or string table name (e.g. PaperMetadata, "paper_metadata").
        session: Optional SQLAlchemy/SQLModel Session. If omitted, uses get_session_context().
        limit: Optional maximum number of records to return.
        offset: Number of records to skip (default 0).
        **filters: Column name to value mappings. Supports scalar equality,
            in-list membership for sequences, and IS NULL for None.

    Returns:
        List of matching model instances.
    """
    target_model = _resolve_model(model)

    def _query(s: Session) -> List[Any]:
        statement = select(target_model)
        for col_name, value in filters.items():
            if not hasattr(target_model, col_name):
                raise ValueError(
                    f"Model '{target_model.__name__}' has no attribute or column '{col_name}'"
                )
            col_attr = getattr(target_model, col_name)
            if value is None:
                statement = statement.where(col_attr.is_(None))
            elif isinstance(value, (list, tuple, set)):
                statement = statement.where(col_attr.in_(value))
            else:
                statement = statement.where(col_attr == value)

        if offset:
            statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)

        return list(s.exec(statement).all())

    if session:
        return _query(session)
    with get_session_context() as s:
        return _query(s)
