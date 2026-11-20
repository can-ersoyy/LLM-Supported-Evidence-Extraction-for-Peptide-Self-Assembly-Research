# Alembic with SQLModel Reference Guide

This document is a quick, practical cheat sheet for database migrations using **Alembic** and **SQLModel** in this project.

---

## 1. Initial Setup (One-Time)

### A. Install Alembic
```bash
uv add alembic
```

### B. Initialize Directory
From the project root (`/var/home/can/Documents/idp-docs/project`):
```bash
alembic init alembic
```

### C. Configure `alembic/env.py`
Open `alembic/env.py` and ensure the following:

```python
from sqlmodel import SQLModel

# 1. Import all SQLModel models so Alembic can detect them
from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.database import DATABASE_URL

# 2. Set target_metadata to SQLModel.metadata
target_metadata = SQLModel.metadata

# 3. Dynamically set the database connection URL
config.set_main_option("sqlalchemy.url", DATABASE_URL)
```

> **SQLite Tip (`render_as_batch=True`)**:
> SQLite has limitations altering tables (like dropping columns or foreign keys). In `alembic/env.py`, inside `run_migrations_online()`, enable batch mode:
> ```python
> with connectable.connect() as connection:
>     context.configure(
>         connection=connection,
>         target_metadata=target_metadata,
>         render_as_batch=True,  # <-- Add this for full SQLite support
>     )
>     with context.begin_transaction():
>         context.run_migrations()
> ```

---

## 2. Daily Workflow Cheat Sheet

Whenever you add, modify, or delete columns in your SQLModel classes:

### Step 1: Update Your SQLModel Class
Example: Add a new column in `paper_metadata.py`:
```python
journal: Optional[str] = Field(default=None)
```

### Step 2: Auto-generate Migration Script
```bash
alembic revision --autogenerate -m "add journal to paper_metadata"
```
* Alembic detects the difference between your Python models and the database.
* It generates a new migration script in `alembic/versions/<revision_id>_add_journal_to_paper_metadata.py`.

### Step 3: Review the Generated Migration File
Always open the generated file in `alembic/versions/` to verify the `upgrade()` and `downgrade()` functions.

### Step 4: Apply the Migration to the Database
```bash
alembic upgrade head
```
Your database is now up to date!

---

## 3. Essential Command Reference

| Action | Command |
| :--- | :--- |
| **Apply all pending migrations** | `alembic upgrade head` |
| **Upgrade by 1 revision** | `alembic upgrade +1` |
| **Rollback last migration** | `alembic downgrade -1` |
| **Rollback to a specific revision** | `alembic downgrade <revision_id>` |
| **Rollback everything (base state)** | `alembic downgrade base` |
| **Check current database revision** | `alembic current` |
| **View migration history** | `alembic history --verbose` |
| **Create manual (empty) revision** | `alembic revision -m "custom_sql"` |

---

## 4. Troubleshooting & FAQ

### Q: `Target database is not up to date`
Run:
```bash
alembic upgrade head
```

### Q: Alembic does not detect my new model/table
Make sure the new model is imported in `alembic/env.py` (or in `peptide_extractor/db/__init__.py`), otherwise Python won't register it in `SQLModel.metadata`.

### Q: How to reset the database completely during early dev
If you ever want to start clean:
1. Delete the SQLite file: `rm data/papers.db`
2. Run migrations: `alembic upgrade head`
