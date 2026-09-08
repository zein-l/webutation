"""Engine, session factory, and an Alembic-free schema bootstrap.

Migrations are deliberately out of scope. ``init_db`` issues ``create_all``,
which creates missing tables and does nothing to existing ones. It will not
alter a table whose definition has drifted, so during development the way to
pick up a schema change is to drop and recreate.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:dev@localhost:5434/webutation"
)


class Base(DeclarativeBase):
    """Declarative base shared by every model in :mod:`app.models`."""


engine = create_engine(DATABASE_URL, future=True)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(drop_first: bool = False) -> None:
    """Create the schema from the model metadata.

    Importing the models here rather than at module scope keeps ``models``
    free to import ``Base`` from this module without a circular import, while
    still guaranteeing every table is registered before ``create_all`` runs.
    """
    from app import models  # noqa: F401  (registers the tables on Base)

    if drop_first:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    # Running "python -m app.db" executes this file as __main__, while
    # app.models imports it again as app.db. Those are two distinct module
    # objects with two distinct Base classes, so calling the local init_db
    # here would run create_all against empty metadata and create nothing.
    # Importing through the package path gets the same objects the models
    # registered on.
    import sys

    from app.db import engine as pkg_engine, init_db as pkg_init_db

    # create_all leaves existing objects alone, and that includes enum types:
    # a status added to the model is not added to a type Postgres already has,
    # and the mismatch only shows up later as a failed insert. Recreating is
    # the documented way to pick up a schema change, so it gets a name here
    # rather than being retyped as one-off Python each time.
    drop_first = "--drop" in sys.argv[1:]
    if drop_first:
        print(f"Dropping and recreating every table in {DATABASE_URL}")

    pkg_init_db(drop_first=drop_first)
    with pkg_engine.connect() as conn:
        names = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        ).scalars().all()
    print(f"{len(names)} tables:", ", ".join(names) if names else "(none)")
