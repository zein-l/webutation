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

#: Where Postgres lives when nothing says otherwise: the docker-compose service
#: in this repository, which publishes on 5434 because a pre-existing PostgreSQL
#: owns 5432 on the development machine.
DEFAULT_DATABASE_URL = "postgresql://postgres:dev@localhost:5434/webutation"

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    """Declarative base shared by every model in :mod:`app.models`."""


_engine = None
_sessionmaker = None


def get_engine():
    """The engine, created on first use rather than on import.

    ``app.models`` imports this module for ``Base``, and the scoring engine
    imports ``app.models`` for its enums, so importing the HTTP app used to
    reach ``create_engine`` before a single request had been served. Nothing in
    the request path opens a session — runs are held in memory — so the deployed
    service has no database at all, and building an engine for it meant
    resolving a URL that is not there and loading a driver nobody calls.

    ``create_engine`` does not connect, so this was never a failed startup. It
    was a pointless one, and a pointless object pointing at a machine that does
    not exist is the kind of thing that later gets used by accident.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, future=True)
    return _engine


def get_sessionmaker():
    """The session factory, bound to the lazily created engine."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(
            bind=get_engine(), expire_on_commit=False, future=True
        )
    return _sessionmaker


def __getattr__(name: str):
    """Keep ``from app.db import engine`` working, without doing it on import.

    PEP 562 module-level attribute access: the name resolves the first time it
    is read, which for the persistence layer and its tests is exactly when they
    are about to use it, and for the HTTP service is never.
    """
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_sessionmaker()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error."""
    session = get_sessionmaker()()
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

    # get_engine(), not a bare ``engine``: module-level __getattr__ resolves
    # attribute access on the module from outside, not a global lookup from a
    # function inside it, so the bare name would be a NameError here.
    active = get_engine()
    if drop_first:
        Base.metadata.drop_all(active)
    Base.metadata.create_all(active)


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
