"""The deployed service runs with no Postgres and no DATABASE_URL.

Nothing in the HTTP request path opens a session: runs are held in memory by
app/report.py and are lost on restart. Provisioning a database for the
deployment cost money to serve nothing, so it was removed.

The persistence layer stays in the codebase and is still exercised locally
against docker-compose by tests/test_persistence.py. This file guards the other
half of that arrangement: that the app does not reach for a database it does not
have. It deliberately carries no skipif, because the case it describes is
precisely the one where no database is running.
"""

from __future__ import annotations

import os
import subprocess
import sys


def run_without_database(statements: str) -> subprocess.CompletedProcess:
    """Execute in a clean interpreter with DATABASE_URL removed.

    A subprocess rather than monkeypatching, because this process has usually
    imported app.db already — and may have built its engine — long before this
    test runs. Import-time behaviour can only be observed from a fresh import.
    """
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    return subprocess.run(
        [sys.executable, "-c", statements],
        capture_output=True,
        text=True,
        env=env,
    )


def test_importing_the_api_creates_no_engine() -> None:
    """app.api -> adapters -> app.models -> app.db used to reach create_engine.

    create_engine does not connect, so this was never a failed startup. It was a
    pointless one, and a pointless object pointing at a machine that does not
    exist is the kind of thing that later gets used by accident.
    """
    result = run_without_database(
        "import app.api\n"
        "import app.db as db\n"
        "assert db._engine is None, 'an engine was created on import'\n"
        "assert db._sessionmaker is None, 'a sessionmaker was built on import'\n"
        "print('no engine')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "no engine" in result.stdout


def test_the_app_serves_health_with_no_database_configured() -> None:
    """The health check Render polls must answer without Postgres."""
    result = run_without_database(
        "from fastapi.testclient import TestClient\n"
        "import app.api as api\n"
        "r = TestClient(api.app).get('/health')\n"
        "assert r.status_code == 200, r.status_code\n"
        "assert r.json()['status'] == 'ok'\n"
        "import app.db as db\n"
        "assert db._engine is None, 'health touched the database'\n"
        "print('health ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "health ok" in result.stdout


def test_a_search_runs_with_no_database_configured() -> None:
    """A fixture run exercises the whole pipeline and must not need Postgres."""
    result = run_without_database(
        "from fastapi.testclient import TestClient\n"
        "import app.api as api\n"
        "c = TestClient(api.app)\n"
        "r = c.post('/runs', json={'fixture': '01_same_name_two_cities'})\n"
        "assert r.status_code == 202, (r.status_code, r.text)\n"
        "import time\n"
        "run_id = r.json()['run_id']\n"
        "for _ in range(100):\n"
        "    body = c.get(f'/runs/{run_id}').json()\n"
        "    if body['status'] in ('complete', 'failed', 'error'):\n"
        "        break\n"
        "    time.sleep(0.1)\n"
        "assert body['status'] == 'complete', body.get('error')\n"
        "assert body['report']['candidates']\n"
        "import app.db as db\n"
        "assert db._engine is None, 'the request path opened a database session'\n"
        "print('run ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "run ok" in result.stdout


def test_the_persistence_layer_still_works_when_asked() -> None:
    """Lazy, not removed. Local persistence must keep functioning."""
    from app import db as db_module

    assert db_module.get_engine() is db_module.get_engine(), "cached, not rebuilt"
    assert db_module.engine is db_module.get_engine(), "attribute access still works"
    assert db_module.SessionLocal is db_module.get_sessionmaker()


def test_an_unknown_attribute_still_raises() -> None:
    """__getattr__ must not turn every typo into something that looks valid."""
    import pytest

    from app import db as db_module

    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        db_module.nonexistent
