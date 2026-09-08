"""The report must not die on a name it cannot spell.

Results come from the whole web. Google Lens in particular returns pages in
every script there is, and a Windows console defaults to a codepage that can
encode almost none of them. A report that crashes halfway through printing
candidates loses everything after the first non-Latin name, which is exactly
what happened on the first run wide enough to include one.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_subject.py"

#: Scripts a cp1252 console cannot represent, plus one it can, so the test
#: fails for the right reason rather than because everything is exotic.
AWKWARD_NAMES = [
    "マイケル・ペトリー",          # Japanese
    "Михаил Петров",              # Cyrillic
    "米高·彼德里",                 # Chinese
    "Μιχάλης Πέτρου",             # Greek
    "Michaël Pétrie",             # Latin with diacritics
    "مايكل بيتري",                # Arabic
    "Michael Petrie 🕵",           # emoji
]


def load_script():
    """Import scripts/run_subject.py, which is a script rather than a module."""
    spec = importlib.util.spec_from_file_location("run_subject_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_forces_utf8_on_stdout() -> None:
    """The fix itself: whatever the console codepage is, output is UTF-8."""
    load_script()
    assert (sys.stdout.encoding or "").lower().replace("-", "") == "utf8"


@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_a_non_latin_name_renders_without_raising(name, capsys) -> None:
    from app.adapters.base import AssertionDraft, Query
    from app.engine.candidates import CandidateDraft

    module = load_script()
    draft = AssertionDraft(
        predicate="name",
        raw_value=name,
        normalized_value=name.casefold(),
        record_ref="r1",
        publisher="example.com",
        origin_key="example.com",
    )
    candidate = CandidateDraft(name_key=name.casefold(), assertions=[draft])

    # The whole per-candidate render, not just a print of the name.
    module.print_candidate(1, candidate, Query(name="Michael Petrie"), 1)
    module.print_unattached([draft])

    assert name.casefold()[:6] in capsys.readouterr().out


def test_clipping_does_not_split_or_drop_characters() -> None:
    module = load_script()
    assert module.clip("マイケル・ペトリー", 6) == "マイケ..."
    assert module.clip("Michael", 20) == "Michael"
    assert module.clip(None, 10) == "-"


def test_the_report_survives_a_cp1252_console() -> None:
    """Reproduces the original crash by forcing the Windows console codepage.

    Run in a subprocess because the encoding has to be wrong *before* the
    script reconfigures it, which is the exact ordering the bug depended on.
    Without the fix this exits non-zero with UnicodeEncodeError.
    """
    snippet = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('rs', r'{SCRIPT}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "from app.adapters.base import AssertionDraft, Query\n"
        "from app.engine.candidates import CandidateDraft\n"
        "name = 'マイケル・ペトリー'\n"
        "d = AssertionDraft(predicate='name', raw_value=name, "
        "normalized_value=name, record_ref='r1', origin_key='example.com')\n"
        "c = CandidateDraft(name_key=name, assertions=[d])\n"
        "m.print_candidate(1, c, Query(name='Michael Petrie'), 1)\n"
        "print('RENDERED OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
        env={
            **_clean_env(),
            "PYTHONIOENCODING": "cp1252",
            "PYTHONUTF8": "0",
        },
        timeout=180,
    )
    assert result.returncode == 0, (
        f"the report crashed on a non-Latin name:\n{result.stderr[-1500:]}"
    )
    assert "RENDERED OK" in result.stdout
    assert "UnicodeEncodeError" not in result.stderr


def _clean_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    return env
