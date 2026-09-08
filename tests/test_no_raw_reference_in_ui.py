"""The view must not print an internal record reference, ever.

A record the source returned without a URL is identified by a hash of its own
bytes. The server sends a `source` block that has already turned that into
something readable, and the first version of the view fell back to printing the
raw reference when the block was absent.

That fallback was not theoretical. A page left open across a deploy holds an
older report in memory and renders it through today's component, and every
rejected entry in a pre-fix report lacks `source` — 82 of 82 in the run this
was found on. One stale tab was enough to bring the hashes back.

These are source-level assertions, which is a weaker thing than a rendering
test. There is no JavaScript test runner in this project, and a weak guard on
the exact regression that happened is worth more than none.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCE = Path("frontend/src/components/Source.jsx")
REJECTED = Path("frontend/src/components/Rejected.jsx")
CANDIDATE = Path("frontend/src/components/Candidate.jsx")


def test_the_component_exists() -> None:
    assert SOURCE.exists(), "the view that renders a record's origin"


def test_it_never_renders_the_bare_reference() -> None:
    """The defect, stated exactly: {fallbackRef} must not be rendered."""
    body = SOURCE.read_text(encoding="utf-8")
    rendered = re.findall(r"\{\s*fallbackRef\s*\}", body)
    assert not rendered, (
        "Source.jsx renders fallbackRef directly; a report without a source "
        "block would print a raw sha256 identity"
    )


def test_it_derives_the_same_rule_the_server_uses() -> None:
    """Absent a source block it must apply the rule, not print the reference."""
    body = SOURCE.read_text(encoding="utf-8")
    assert "sha256:" in body, "the hashed-reference case must be recognised"
    assert "no link supplied by the source" in body


@pytest.mark.parametrize("path", [REJECTED, CANDIDATE])
def test_no_view_prints_a_record_reference_itself(path: Path) -> None:
    """Every view goes through Source rather than rendering the ref.

    Only rendered children count. ``key={r.record_ref}`` is React identity and
    ``fallbackRef={r.record_ref}`` is a prop Source no longer prints; neither
    puts anything on screen, and a check that flagged them would fail on
    correct code.
    """
    body = path.read_text(encoding="utf-8")
    # A JSX expression rendered as content is never preceded by "=", which
    # makes it an attribute value, nor by "$", which makes it interpolation
    # inside a template literal — as in key={`${a.record_ref}-...`}.
    offenders = [
        m.group(0)
        for m in re.finditer(r"(?<![=\w$]) *\{\s*\w+\.record_ref\s*\}", body)
    ]
    assert not offenders, f"{path.name} renders a record_ref as content: {offenders}"


def test_the_server_rule_and_the_view_rule_agree() -> None:
    """Both must treat a sha256 reference as having no URL."""
    from app.report import record_source

    hashed = record_source("serpapi:google:sha256:8ef28373476b00e7:20", None)
    assert hashed["url"] is None
    assert hashed["note"] == "no link supplied by the source"

    linked = record_source("serpapi:google:https://a.example/p", None)
    assert linked["url"] == "https://a.example/p"
    assert linked["note"] is None
