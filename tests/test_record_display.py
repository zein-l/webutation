"""A record's internal identity is not a thing to show a reader.

Results the source returned without a URL are identified by a hash of their own
bytes — "serpapi:google:sha256:0f085f7e010adeb4:10". That is a sound identity
and an unreadable thing to print in a report: it looks like a bug rather than
the fact it represents, which is that the source supplied no link.
"""

from __future__ import annotations

import json

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates
from app.models import SourceOrigin
from app.report import build_report, record_source

# Verbatim from a live run.
HASH_REF = "serpapi:google:sha256:0f085f7e010adeb4:10"
URL_REF = "serpapi:google:https://www.linkedin.com/in/michaelpetrie"


def test_a_url_reference_shows_the_url() -> None:
    source = record_source(URL_REF, "Michael Petrie - Webutation, Inc")
    assert source["url"] == "https://www.linkedin.com/in/michaelpetrie"
    assert source["note"] is None


def test_a_hash_reference_shows_the_title_and_says_why() -> None:
    source = record_source(HASH_REF, "Michael Petrie joins WMCO management team")
    assert source["url"] is None
    assert source["title"] == "Michael Petrie joins WMCO management team"
    assert source["note"] == "no link supplied by the source"


def test_a_hash_reference_with_no_title_still_hides_the_hash() -> None:
    source = record_source(HASH_REF, None)
    assert source["url"] is None
    assert source["title"] is None
    assert "sha256" not in json.dumps(source)


@pytest.mark.parametrize("ref", [HASH_REF, URL_REF])
def test_the_hash_never_appears_in_what_is_shown(ref: str) -> None:
    assert "sha256" not in json.dumps(record_source(ref, "a title"))


# --- through a whole report ---------------------------------------------


def linkless_record(ref: str, title: str) -> list[AssertionDraft]:
    """A result with a name and no profile_url — the case that hashes."""
    return [
        AssertionDraft(
            predicate="name", raw_value=title, normalized_value=title.casefold(),
            record_ref=ref, publisher="news.example", origin_key="news.example",
            source_origin=SourceOrigin.known_origin,
        )
    ]


def test_every_rejected_record_can_be_shown_without_a_hash() -> None:
    drafts = [
        *linkless_record("serpapi:google:sha256:0f085f7e010adeb4:10",
                         "Michael Petrie joins WMCO management team"),
        *linkless_record("serpapi:google:sha256:1a2b3c4d5e6f7081:11",
                         "Meet the PPG Team"),
    ]
    subject = Query(name="Michael Petrie", context="Webutation")
    report = build_report(
        subject=subject, runs=[],
        formation=form_candidates(drafts, subject=subject, record_context={}),
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": len(drafts)},
    )

    assert report["rejected"], "the fixture must produce rejections"
    for entry in report["rejected"]:
        source = entry["source"]
        assert source["url"] is None
        assert source["title"], "a linkless record must still say what it was"
        assert source["note"] == "no link supplied by the source"
        assert "sha256" not in json.dumps(source)


def test_a_linked_record_keeps_its_url_in_the_report() -> None:
    drafts = [
        AssertionDraft(
            predicate="name", raw_value="Michael Petrie",
            normalized_value="michael petrie",
            record_ref="serpapi:google:https://a.example/p",
            publisher="a.example", origin_key="a.example",
            source_origin=SourceOrigin.known_origin,
        )
    ]
    subject = Query(name="Michael Petrie", context="Webutation")
    report = build_report(
        subject=subject, runs=[],
        formation=form_candidates(drafts, subject=subject, record_context={}),
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": 1},
    )
    for entry in report["rejected"]:
        assert entry["source"]["url"] == "https://a.example/p"


# --- the ordering the section claims -------------------------------------


def test_the_two_rejection_groups_stay_separate_and_sorted() -> None:
    """A 1.00 refused on substance is not "closer" than a 0.40 refused on score.

    They were shown as one list ordered "closest first", which put entries at
    0.00 above entries at 1.00. Within each group the order is by score; the
    groups are not ranked against each other, and the view labels them.
    """
    drafts = []
    # Name-only records: high strength, refused on substance.
    for i in range(2):
        drafts += [
            AssertionDraft(
                predicate="name", raw_value="Michael Petrie",
                normalized_value="michael petrie", record_ref=f"serpapi:google:https://n{i}.example/p",
                publisher=f"n{i}.example", origin_key=f"n{i}.example",
                source_origin=SourceOrigin.known_origin,
            )
        ]
    # A record that simply scores low.
    drafts += [
        AssertionDraft(
            predicate="name", raw_value="Someone Else",
            normalized_value="someone else", record_ref="serpapi:google:https://low.example/p",
            publisher="low.example", origin_key="low.example",
            source_origin=SourceOrigin.known_origin,
        )
    ]
    subject = Query(name="Michael Petrie", context="Webutation")
    report = build_report(
        subject=subject, runs=[],
        formation=form_candidates(drafts, subject=subject, record_context={}),
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": len(drafts)},
    )

    rejected = report["rejected"]
    kinds = [r["decided_by"] for r in rejected]
    assert kinds == sorted(kinds, key=lambda k: k == "substance"), (
        "threshold rejections must precede substance ones"
    )
    for kind in ("threshold", "substance"):
        group = [r["strength"] for r in rejected if r["decided_by"] == kind]
        assert group == sorted(group, reverse=True), f"{kind} group must sort by score"
