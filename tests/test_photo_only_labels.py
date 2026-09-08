"""A photo-only run must not describe a comparison that never happened.

On a search with a photograph and nothing else, every candidate that is not the
anchor was labelled "Other person, same name" and "Held by: a shared name
only". No name was supplied, so no name was ever compared — and the rejection
list said so, reporting all 59 of them as decided on face similarity. The two
halves of the same report contradicted each other.

The grouping basis genuinely reads "name_only", which is where the wording came
from. But that describes how records were clustered against *each other*, by
their own name keys, and says nothing about the subject. Rendering it as a
shared name asserts a match to a name the caller never gave.
"""

from __future__ import annotations

import json

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates
from app.models import GroupingBasis, SourceOrigin
from app.report import _plain_basis, build_report


def record(ref: str, name: str, origin: str) -> list[AssertionDraft]:
    def draft(predicate: str, value: str) -> AssertionDraft:
        return AssertionDraft(
            predicate=predicate, raw_value=value,
            normalized_value=value.casefold(), record_ref=ref,
            publisher=origin, origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        )

    return [draft("name", name), draft("profile_url", f"https://{origin}/p")]


PHOTO_ONLY = Query(photo_url="https://x.test/subject.png")
NAMED = Query(name="Michael Petrie")


def photo_only_report():
    """The real shape: unrelated pages returned by a reverse image search."""
    drafts = [
        *record("r1", "Michael G Sondag CFP", "wealth.example"),
        *record("r2", "Dr. Jose A. Jaller, MD", "clinic.example"),
        *record("r3", "XING", "xing.example"),
    ]
    formation = form_candidates(drafts, subject=PHOTO_ONLY, record_context={})
    return build_report(
        subject=PHOTO_ONLY,
        runs=[],
        formation=formation,
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": len(drafts)},
    )


# --- the contradiction, stated directly ----------------------------------


def test_no_candidate_claims_a_shared_name_when_no_name_was_given() -> None:
    report = photo_only_report()
    assert report["candidates"], "the fixture must produce candidates"
    for candidate in report["candidates"]:
        assert candidate["held_by"] != "a shared name only"
        assert "shared name" not in candidate["held_by"]


def test_the_whole_report_never_says_the_name_matched() -> None:
    """Belt and braces: no wording anywhere may imply a name comparison."""
    report = photo_only_report()
    text = json.dumps(report).lower()
    for phrase in ("shared name", "same name", "sharing this name"):
        assert phrase not in text, f"{phrase!r} appears with no name supplied"


def test_the_report_says_which_comparisons_were_possible() -> None:
    report = photo_only_report()
    assert report["compared"] == {
        "name": False,
        "context": False,
        "locality": False,
        "face": True,
    }


def test_a_lone_record_says_it_was_matched_against_nothing() -> None:
    candidates = photo_only_report()["candidates"]
    assert all(
        c["held_by"] == "a single record, matched against nothing" for c in candidates
    ), [c["held_by"] for c in candidates]


# --- and the named case is untouched -------------------------------------


def test_a_named_search_still_reports_a_shared_name() -> None:
    """The wording is only wrong when the comparison did not happen."""
    drafts = [
        *record("r1", "Michael Petrie", "a.example"),
        *record("r2", "Michael Petrie", "b.example"),
    ]
    formation = form_candidates(drafts, subject=NAMED, record_context={})
    report = build_report(
        subject=NAMED, runs=[], formation=formation,
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": 4},
    )

    assert report["compared"]["name"] is True
    assert any("shared name" in c["held_by"] for c in report["candidates"])


def test_the_basis_helper_needs_the_subject_to_know() -> None:
    """Called without a subject it cannot tell, and must not guess a name."""
    from app.engine.candidates import CandidateDraft

    candidate = CandidateDraft(
        name_key="somebody else",
        grouping_basis=GroupingBasis.name_only,
        assertions=record("r1", "Somebody Else", "a.example"),
    )
    assert _plain_basis(candidate, PHOTO_ONLY) != "a shared name only"
    assert _plain_basis(candidate, NAMED) == "a shared name only"
