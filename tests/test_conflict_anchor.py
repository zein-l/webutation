"""Anchor-grouped conflicts name the anchor, not a merge.

An anchor candidate spans several blocking keys, so ``is_merged`` is true even
though nothing was merged: each record matched the subject on its own and the
records were never compared with each other. Reporting that as
``possible_bad_merge`` sends a reviewer to inspect a merge that never happened.
"""

from __future__ import annotations

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import CandidateDraft, form_candidates
from app.engine.conflicts import detect_conflicts
from app.models import ConflictReason

SUBJECT = Query(name="Michael Petrie", context="logistics operations Austin")


def claim(predicate: str, value: str, ref: str) -> AssertionDraft:
    return AssertionDraft(
        predicate=predicate,
        raw_value=value,
        normalized_value=value.casefold(),
        record_ref=ref,
        publisher=ref,
        origin_key=ref,
    )


def two_records_disagreeing_on_birth_date() -> list[AssertionDraft]:
    """Same name, different cities, incompatible birth dates."""
    return [
        claim("name", "Michael Petrie", "rec-austin"),
        claim("city", "Austin, TX", "rec-austin"),
        claim("birth_date", "1979-04-02", "rec-austin"),
        claim("name", "Michael Petrie", "rec-boston"),
        claim("city", "Boston, MA", "rec-boston"),
        claim("birth_date", "1981-11-30", "rec-boston"),
    ]


def birth_conflicts(candidate: CandidateDraft):
    return [c for c in detect_conflicts(candidate) if c.predicate == "birth_date"]


# --- the fix ---------------------------------------------------------------


def test_an_anchor_grouped_conflict_reports_the_anchor_reason() -> None:
    """Built through real formation, so the anchor candidate is genuine."""
    drafts = two_records_disagreeing_on_birth_date()
    context = {
        "rec-austin": "Operations lead in logistics, based in Austin.",
        "rec-boston": "Logistics operations manager in Austin.",
    }

    formation = form_candidates(drafts, subject=SUBJECT, record_context=context)
    anchor = formation.anchor_candidate
    assert anchor is not None
    assert set(anchor.anchor_matches) == {"rec-austin", "rec-boston"}

    # The condition that used to produce the wrong wording.
    assert anchor.is_merged is True
    assert anchor.merge_evidence == []

    (conflict,) = birth_conflicts(anchor)
    assert conflict.reason is ConflictReason.possible_bad_anchor
    assert conflict.reason is not ConflictReason.possible_bad_merge
    assert "anchor may be wrong" in conflict.basis
    assert "merge" not in conflict.basis.replace("merged", "")


def test_a_merged_candidate_still_reports_the_merge_reason() -> None:
    """The existing behaviour is untouched for candidates that really merged."""
    candidate = CandidateDraft(
        name_key="michael petrie",
        locality_keys={"austin tx", "boston ma"},
        blocking_keys={
            ("michael petrie", "austin tx"),
            ("michael petrie", "boston ma"),
        },
        merge_evidence=["face distance 0.310 below 0.38"],
        assertions=two_records_disagreeing_on_birth_date(),
    )

    (conflict,) = birth_conflicts(candidate)
    assert conflict.reason is ConflictReason.possible_bad_merge
    assert "merge may be wrong" in conflict.basis


def test_the_two_reasons_are_distinguishable_on_identical_evidence() -> None:
    """Same claims, same disagreement. Only the grouping differs."""
    drafts = two_records_disagreeing_on_birth_date()
    shape = {
        "name_key": "michael petrie",
        "locality_keys": {"austin tx", "boston ma"},
        "blocking_keys": {
            ("michael petrie", "austin tx"),
            ("michael petrie", "boston ma"),
        },
    }

    anchored = CandidateDraft(
        **shape,
        is_anchor=True,
        anchor_matches={"rec-austin": None, "rec-boston": None},
        assertions=list(drafts),
    )
    merged = CandidateDraft(
        **shape,
        merge_evidence=["face distance 0.310 below 0.38"],
        assertions=list(drafts),
    )

    assert birth_conflicts(anchored)[0].reason is ConflictReason.possible_bad_anchor
    assert birth_conflicts(merged)[0].reason is ConflictReason.possible_bad_merge


# --- the boundaries hold ---------------------------------------------------


def test_a_shared_distinctive_attribute_still_beats_both_doubts() -> None:
    """Strong positive evidence means the facts disagree, not the grouping."""
    drafts = [
        *two_records_disagreeing_on_birth_date(),
        claim("email", "mp@example.com", "rec-austin"),
        claim("email", "mp@example.com", "rec-boston"),
    ]
    candidate = CandidateDraft(
        name_key="michael petrie",
        locality_keys={"austin tx", "boston ma"},
        blocking_keys={
            ("michael petrie", "austin tx"),
            ("michael petrie", "boston ma"),
        },
        is_anchor=True,
        anchor_matches={"rec-austin": None, "rec-boston": None},
        assertions=drafts,
    )

    (conflict,) = birth_conflicts(candidate)
    assert conflict.reason is ConflictReason.incompatible_normalized_values


def test_an_unanchored_unmerged_candidate_blames_neither() -> None:
    candidate = CandidateDraft(
        name_key="michael petrie",
        locality_keys={"austin tx", "boston ma"},
        assertions=two_records_disagreeing_on_birth_date(),
    )
    (conflict,) = birth_conflicts(candidate)
    assert conflict.reason is ConflictReason.incompatible_normalized_values


def test_a_record_attached_to_an_anchor_candidate_by_a_merge_blames_the_merge() -> None:
    """Anchoring takes the blame only for records anchoring actually gathered."""
    candidate = CandidateDraft(
        name_key="michael petrie",
        locality_keys={"austin tx", "boston ma"},
        blocking_keys={
            ("michael petrie", "austin tx"),
            ("michael petrie", "boston ma"),
        },
        is_anchor=True,
        # Only one of the two records was anchored; a merge brought the other.
        anchor_matches={"rec-austin": None},
        merge_evidence=["face distance 0.310 below 0.38"],
        assertions=two_records_disagreeing_on_birth_date(),
    )

    (conflict,) = birth_conflicts(candidate)
    assert conflict.reason is ConflictReason.possible_bad_merge


def test_the_enum_carries_both_reasons() -> None:
    assert ConflictReason.possible_bad_anchor.value == "possible_bad_anchor"
    assert {r.value for r in ConflictReason} == {
        "incompatible_normalized_values",
        "possible_bad_merge",
        "possible_bad_anchor",
    }
