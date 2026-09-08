"""Conflict detection: what counts as a conflict, and what does not.

Most of these tests assert that nothing is emitted. That is the point. A system
that flags every disagreement teaches reviewers to ignore it, and the plan is
explicit that a move, a second job, and a silent source are not conflicts.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timezone

from app.adapters.base import AssertionDraft
from app.engine.candidates import CandidateDraft
from app.engine.conflicts import (
    ConflictConfig,
    Exclusivity,
    detect_conflicts,
)
from app.models import ConflictReason, ConflictStatus

_REF_SEQUENCE = itertools.count(1)


def when(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def claim(
    predicate: str,
    value: str,
    *,
    record_ref: str | None = None,
    origin: str = "origin_a",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> AssertionDraft:
    return AssertionDraft(
        predicate=predicate,
        raw_value=value,
        normalized_value=value.casefold(),
        record_ref=record_ref or f"rec-{next(_REF_SEQUENCE)}",
        origin_key=origin,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def candidate_of(
    *assertions: AssertionDraft,
    name: str = "michael petrie",
    localities: set[str | None] | None = None,
    merge_evidence: list[str] | None = None,
) -> CandidateDraft:
    return CandidateDraft(
        name_key=name,
        locality_keys=localities if localities is not None else {"austin tx"},
        assertions=list(assertions),
        merge_evidence=merge_evidence or [],
    )


# --- genuine conflicts -----------------------------------------------------


def test_two_birth_dates_conflict() -> None:
    """Exclusive predicate, incompatible values. Time does not enter into it."""
    a = claim("birth_date", "1979-04-02")
    b = claim("birth_date", "1981-11-30")

    (conflict,) = detect_conflicts(candidate_of(a, b))

    assert conflict.predicate == "birth_date"
    assert conflict.reason is ConflictReason.incompatible_normalized_values
    assert conflict.status is ConflictStatus.unresolved
    assert conflict.potential is False
    assert set(conflict.values) == {"1979-04-02", "1981-11-30"}
    assert conflict.assertions == [a, b]


def test_government_identifiers_are_exclusive() -> None:
    a = claim("passport_number", "X1234567")
    b = claim("passport_number", "Y7654321")
    (conflict,) = detect_conflicts(candidate_of(a, b))
    assert conflict.reason is ConflictReason.incompatible_normalized_values


def test_addresses_with_overlapping_validity_conflict() -> None:
    """Two homes over one period is a contradiction, not a history."""
    a = claim("address", "1204 W 6th St", valid_from=when(2018), valid_to=when(2024))
    b = claim("address", "88 Beacon St", valid_from=when(2020), valid_to=when(2026))

    (conflict,) = detect_conflicts(candidate_of(a, b))

    assert conflict.potential is False
    assert "overlapping validity" in conflict.basis
    # No merge happened, so the facts are what disagree.
    assert conflict.reason is ConflictReason.incompatible_normalized_values


# --- disagreements that are not conflicts ---------------------------------


def test_two_employers_do_not_conflict() -> None:
    """Employment is multivalued. Simultaneous employment is possible."""
    a = claim("employer", "Meridian Freight")
    b = claim("employer", "Acme Logistics")

    assert detect_conflicts(candidate_of(a, b)) == []


def test_two_job_titles_do_not_conflict() -> None:
    a = claim("job_title", "Operations Manager")
    b = claim("job_title", "Consultant")
    assert detect_conflicts(candidate_of(a, b)) == []


def test_addresses_over_non_overlapping_periods_are_a_historical_change() -> None:
    """The person moved. Both claims were true, at different times."""
    a = claim("address", "1204 W 6th St", valid_from=when(2010), valid_to=when(2014))
    b = claim("address", "88 Beacon St", valid_from=when(2015), valid_to=when(2020))

    assert detect_conflicts(candidate_of(a, b)) == []


def test_one_source_silent_is_missing_evidence_not_conflict() -> None:
    """Silence is not disagreement."""
    stated = claim("birth_date", "1979-04-02")
    unrelated = claim("employer", "Meridian Freight")

    assert detect_conflicts(candidate_of(stated, unrelated)) == []


def test_two_spellings_of_one_date_are_not_a_conflict() -> None:
    """Comparison is on normalized values, never raw ones."""
    a = AssertionDraft(
        predicate="birth_date",
        raw_value="April 2, 1979",
        normalized_value="1979-04-02",
        record_ref="rec-a",
    )
    b = AssertionDraft(
        predicate="birth_date",
        raw_value="04/02/1979",
        normalized_value="1979-04-02",
        record_ref="rec-b",
    )

    assert a.raw_value != b.raw_value
    assert detect_conflicts(candidate_of(a, b)) == []


def test_an_unnormalized_value_cannot_contradict_anything() -> None:
    """A value that could not be parsed is never compared against a real one."""
    parsed = claim("birth_date", "1979-04-02")
    unparsed = AssertionDraft(
        predicate="birth_date",
        raw_value="sometime in the seventies",
        normalized_value=None,
        record_ref="rec-x",
    )

    assert detect_conflicts(candidate_of(parsed, unparsed)) == []


def test_agreement_across_many_sources_is_not_a_conflict() -> None:
    same = [claim("birth_date", "1979-04-02", origin=f"o{i}") for i in range(5)]
    assert detect_conflicts(candidate_of(*same)) == []


def test_no_assertions_returns_empty_without_raising() -> None:
    assert detect_conflicts(candidate_of()) == []


# --- unknown validity is flagged, not asserted ----------------------------


def test_addresses_with_unknown_validity_flag_as_potential() -> None:
    """Overlap could not be shown, so the conflict is not asserted."""
    a = claim("address", "1204 W 6th St")
    b = claim("address", "88 Beacon St")

    (conflict,) = detect_conflicts(candidate_of(a, b))

    assert conflict.potential is True
    assert conflict.status is ConflictStatus.unresolved
    assert "unknown" in conflict.basis
    assert conflict.reason is ConflictReason.incompatible_normalized_values


def test_one_sided_validity_is_still_unknown() -> None:
    """Half a period on one side establishes nothing about overlap."""
    a = claim("address", "1204 W 6th St", valid_from=when(2010), valid_to=when(2014))
    b = claim("address", "88 Beacon St")

    (conflict,) = detect_conflicts(candidate_of(a, b))
    assert conflict.potential is True


def test_open_ended_periods_overlap() -> None:
    """Two sources each saying "still current" do contradict each other."""
    a = claim("address", "1204 W 6th St", valid_from=when(2018))
    b = claim("address", "88 Beacon St", valid_from=when(2020))

    (conflict,) = detect_conflicts(candidate_of(a, b))
    assert conflict.potential is False


def test_exclusive_predicates_ignore_validity_entirely() -> None:
    """A birth date is not something a person has a series of."""
    a = claim("birth_date", "1979-04-02", valid_from=when(2010), valid_to=when(2014))
    b = claim("birth_date", "1981-11-30", valid_from=when(2015), valid_to=when(2020))

    (conflict,) = detect_conflicts(candidate_of(a, b))
    assert conflict.potential is False
    assert conflict.reason is ConflictReason.incompatible_normalized_values


# --- questioning the merge -------------------------------------------------


def test_conflict_on_weakly_merged_records_emits_possible_bad_merge() -> None:
    """The merge may be wrong, not the facts.

    Two records in different blocking keys, joined by a face-distance merge
    rather than a shared distinctive attribute, then disagreeing on an
    exclusive predicate. That pattern is what an over-merged candidate looks
    like from the inside.
    """
    austin = [
        claim("name", "Michael Petrie", record_ref="rec-austin"),
        claim("city", "Austin, TX", record_ref="rec-austin"),
        claim("birth_date", "1979-04-02", record_ref="rec-austin"),
    ]
    boston = [
        claim("name", "Michael Petrie", record_ref="rec-boston"),
        claim("city", "Boston, MA", record_ref="rec-boston"),
        claim("birth_date", "1981-11-30", record_ref="rec-boston"),
    ]
    candidate = candidate_of(
        *austin,
        *boston,
        localities={"austin tx", "boston ma"},
        merge_evidence=["face distance 0.310 below 0.38"],
    )

    conflicts = detect_conflicts(candidate)
    birth = [c for c in conflicts if c.predicate == "birth_date"]

    assert len(birth) == 1
    assert birth[0].reason is ConflictReason.possible_bad_merge
    assert "merge may be wrong" in birth[0].basis
    assert birth[0].record_refs == {"rec-austin", "rec-boston"}


def test_a_shared_distinctive_attribute_keeps_the_blame_on_the_facts() -> None:
    """A shared email is strong enough that the disagreement is about the data."""
    austin = [
        claim("name", "Michael Petrie", record_ref="rec-austin"),
        claim("city", "Austin, TX", record_ref="rec-austin"),
        claim("email", "mp@example.com", record_ref="rec-austin"),
        claim("birth_date", "1979-04-02", record_ref="rec-austin"),
    ]
    boston = [
        claim("name", "Michael Petrie", record_ref="rec-boston"),
        claim("city", "Boston, MA", record_ref="rec-boston"),
        claim("email", "mp@example.com", record_ref="rec-boston"),
        claim("birth_date", "1981-11-30", record_ref="rec-boston"),
    ]
    candidate = candidate_of(
        *austin,
        *boston,
        localities={"austin tx", "boston ma"},
        merge_evidence=["shared distinctive attribute email='mp@example.com'"],
    )

    birth = [c for c in detect_conflicts(candidate) if c.predicate == "birth_date"]
    assert len(birth) == 1
    assert birth[0].reason is ConflictReason.incompatible_normalized_values


def test_conflict_within_one_block_is_about_the_facts() -> None:
    """Records that agreed on name and locality were never merged."""
    a = [
        claim("name", "Michael Petrie", record_ref="rec-1"),
        claim("city", "Austin, TX", record_ref="rec-1"),
        claim("birth_date", "1979-04-02", record_ref="rec-1"),
    ]
    b = [
        claim("name", "Michael Petrie", record_ref="rec-2"),
        claim("city", "Austin, TX", record_ref="rec-2"),
        claim("birth_date", "1981-11-30", record_ref="rec-2"),
    ]

    birth = [
        c for c in detect_conflicts(candidate_of(*a, *b)) if c.predicate == "birth_date"
    ]
    assert len(birth) == 1
    assert birth[0].reason is ConflictReason.incompatible_normalized_values


# --- nothing is resolved ---------------------------------------------------


def test_preferred_assertion_is_never_set_by_detection() -> None:
    a = claim("birth_date", "1979-04-02")
    b = claim("birth_date", "1981-11-30")

    (conflict,) = detect_conflicts(candidate_of(a, b))

    assert conflict.preferred_assertion is None
    assert conflict.status is ConflictStatus.unresolved


def test_conflicting_assertions_remain_untouched_and_queryable() -> None:
    a = claim("birth_date", "1979-04-02")
    b = claim("birth_date", "1981-11-30")
    candidate = candidate_of(a, b)
    before = list(candidate.assertions)

    (conflict,) = detect_conflicts(candidate)

    # Both survive, unmodified, still attached to the candidate.
    assert candidate.assertions == before
    assert a.normalized_value == "1979-04-02"
    assert b.normalized_value == "1981-11-30"
    assert set(conflict.assertions) <= set(candidate.assertions)


# --- configuration ---------------------------------------------------------


def test_exclusivity_is_configurable_per_predicate() -> None:
    """Marking employer exclusive is a config change, not a code change."""
    a = claim("employer", "Meridian Freight")
    b = claim("employer", "Acme Logistics")
    candidate = candidate_of(a, b)

    assert detect_conflicts(candidate) == []

    strict = ConflictConfig(exclusivity={"employer": Exclusivity.exclusive})
    (conflict,) = detect_conflicts(candidate, strict)
    assert conflict.predicate == "employer"


def test_unlisted_predicates_are_multivalued_by_default() -> None:
    a = claim("hobby", "sailing")
    b = claim("hobby", "chess")
    assert detect_conflicts(candidate_of(a, b)) == []


def test_face_merges_can_be_declared_strong() -> None:
    """The default questions a face merge; config can decide otherwise."""
    austin = [
        claim("name", "Michael Petrie", record_ref="rec-austin"),
        claim("city", "Austin, TX", record_ref="rec-austin"),
        claim("birth_date", "1979-04-02", record_ref="rec-austin"),
    ]
    boston = [
        claim("name", "Michael Petrie", record_ref="rec-boston"),
        claim("city", "Boston, MA", record_ref="rec-boston"),
        claim("birth_date", "1981-11-30", record_ref="rec-boston"),
    ]
    candidate = candidate_of(
        *austin,
        *boston,
        localities={"austin tx", "boston ma"},
        merge_evidence=["face distance 0.310 below 0.38"],
    )

    trusting = ConflictConfig(
        strong_merge_markers=("shared distinctive attribute", "face distance")
    )
    birth = [
        c for c in detect_conflicts(candidate, trusting) if c.predicate == "birth_date"
    ]
    assert birth[0].reason is ConflictReason.incompatible_normalized_values


# --- the clean-match fixture has nothing to report ------------------------


def test_fixture_06_produces_no_conflicts() -> None:
    from app.adapters.base import Query
    from app.adapters.fixture import FixtureAdapter
    from app.engine.candidates import form_candidates

    adapter = FixtureAdapter("06_clean_match")
    drafts = adapter.normalize(adapter.collect(Query(name="Marcus Webb")))
    candidate = form_candidates(drafts).candidates[0]

    assert detect_conflicts(candidate) == []


# --- a merge must exist before it can be blamed ---------------------------


def test_an_unmerged_candidate_never_blames_the_merge() -> None:
    """No merge happened, so there is nothing to question.

    Locality predicates form part of the blocking key, so two records that
    disagree about a city always land in different blocks. Without requiring a
    real merge, that disagreement would be used to convict a merge that never
    took place.
    """
    a = claim("city", "Austin, TX", record_ref="rec-1")
    b = claim("city", "Boston, MA", record_ref="rec-2")

    candidate = candidate_of(a, b, merge_evidence=[])
    assert candidate.is_merged is False

    (conflict,) = detect_conflicts(candidate)
    assert conflict.reason is ConflictReason.incompatible_normalized_values


def test_the_same_disagreement_blames_the_merge_once_one_exists() -> None:
    """Identical claims, differing only in whether the candidate was merged."""
    def build(merge_evidence: list[str]) -> CandidateDraft:
        return CandidateDraft(
            name_key="michael petrie",
            locality_keys={"austin tx", "boston ma"},
            blocking_keys={
                ("michael petrie", "austin tx"),
                ("michael petrie", "boston ma"),
            },
            merge_evidence=merge_evidence,
            assertions=[
                claim("name", "Michael Petrie", record_ref="rec-austin"),
                claim("city", "Austin, TX", record_ref="rec-austin"),
                claim("birth_date", "1979-04-02", record_ref="rec-austin"),
                claim("name", "Michael Petrie", record_ref="rec-boston"),
                claim("city", "Boston, MA", record_ref="rec-boston"),
                claim("birth_date", "1981-11-30", record_ref="rec-boston"),
            ],
        )

    merged = build(["face distance 0.310 below 0.38"])
    birth = [c for c in detect_conflicts(merged) if c.predicate == "birth_date"]
    assert birth[0].reason is ConflictReason.possible_bad_merge
