"""Candidate formation: grouping and attachment.

The cases here are the ones the plan calls out as load-bearing. Two people
sharing a name must stay two people; positive evidence must be required to
merge them; a draft that matches nothing must be reported rather than forced
somewhere; and zero candidates must be a successful outcome.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timezone

from app.adapters.base import AssertionDraft
from app.engine.candidates import (
    CandidateDraft,
    FormationConfig,
    form_candidates,
    normalize_locality_key,
    normalize_name_key,
)

JAN = datetime(2024, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2024, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2024, 3, 1, tzinfo=timezone.utc)

#: One record_ref per record() call.
_REF_SEQUENCE = itertools.count(1)


def record(
    publisher: str,
    origin: str,
    observed_at: datetime,
    **fields: str,
) -> list[AssertionDraft]:
    """Drafts as one adapter result would emit them.

    Every draft carries the same record_ref, which is what tells formation
    these claims describe one person. Each call gets a fresh reference, because
    one call represents one source record, and two calls must stay two records
    even when they share a publisher and an origin.
    """
    ref = f"{publisher}/{origin}/{next(_REF_SEQUENCE)}"
    return [
        AssertionDraft(
            predicate=predicate,
            raw_value=value,
            normalized_value=value.casefold(),
            record_ref=ref,
            publisher=publisher,
            origin_key=origin,
            observed_at=observed_at,
        )
        for predicate, value in fields.items()
    ]


def unlinked(**fields: str) -> list[AssertionDraft]:
    """Drafts with no record_ref, each standing as its own record."""
    return [
        AssertionDraft(
            predicate=predicate,
            raw_value=value,
            normalized_value=value.casefold(),
        )
        for predicate, value in fields.items()
    ]


# --- key normalization -----------------------------------------------------


def test_name_key_normalization() -> None:
    assert normalize_name_key("Michael  J. Petrie, Jr.") == "michael j petrie"
    assert normalize_name_key("MICHAEL PETRIE III") == "michael petrie"
    assert normalize_name_key("Mary-Jane O'Brien") == "mary jane o brien"
    assert normalize_name_key("") is None
    assert normalize_name_key(None) is None


def test_locality_key_normalization() -> None:
    assert normalize_locality_key("Austin, TX") == "austin tx"
    assert normalize_locality_key("  Austin   TX  ") == "austin tx"
    assert normalize_locality_key(None) is None


def test_bare_v_is_not_stripped_as_a_suffix() -> None:
    """A lone "v" is indistinguishable from a middle initial."""
    assert normalize_name_key("Michael V Petrie") == "michael v petrie"


# --- the case the plan names explicitly ------------------------------------


def test_same_name_different_cities_stays_two_candidates() -> None:
    """Never merge on name alone. This is the headline requirement."""
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 2
    assert result.unattached == []
    assert all(c.name_key == "michael petrie" for c in result.candidates)
    assert {frozenset(c.locality_keys) for c in result.candidates} == {
        frozenset({"austin tx"}),
        frozenset({"boston ma"}),
    }
    assert not any(c.is_merged for c in result.candidates)


def test_shared_name_and_employer_alone_does_not_merge() -> None:
    """Employer is common. Only employer plus title is distinctive."""
    drafts = [
        *record(
            "a.example", "origin_a", JAN,
            name="Michael Petrie", city="Austin, TX", employer="Amazon",
        ),
        *record(
            "b.example", "origin_b", FEB,
            name="Michael Petrie", city="Boston, MA", employer="Amazon",
        ),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 2


def test_shared_employer_and_title_does_merge() -> None:
    drafts = [
        *record(
            "a.example", "origin_a", JAN,
            name="Michael Petrie", city="Austin, TX",
            employer="Meridian Freight", job_title="Operations Manager",
        ),
        *record(
            "b.example", "origin_b", FEB,
            name="Michael Petrie", city="Boston, MA",
            employer="Meridian Freight", job_title="Operations Manager",
        ),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    assert result.candidates[0].is_merged
    assert "employer+job_title" in result.candidates[0].merge_evidence[0]


# --- merging on positive evidence -----------------------------------------


def test_shared_email_merges_across_differing_localities() -> None:
    drafts = [
        *record(
            "a.example", "origin_a", JAN,
            name="Michael Petrie", city="Austin, TX", email="m.petrie@example.com",
        ),
        *record(
            "b.example", "origin_b", FEB,
            name="Michael Petrie", city="Boston, MA", email="m.petrie@example.com",
        ),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.is_merged
    assert candidate.locality_keys == {"austin tx", "boston ma"}
    assert len(candidate.assertions) == 6
    assert result.unattached == []
    assert "email" in candidate.merge_evidence[0]


def test_face_distance_below_threshold_merges() -> None:
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]

    def close_faces(left: CandidateDraft, right: CandidateDraft) -> float:
        return 0.11

    result = form_candidates(drafts, face_distance=close_faces)

    assert len(result.candidates) == 1
    assert "face distance 0.110" in result.candidates[0].merge_evidence[0]


def test_face_distance_above_threshold_does_not_merge() -> None:
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]

    result = form_candidates(drafts, face_distance=lambda left, right: 0.91)

    assert len(result.candidates) == 2


def test_face_signal_unavailable_never_degrades_to_name_matching() -> None:
    """A callable returning None must behave exactly like no callable."""
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]

    without = form_candidates(drafts)
    unavailable = form_candidates(drafts, face_distance=lambda left, right: None)

    assert len(without.candidates) == len(unavailable.candidates) == 2


# --- attachment ------------------------------------------------------------


def test_draft_below_threshold_lands_in_unattached() -> None:
    """Same name, different city, no positive evidence: not forced on."""
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("a.example", "origin_a", MAR, name="Michael Petrie", city="Boston, MA"),
    ]
    # One shared block would defeat the point, so make the second a stray
    # record with a locality that matches no candidate.
    stray = record("c.example", "origin_c", FEB, city="Reykjavik")
    result = form_candidates([*drafts, *stray])

    assert len(result.candidates) == 2
    assert len(result.unattached) == 1
    assert result.unattached[0].predicate == "city"
    assert result.unattached[0].raw_value == "Reykjavik"


def test_unattached_drafts_are_never_silently_dropped() -> None:
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("z.example", "origin_z", FEB, phone="+3545551234"),
    ]

    result = form_candidates(drafts)

    total = sum(len(c.assertions) for c in result.candidates) + len(result.unattached)
    assert total == len(drafts)


def test_lowering_the_threshold_changes_attachment() -> None:
    """Thresholds are configuration, not behaviour baked into the code."""
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
        *record("c.example", "origin_c", MAR, name="Michael Petrie"),
    ]

    strict = form_candidates(drafts)
    assert len(strict.unattached) == 1

    permissive = form_candidates(
        drafts, config=FormationConfig(attachment_threshold=0.30)
    )
    assert permissive.unattached == []


# --- zero candidates is a valid success -----------------------------------


def test_nothing_matching_returns_zero_candidates_without_raising() -> None:
    drafts = [
        *record("a.example", "origin_a", JAN, phone="+15125550001"),
        *record("b.example", "origin_b", FEB, email="someone@example.com"),
    ]

    result = form_candidates(drafts)

    assert result.candidates == []
    assert len(result.unattached) == 2
    assert result.formed_nothing is True


def test_empty_input_returns_empty_formation_without_raising() -> None:
    result = form_candidates([])
    assert result.candidates == []
    assert result.unattached == []
    assert result.formed_nothing is True


# --- missing locality, the dominant real-world case -----------------------


def test_no_locality_and_no_competing_name_forms_its_own_candidate() -> None:
    """Nothing to be wrong about, so it stands alone."""
    drafts = record("a.example", "origin_a", JAN, name="Michael Petrie", employer="Acme")

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    assert result.candidates[0].locality_keys == {None}
    assert result.unattached == []


def test_no_locality_does_not_join_a_localized_block_without_evidence() -> None:
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", employer="Acme"),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    assert result.candidates[0].locality_keys == {"austin tx"}
    assert not result.candidates[0].is_merged
    # The locality-free record was not absorbed, and was not discarded either.
    assert {d.predicate for d in result.unattached} == {"name", "employer"}


def test_no_locality_joins_a_localized_block_on_positive_evidence() -> None:
    drafts = [
        *record(
            "a.example", "origin_a", JAN,
            name="Michael Petrie", city="Austin, TX", email="mp@example.com",
        ),
        *record(
            "b.example", "origin_b", FEB,
            name="Michael Petrie", email="mp@example.com",
        ),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.is_merged
    assert candidate.locality_keys == {"austin tx", None}
    assert result.unattached == []


def test_no_locality_with_two_competing_names_stays_unattached() -> None:
    """Choosing between equally plausible candidates would be a guess."""
    drafts = [
        *record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
        *record("c.example", "origin_c", MAR, name="Michael Petrie", employer="Acme"),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 2
    assert {d.predicate for d in result.unattached} == {"name", "employer"}
    assert all(len(c.locality_keys) == 1 for c in result.candidates)


def test_record_with_neither_name_nor_locality_attaches_only_on_evidence() -> None:
    shared_email = "mp@example.com"
    drafts = [
        *record(
            "a.example", "origin_a", JAN,
            name="Michael Petrie", city="Austin, TX", email=shared_email,
        ),
        *record("b.example", "origin_b", FEB, email=shared_email, phone="+15125550001"),
        *record("c.example", "origin_c", MAR, phone="+445550009999"),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    predicates = {d.predicate for d in result.candidates[0].assertions}
    assert predicates == {"name", "city", "email", "phone"}
    # The unrelated phone record shares nothing and stays out.
    assert len(result.unattached) == 1
    assert result.unattached[0].raw_value == "+445550009999"


# --- integration with the fixture adapter ---------------------------------


def test_fixture_06_forms_one_clean_candidate() -> None:
    """All three records share a blocking key, so nothing needs merging."""
    from app.adapters.base import Query
    from app.adapters.fixture import FixtureAdapter

    adapter = FixtureAdapter("06_clean_match")
    drafts = adapter.normalize(adapter.collect(Query(name="Marcus Webb")))

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    assert result.unattached == []

    candidate = result.candidates[0]
    assert candidate.name_key == "marcus a webb"
    assert len(candidate.assertions) == len(drafts) == 9
    assert not candidate.is_merged


# --- record_ref is what groups drafts -------------------------------------


def test_null_record_refs_are_not_pooled_together() -> None:
    """Each unreferenced draft is its own record, never one shared bucket.

    Pooling would let one person's name pair with another's locality.
    """
    drafts = [
        *unlinked(name="Michael Petrie"),
        *unlinked(city="Boston, MA"),
    ]

    result = form_candidates(drafts)

    # The name alone blocks and stands as a candidate; the orphan city does not
    # join it, because nothing says they describe the same person.
    assert len(result.candidates) == 1
    assert result.candidates[0].locality_keys == {None}
    assert [d.raw_value for d in result.unattached] == ["Boston, MA"]


def test_record_ref_groups_drafts_that_publisher_alone_would_not() -> None:
    """Two people in one publisher's output stay two records."""
    shared_publisher = "directory.example"
    drafts = [
        AssertionDraft(
            predicate="name", raw_value="Michael Petrie",
            normalized_value="michael petrie",
            record_ref="row-1", publisher=shared_publisher,
            origin_key="directory", observed_at=JAN,
        ),
        AssertionDraft(
            predicate="city", raw_value="Austin, TX", normalized_value="austin, tx",
            record_ref="row-1", publisher=shared_publisher,
            origin_key="directory", observed_at=JAN,
        ),
        AssertionDraft(
            predicate="name", raw_value="Michael Petrie",
            normalized_value="michael petrie",
            record_ref="row-2", publisher=shared_publisher,
            origin_key="directory", observed_at=JAN,
        ),
        AssertionDraft(
            predicate="city", raw_value="Boston, MA", normalized_value="boston, ma",
            record_ref="row-2", publisher=shared_publisher,
            origin_key="directory", observed_at=JAN,
        ),
    ]

    result = form_candidates(drafts)

    # Same publisher, same origin, same date: the old inference would have
    # collapsed these into one record and one candidate.
    assert len(result.candidates) == 2
    assert {frozenset(c.locality_keys) for c in result.candidates} == {
        frozenset({"austin tx"}),
        frozenset({"boston ma"}),
    }
