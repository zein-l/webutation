"""Fixture 6, the clean match, through the adapter path and into formation.

The point of a clean match is that nothing about it is ambiguous: one person,
every claim attached, nothing stranded. That end state is asserted here, since
a fixture that cannot demonstrate it would validate nothing.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pytest

from app.adapters.base import Adapter, AssertionDraft, Query, RawResponse
from app.adapters.fixture import FixtureAdapter
from app.engine.candidates import form_candidates
from app.models import AccessCategory, EvidenceKind, SourceOrigin

#: Nine drafts across three source records.
EXPECTED_DRAFTS = 9


@pytest.fixture
def adapter() -> FixtureAdapter:
    return FixtureAdapter("06_clean_match")


@pytest.fixture
def raw(adapter: FixtureAdapter) -> RawResponse:
    return adapter.collect(Query(name="Marcus Webb", address="Austin, TX"))


@pytest.fixture
def drafts(adapter: FixtureAdapter, raw: RawResponse) -> list[AssertionDraft]:
    return adapter.normalize(raw)


# --- draft counts and origins ---------------------------------------------


def test_fixture_06_draft_count(drafts: list[AssertionDraft]) -> None:
    assert len(drafts) == EXPECTED_DRAFTS


def test_fixture_06_spans_three_distinct_origins(
    drafts: list[AssertionDraft],
) -> None:
    origins = {d.origin_key for d in drafts}
    assert len(origins) == 3
    assert origins == {"travis_county_clerk", "linkedin", "pdl"}


def test_fixture_06_spans_three_source_records(
    drafts: list[AssertionDraft],
) -> None:
    """One record per search result, each separately identified."""
    refs = {d.record_ref for d in drafts}
    assert len(refs) == 3
    assert all(ref is not None for ref in refs)
    assert Counter(d.record_ref for d in drafts) == Counter(
        {
            "06_clean_match:travis-mw-88213": 3,
            "06_clean_match:li-marcus-webb-atx": 4,
            "06_clean_match:pdl-MW-4471902": 2,
        }
    )


# --- the clean-match outcome ----------------------------------------------


def test_fixture_06_forms_exactly_one_candidate_with_nothing_unattached(
    drafts: list[AssertionDraft],
) -> None:
    """This is what "clean match" means, and the reason the fixture exists."""
    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    assert result.unattached == []

    candidate = result.candidates[0]
    assert candidate.name_key == "marcus a webb"
    assert candidate.locality_keys == {"austin tx"}
    assert len(candidate.assertions) == EXPECTED_DRAFTS
    # One block held all three records, so no merge was needed and none is
    # claimed. Merging here would have meant the blocking key failed.
    assert not candidate.is_merged
    assert candidate.merge_evidence == []


def test_each_source_names_the_person_in_its_own_vocabulary(
    drafts: list[AssertionDraft],
) -> None:
    """full_name, profile_name and display_name all mean "name".

    They also differ in case and punctuation, so agreement depends on
    normalization rather than on the sources happening to match.
    """
    names = [d for d in drafts if d.predicate == "name"]
    assert len(names) == 3
    assert {d.raw_value for d in names} == {
        "MARCUS A. WEBB",
        "Marcus A. Webb",
        "Marcus A Webb",
    }
    assert {d.normalized_value for d in names} == {"marcus a webb"}


def test_each_source_states_the_locality_in_its_own_vocabulary(
    drafts: list[AssertionDraft],
) -> None:
    """city, location and locality all mean "city"."""
    localities = [d for d in drafts if d.predicate == "city"]
    assert len(localities) == 3
    assert {d.normalized_value for d in localities} == {"austin, tx"}


# --- draft contract -------------------------------------------------------


def test_every_draft_names_its_origin(drafts: list[AssertionDraft]) -> None:
    """A known origin with no origin_key would be uncountable."""
    for d in drafts:
        assert d.origin_key is not None
        assert d.source_origin is SourceOrigin.known_origin


def test_publishers_and_origins_stay_separable(
    drafts: list[AssertionDraft],
) -> None:
    """Corroboration counts origins, so the two counts must not be conflated."""
    assert len({d.publisher for d in drafts}) == 3
    assert len({d.origin_key for d in drafts}) == 3


def test_predicates_and_normalized_values(drafts: list[AssertionDraft]) -> None:
    by_predicate: dict[str, list[AssertionDraft]] = {}
    for d in drafts:
        by_predicate.setdefault(d.predicate, []).append(d)

    assert set(by_predicate) == {"name", "birth_date", "employer", "job_title", "city"}

    # Normalization produces a canonical form, not the publisher's rendering.
    (birth_date,) = by_predicate["birth_date"]
    assert birth_date.raw_value == "04/02/1979"
    assert birth_date.normalized_value == "1979-04-02"

    (employer,) = by_predicate["employer"]
    assert employer.normalized_value == "meridian freight systems"

    (job_title,) = by_predicate["job_title"]
    assert job_title.normalized_value == "operations manager"


def test_access_category_is_recorded_per_result(
    drafts: list[AssertionDraft],
) -> None:
    """Descriptive metadata, carried through without adjudicating use."""
    by_origin = {d.origin_key: d.access_category for d in drafts}
    assert by_origin["travis_county_clerk"] is AccessCategory.PUBLIC_WEB
    assert by_origin["linkedin"] is AccessCategory.PUBLIC_WEB
    assert by_origin["pdl"] is AccessCategory.LICENSED


def test_no_conflicting_values_for_any_predicate(
    drafts: list[AssertionDraft],
) -> None:
    """Fixture 6 is the unambiguous case: one value per predicate."""
    seen: dict[str, set[str | None]] = {}
    for d in drafts:
        seen.setdefault(d.predicate, set()).add(d.normalized_value)
    assert all(len(values) == 1 for values in seen.values())


# --- the collect / normalize contract --------------------------------------


def test_fixture_adapter_satisfies_the_adapter_protocol(
    adapter: FixtureAdapter,
) -> None:
    assert isinstance(adapter, Adapter)


def test_collect_returns_a_traceable_raw_response(raw: RawResponse) -> None:
    assert raw.source == "fixture:06_clean_match"
    assert raw.artifact_ref.startswith("fixture:06_clean_match:")
    assert raw.fetched_at.tzinfo is not None
    # The payload is the source response, untouched.
    assert "search_metadata" in raw.payload
    assert len(raw.payload["results"]) == 3


def test_artifact_ref_is_stable_across_collections(
    adapter: FixtureAdapter,
) -> None:
    """Same bytes, same reference, so a draft traces back to its artifact."""
    first = adapter.collect(Query(name="Marcus Webb"))
    second = adapter.collect(Query(name="Marcus Webb"))
    assert first.artifact_ref == second.artifact_ref


def test_observed_at_is_not_the_fetch_time(
    drafts: list[AssertionDraft], raw: RawResponse
) -> None:
    """Freshness comes from observation, never from collection."""
    for d in drafts:
        assert d.observed_at is not None
        assert d.observed_at < raw.fetched_at

    birth_date = next(d for d in drafts if d.predicate == "birth_date")
    assert birth_date.observed_at == datetime(2019, 4, 2, tzinfo=timezone.utc)


def test_normalize_is_pure(adapter: FixtureAdapter, raw: RawResponse) -> None:
    """Re-normalizing a stored artifact gives the same drafts."""
    assert adapter.normalize(raw) == adapter.normalize(raw)


def test_record_ref_falls_back_to_link_then_position(
    adapter: FixtureAdapter, raw: RawResponse
) -> None:
    """A source without explicit ids is still separable into records."""
    payload = {
        "results": [
            {
                "position": 1,
                "link": "https://a.example/p/1",
                "fields": {"full_name": "Marcus A. Webb"},
            },
            {
                "position": 2,
                "fields": {"full_name": "Marcus A. Webb"},
            },
        ]
    }
    stripped = RawResponse(
        payload=payload,
        fetched_at=raw.fetched_at,
        artifact_ref=raw.artifact_ref,
        source=raw.source,
    )
    first, second = adapter.normalize(stripped)
    assert first.record_ref == "06_clean_match:https://a.example/p/1"
    assert second.record_ref == "06_clean_match:position:2"


def test_unknown_lineage_is_not_credited_to_the_adapter(
    adapter: FixtureAdapter, raw: RawResponse
) -> None:
    """A record with no declared origin stays unknown, never uncountable."""
    payload = {
        "results": [
            {
                "position": 1,
                "publisher": "mirror.example",
                "record_date": "2024-01-05",
                "fields": {"current_employer": "Meridian Freight Systems"},
            }
        ]
    }
    stripped = RawResponse(
        payload=payload,
        fetched_at=raw.fetched_at,
        artifact_ref=raw.artifact_ref,
        source=raw.source,
    )
    (draft,) = adapter.normalize(stripped)
    assert draft.origin_key is None
    assert draft.source_origin is SourceOrigin.unknown


# --- evidence kind is its own axis ----------------------------------------


def test_each_result_declares_how_it_knows_what_it_claims(
    drafts: list[AssertionDraft],
) -> None:
    """A filing office observed it; a profile asserts it; an aggregator copies it."""
    by_origin = {d.origin_key: d.evidence_kind for d in drafts}
    assert by_origin["travis_county_clerk"] is EvidenceKind.direct_observation
    assert by_origin["linkedin"] is EvidenceKind.self_reported
    assert by_origin["pdl"] is EvidenceKind.republished


def test_evidence_kind_is_independent_of_lineage(
    drafts: list[AssertionDraft],
) -> None:
    """All three origins are known_origin, yet reliability differs across them.

    This is the separation the two fields exist to express: lineage says the
    origin is identified, evidence kind says how that origin came to know it.
    """
    assert {d.source_origin for d in drafts} == {SourceOrigin.known_origin}
    assert len({d.evidence_kind for d in drafts}) == 3


def test_evidence_kind_can_be_overridden_per_claim(
    adapter: FixtureAdapter, raw: RawResponse
) -> None:
    """Record-level default, per-field override, mirroring how origin works."""
    payload = {
        "results": [
            {
                "position": 1,
                "evidence": {
                    "kind": "self_reported",
                    "by_field": {"date_of_birth": "direct_observation"},
                },
                "fields": {
                    "current_employer": "Meridian Freight Systems",
                    "date_of_birth": "04/02/1979",
                },
            }
        ]
    }
    stripped = RawResponse(
        payload=payload,
        fetched_at=raw.fetched_at,
        artifact_ref=raw.artifact_ref,
        source=raw.source,
    )
    by_predicate = {d.predicate: d.evidence_kind for d in adapter.normalize(stripped)}
    assert by_predicate["employer"] is EvidenceKind.self_reported
    assert by_predicate["birth_date"] is EvidenceKind.direct_observation


def test_absent_or_unrecognised_evidence_kind_falls_back_to_unknown(
    adapter: FixtureAdapter, raw: RawResponse
) -> None:
    """Never inferred from lineage, which would rebuild the coupling."""
    payload = {
        "results": [
            {
                "position": 1,
                "lineage": {"origin": "county", "state": "known_origin"},
                "fields": {"current_employer": "Meridian Freight Systems"},
            },
            {
                "position": 2,
                "evidence": {"kind": "hearsay"},
                "fields": {"job_title": "Operations Manager"},
            },
        ]
    }
    stripped = RawResponse(
        payload=payload,
        fetched_at=raw.fetched_at,
        artifact_ref=raw.artifact_ref,
        source=raw.source,
    )
    absent, unrecognised = adapter.normalize(stripped)
    assert absent.source_origin is SourceOrigin.known_origin
    assert absent.evidence_kind is EvidenceKind.unknown
    assert unrecognised.evidence_kind is EvidenceKind.unknown
