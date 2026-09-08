"""Scored drafts must survive a trip through the database unchanged.

A score is only trustworthy if the inputs behind it are durable. If a column is
missing for something scoring reads, a reloaded assertion re-scores differently
and the stored number quietly stops matching the evidence. These tests pin the
round trip so that gap shows up here rather than in a report.

Requires Postgres. Skipped, not failed, when it is unreachable, so the rest of
the suite stays runnable without Docker.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.adapters.base import AssertionDraft, Query
from app.adapters.fixture import FixtureAdapter
from app.db import engine, session_scope
from app.engine.candidates import CandidateDraft, form_candidates
from app.engine.scoring import score_identity, score_item
from app.models import (
    Assertion,
    Candidate,
    EvidenceKind,
    GroupingBasis,
    SourceOrigin,
    Subject,
)

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _database_available(),
    reason="Postgres unavailable; start it with docker compose up -d",
)


def _to_row(draft: AssertionDraft, subject_id: int, candidate_id: int, collector: str):
    """Persist every field of a draft that the schema has a home for."""
    return Assertion(
        subject_id=subject_id,
        candidate_id=candidate_id,
        predicate=draft.predicate,
        raw_value=draft.raw_value,
        normalized_value=draft.normalized_value,
        record_ref=draft.record_ref,
        origin_key=draft.origin_key,
        source_origin=draft.source_origin,
        evidence_kind=draft.evidence_kind,
        observed_at=draft.observed_at,
        collector=collector,
    )


def _to_draft(row: Assertion) -> AssertionDraft:
    """Rebuild a draft from a stored row, reading only stored columns.

    ``publisher`` is deliberately not restored: it lives on Provenance, and
    scoring never reads it. Corroboration counts origins, so a missing
    publisher cannot change a score.
    """
    return AssertionDraft(
        predicate=row.predicate,
        raw_value=row.raw_value,
        normalized_value=row.normalized_value,
        record_ref=row.record_ref,
        origin_key=row.origin_key,
        source_origin=row.source_origin,
        evidence_kind=row.evidence_kind,
        observed_at=row.observed_at,
    )


@pytest.fixture
def fixture_candidate() -> tuple[FixtureAdapter, CandidateDraft]:
    adapter = FixtureAdapter("06_clean_match")
    drafts = adapter.normalize(adapter.collect(Query(name="Marcus Webb")))
    return adapter, form_candidates(drafts).candidates[0]


@pytest.fixture
def clean_db():
    """Each test starts and ends with no subjects."""
    with session_scope() as session:
        for subject in session.query(Subject).all():
            session.delete(subject)
    yield
    with session_scope() as session:
        for subject in session.query(Subject).all():
            session.delete(subject)


# --- the round trip --------------------------------------------------------


def test_scored_draft_round_trips_with_q_unchanged(clean_db, fixture_candidate) -> None:
    """Persist, reload, re-score: every q must come back identical."""
    adapter, candidate = fixture_candidate

    before = {
        (d.origin_key, d.predicate): score_item(d, candidate, now=NOW)
        for d in candidate.assertions
    }

    with session_scope() as session:
        subject = Subject(name="Marcus Webb", address="Austin, TX")
        session.add(subject)
        session.flush()
        row_candidate = Candidate(
            subject_id=subject.id,
            display_name="Marcus A. Webb",
            identity_confidence=0.78,
        )
        session.add(row_candidate)
        session.flush()
        for draft in candidate.assertions:
            session.add(
                _to_row(draft, subject.id, row_candidate.id, adapter.name)
            )
        session.flush()
        subject_id = subject.id

    with session_scope() as session:
        rows = session.get(Subject, subject_id).assertions
        assert len(rows) == len(candidate.assertions)

        reloaded = [_to_draft(row) for row in rows]
        reloaded_candidate = CandidateDraft(
            name_key=candidate.name_key,
            locality_keys=candidate.locality_keys,
            assertions=reloaded,
        )

        for draft in reloaded:
            after = score_item(draft, reloaded_candidate, now=NOW)
            original = before[(draft.origin_key, draft.predicate)]

            assert float(after) == pytest.approx(float(original)), (
                f"q changed for {draft.origin_key}/{draft.predicate}"
            )
            assert after.components == original.components
            assert after.attachment_basis == original.attachment_basis
            assert after.freshness_unknown == original.freshness_unknown


def test_reliability_basis_survives_the_round_trip(clean_db, fixture_candidate) -> None:
    """evidence_kind is what reliability reads, so it must come back exactly."""
    adapter, candidate = fixture_candidate
    expected = {
        (d.origin_key, d.predicate): d.evidence_kind for d in candidate.assertions
    }

    with session_scope() as session:
        subject = Subject(name="Marcus Webb")
        session.add(subject)
        session.flush()
        for draft in candidate.assertions:
            session.add(_to_row(draft, subject.id, None, adapter.name))
        session.flush()
        subject_id = subject.id

    with session_scope() as session:
        rows = session.get(Subject, subject_id).assertions
        actual = {(r.origin_key, r.predicate): r.evidence_kind for r in rows}

    assert actual == expected
    # All three kinds are represented, so this is not a trivially-equal check.
    assert set(actual.values()) == {
        EvidenceKind.direct_observation,
        EvidenceKind.self_reported,
        EvidenceKind.republished,
    }


def test_the_two_axes_persist_independently(clean_db, fixture_candidate) -> None:
    """One lineage value across three different evidence kinds, stored as such."""
    adapter, candidate = fixture_candidate

    with session_scope() as session:
        subject = Subject(name="Marcus Webb")
        session.add(subject)
        session.flush()
        for draft in candidate.assertions:
            session.add(_to_row(draft, subject.id, None, adapter.name))
        session.flush()
        subject_id = subject.id

    with session_scope() as session:
        rows = session.get(Subject, subject_id).assertions
        assert {r.source_origin for r in rows} == {SourceOrigin.known_origin}
        assert len({r.evidence_kind for r in rows}) == 3


def test_evidence_kind_defaults_to_unknown_when_not_supplied(clean_db) -> None:
    """A row written without a judgement is not credited with having one."""
    with session_scope() as session:
        subject = Subject(name="Nobody")
        session.add(subject)
        session.flush()
        session.add(
            Assertion(
                subject_id=subject.id,
                predicate="employer",
                normalized_value="acme",
                collector="test",
            )
        )
        session.flush()
        subject_id = subject.id

    with session_scope() as session:
        (row,) = session.get(Subject, subject_id).assertions
        assert row.evidence_kind is EvidenceKind.unknown


def test_evidence_kind_is_not_nullable(clean_db) -> None:
    """Enforced by the database, not only by the ORM default.

    Going through the ORM cannot demonstrate this: the column default replaces
    a None before the insert is emitted, so the constraint is never reached.
    A raw insert is the only way to ask the database directly.
    """
    from sqlalchemy.exc import IntegrityError

    with session_scope() as session:
        subject = Subject(name="Nobody")
        session.add(subject)
        session.flush()
        subject_id = subject.id

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.execute(
                text(
                    "INSERT INTO assertion "
                    "(subject_id, predicate, collector, source_origin, "
                    " evidence_kind, review_status) "
                    "VALUES (:sid, 'employer', 'test', 'unknown', NULL, "
                    "'unreviewed')"
                ),
                {"sid": subject_id},
            )


def test_identity_score_also_survives(clean_db, fixture_candidate) -> None:
    """p depends on stored fields too, so it must reload identically."""
    adapter, candidate = fixture_candidate
    subject_query = Query(name="Marcus Webb", address="Austin, TX")
    before = score_identity(candidate, subject_query)

    with session_scope() as session:
        subject = Subject(name="Marcus Webb", address="Austin, TX")
        session.add(subject)
        session.flush()
        row_candidate = Candidate(
            subject_id=subject.id,
            identity_confidence=float(before),
            grouping_basis=candidate.grouping_basis,
        )
        session.add(row_candidate)
        session.flush()
        for draft in candidate.assertions:
            session.add(_to_row(draft, subject.id, row_candidate.id, adapter.name))
        session.flush()
        subject_id = subject.id
        candidate_id = row_candidate.id

    with session_scope() as session:
        rows = session.get(Subject, subject_id).assertions
        stored = session.get(Candidate, candidate_id)
        # The grouping basis scales p, so it has to come back with the rest.
        assert stored.grouping_basis is candidate.grouping_basis
        reloaded_candidate = CandidateDraft(
            name_key=candidate.name_key,
            locality_keys=candidate.locality_keys,
            grouping_basis=stored.grouping_basis,
            assertions=[_to_draft(r) for r in rows],
        )
        after = score_identity(reloaded_candidate, subject_query)

    assert float(after) == pytest.approx(float(before))
    assert after.grouping_basis is before.grouping_basis
    assert after.components == before.components
    assert after.consumed_record_refs == before.consumed_record_refs


def test_grouping_basis_defaults_to_unspecified(clean_db) -> None:
    """A candidate written without a judgement is not credited with one."""
    with session_scope() as session:
        subject = Subject(name="Nobody")
        session.add(subject)
        session.flush()
        session.add(Candidate(subject_id=subject.id))
        session.flush()
        subject_id = subject.id

    with session_scope() as session:
        (row,) = session.get(Subject, subject_id).candidates
        assert row.grouping_basis is GroupingBasis.unspecified
