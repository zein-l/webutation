"""Scoring: p, q, and the r derived from them.

The cases the plan treats as load-bearing: a mirror must buy no corroboration,
a re-fetch must buy no freshness, p and q must stay separable in r, and a claim
with no observation date must say so rather than quietly scoring as average.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import CandidateDraft, form_candidates
from app.engine.scoring import (
    ScoringConfig,
    derive_attribution,
    score_identity,
    score_item,
)
from app.models import EvidenceKind, SourceOrigin

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)
_REF_SEQUENCE = itertools.count(1)


def draft(
    predicate: str,
    value: str,
    *,
    origin: str | None = "origin_a",
    lineage: SourceOrigin = SourceOrigin.known_origin,
    evidence: EvidenceKind = EvidenceKind.direct_observation,
    publisher: str = "a.example",
    observed_at: datetime | None = NOW - timedelta(days=30),
    record_ref: str | None = None,
) -> AssertionDraft:
    """A draft with strong defaults, so each test varies one thing.

    ``lineage`` and ``evidence`` are separate parameters because they are
    separate axes: the first drives corroboration, the second reliability.
    """
    return AssertionDraft(
        predicate=predicate,
        raw_value=value,
        normalized_value=value.casefold(),
        record_ref=record_ref if record_ref is not None else f"r{next(_REF_SEQUENCE)}",
        publisher=publisher,
        origin_key=origin,
        source_origin=lineage,
        evidence_kind=evidence,
        observed_at=observed_at,
    )


def candidate_of(*drafts: AssertionDraft, name: str = "michael petrie") -> CandidateDraft:
    return CandidateDraft(
        name_key=name,
        locality_keys={"austin tx"},
        assertions=list(drafts),
    )


SUBJECT = Query(name="Michael Petrie", address="Austin, TX")


# --- corroboration counts origins, never publishers or drafts -------------


def test_adding_a_mirror_source_produces_no_corroboration_gain() -> None:
    """Five sites repeating one aggregator is one confirmation."""
    ref = "record-1"
    original = candidate_of(
        draft("employer", "Meridian Freight", origin="pdl", record_ref=ref)
    )
    baseline = score_identity(original, SUBJECT)

    # Four more publishers, all republishing the same origin.
    mirrored = candidate_of(
        draft("employer", "Meridian Freight", origin="pdl", record_ref=ref),
        *[
            draft(
                "employer",
                "Meridian Freight",
                origin="pdl",
                publisher=f"mirror{i}.example",
                record_ref=f"mirror-{i}",
            )
            for i in range(4)
        ],
    )
    mirrored_score = score_identity(mirrored, SUBJECT)

    assert mirrored_score == baseline
    assert mirrored_score.components["corroboration"] == pytest.approx(0.0)


def test_a_suspected_copy_earns_no_independence_bonus() -> None:
    genuine = candidate_of(
        draft("employer", "Meridian Freight", origin="pdl"),
        draft("employer", "Meridian Freight", origin="linkedin"),
    )
    copied = candidate_of(
        draft("employer", "Meridian Freight", origin="pdl"),
        draft(
            "employer",
            "Meridian Freight",
            origin="scraper",
            lineage=SourceOrigin.suspected_copy,
        ),
    )

    assert score_identity(genuine, SUBJECT) > score_identity(copied, SUBJECT)
    assert score_identity(copied, SUBJECT).components["corroboration"] == pytest.approx(
        0.0
    )


def test_unknown_lineage_is_never_counted_as_independent() -> None:
    unknown = candidate_of(
        draft("employer", "Meridian Freight", origin="pdl"),
        draft(
            "employer",
            "Meridian Freight",
            origin="mystery",
            lineage=SourceOrigin.unknown,
        ),
    )
    assert unknown.assertions[1].source_origin is SourceOrigin.unknown
    assert score_identity(unknown, SUBJECT).components["corroboration"] == pytest.approx(
        0.0
    )


def test_distinct_known_origins_do_raise_corroboration() -> None:
    one = candidate_of(draft("employer", "Meridian Freight", origin="pdl"))
    three = candidate_of(
        draft("employer", "Meridian Freight", origin="pdl"),
        draft("employer", "Meridian Freight", origin="linkedin"),
        draft("employer", "Meridian Freight", origin="county"),
    )
    assert score_identity(three, SUBJECT) > score_identity(one, SUBJECT)


# --- recency uses observed_at, never fetched_at ---------------------------


def test_refetching_an_old_assertion_produces_no_freshness_gain() -> None:
    """A mirror collected today must not refresh a five-year-old claim."""
    old = NOW - timedelta(days=5 * 365)
    stale = draft("phone", "+15125550001", observed_at=old)
    candidate = candidate_of(stale)

    first = score_item(stale, candidate, now=NOW)

    # Re-collected today. fetched_at is not a field scoring may read, and the
    # observation date is unchanged, so nothing about freshness may move.
    refetched = draft("phone", "+15125550001", observed_at=old, origin="origin_a")
    later = score_item(refetched, candidate_of(refetched), now=NOW + timedelta(days=1))

    assert later.components["freshness"] == pytest.approx(
        first.components["freshness"], abs=1e-3
    )
    assert first.components["freshness"] < 0.1


def test_null_observed_at_flags_unknown_freshness_rather_than_defaulting() -> None:
    undated = draft("phone", "+15125550001", observed_at=None)
    dated = draft("phone", "+15125550001", observed_at=NOW - timedelta(days=1))

    unknown = score_item(undated, candidate_of(undated), now=NOW)
    known = score_item(dated, candidate_of(dated), now=NOW)

    assert unknown.freshness_unknown is True
    assert known.freshness_unknown is False
    # The stand-in is neither fresh nor stale, and is not a fresh score.
    config = ScoringConfig()
    assert unknown.components["freshness"] == pytest.approx(
        config.freshness_when_unknown
    )
    assert unknown.components["freshness"] < known.components["freshness"]


def test_field_specific_decay_phone_fast_birth_date_not_at_all() -> None:
    old = NOW - timedelta(days=3 * 365)
    phone = draft("phone", "+15125550001", observed_at=old)
    birth = draft("birth_date", "1979-04-02", observed_at=old)

    phone_score = score_item(phone, candidate_of(phone), now=NOW)
    birth_score = score_item(birth, candidate_of(birth), now=NOW)

    assert phone_score.components["freshness"] < 0.05
    assert birth_score.components["freshness"] == pytest.approx(1.0)
    assert birth_score.freshness_unknown is False


def test_address_decays_faster_than_employer() -> None:
    old = NOW - timedelta(days=400)
    address = draft("address", "1204 W 6th St", observed_at=old)
    employer = draft("employer", "Meridian Freight", observed_at=old)

    a = score_item(address, candidate_of(address), now=NOW)
    e = score_item(employer, candidate_of(employer), now=NOW)
    assert a.components["freshness"] < e.components["freshness"]


# --- q is conditional, and never capped at p ------------------------------


def test_two_candidates_differing_only_in_p_give_different_r_for_one_q() -> None:
    """The product keeps p and q separable where min(p, q) would not."""
    claim = draft("employer", "Meridian Freight")
    candidate = candidate_of(claim)
    q = score_item(claim, candidate, now=NOW)

    strong_p, weak_p = 0.90, 0.40
    strong_r = derive_attribution(strong_p, q)
    weak_r = derive_attribution(weak_p, q)

    assert strong_r != weak_r
    assert strong_r == pytest.approx(strong_p * float(q))
    assert weak_r == pytest.approx(weak_p * float(q))
    assert weak_r < strong_r


def test_q_is_not_capped_at_p() -> None:
    """A high q survives a low p instead of being flattened to it."""
    ref = "rec-1"
    claim = draft("birth_date", "1979-04-02", origin="county", record_ref=ref)
    candidate = candidate_of(
        draft("name", "Michael Petrie", origin="county", record_ref=ref),
        draft("city", "Austin, TX", origin="county", record_ref=ref),
        claim,
        draft("birth_date", "1979-04-02", origin="pdl", record_ref="rec-2"),
    )
    q = score_item(claim, candidate, now=NOW)

    low_p = 0.40
    assert float(q) > low_p
    r = derive_attribution(low_p, q)
    assert r == pytest.approx(low_p * float(q))
    # The product, not the minimum. min(p, q) would have returned 0.40.
    assert r < low_p
    assert r != pytest.approx(min(low_p, float(q)))


def test_the_plans_worked_example() -> None:
    """p=0.40 with q=0.95 and q=0.55 must give 0.38 and 0.22, not 0.40 twice."""
    assert derive_attribution(0.40, 0.95) == pytest.approx(0.38)
    assert derive_attribution(0.40, 0.55) == pytest.approx(0.22)
    # What the rule exists to prevent: the minimum collapses both to one value.
    assert min(0.40, 0.95) == min(0.40, 0.55) == pytest.approx(0.40)


def test_attribution_is_none_when_either_component_is_missing() -> None:
    assert derive_attribution(None, 0.9) is None
    assert derive_attribution(0.9, None) is None
    assert derive_attribution(None, None) is None


def test_record_attachment_uncertainty_makes_q_conditional() -> None:
    """A record attached on name alone carries more risk than one with an email."""
    ref = "shared-record"
    name_only = draft("name", "Michael Petrie", record_ref=ref)
    claim_a = draft("employer", "Meridian Freight", record_ref=ref)
    weak = score_item(claim_a, candidate_of(name_only, claim_a), now=NOW)

    ref2 = "strong-record"
    email = draft("email", "mp@example.com", record_ref=ref2)
    claim_b = draft("employer", "Meridian Freight", record_ref=ref2)
    strong = score_item(claim_b, candidate_of(email, claim_b), now=NOW)

    assert weak.attachment_basis == "name_only"
    assert strong.attachment_basis == "distinctive_attribute"
    assert float(weak) < float(strong)


# --- no double-counting ----------------------------------------------------


def test_evidence_consumed_by_the_match_is_discounted_not_excluded() -> None:
    claim = draft("employer", "Meridian Freight", origin="pdl", record_ref="rec-1")
    corroborating = draft(
        "employer", "Meridian Freight", origin="linkedin", record_ref="rec-2"
    )
    candidate = candidate_of(claim, corroborating)

    identity = score_identity(candidate, SUBJECT)
    assert "rec-2" in identity.consumed_record_refs

    fresh = score_item(claim, candidate, now=NOW)
    discounted = score_item(
        claim, candidate, consumed_record_refs=identity.consumed_record_refs, now=NOW
    )

    # Reused evidence still counts for something, just not full independence.
    assert discounted.components["corroboration"] < fresh.components["corroboration"]
    assert discounted.components["corroboration"] > 0.0


def test_unconsumed_evidence_keeps_full_corroboration_weight() -> None:
    claim = draft("employer", "Meridian Freight", origin="pdl", record_ref="rec-1")
    corroborating = draft(
        "employer", "Meridian Freight", origin="linkedin", record_ref="rec-2"
    )
    candidate = candidate_of(claim, corroborating)

    full = score_item(
        claim, candidate, consumed_record_refs=frozenset({"unrelated"}), now=NOW
    )
    none_passed = score_item(claim, candidate, now=NOW)
    assert full.components["corroboration"] == pytest.approx(
        none_passed.components["corroboration"]
    )


# --- identity signals ------------------------------------------------------


def test_face_distance_when_available_raises_p() -> None:
    candidate = candidate_of(draft("employer", "Meridian Freight"))
    without = score_identity(candidate, SUBJECT)
    close = score_identity(candidate, SUBJECT, face_distance=0.10)
    far = score_identity(candidate, SUBJECT, face_distance=0.80)

    assert close > without > far
    assert without.components["face"] is None
    assert close.components["face"] == pytest.approx(1.0)
    assert far.components["face"] == pytest.approx(0.0)


def test_absent_signal_is_not_scored_as_zero() -> None:
    """Silence must not be read as dissent."""
    candidate = CandidateDraft(
        name_key="michael petrie",
        locality_keys=set(),
        assertions=[draft("employer", "Meridian Freight")],
    )
    no_locality = score_identity(candidate, SUBJECT)
    assert no_locality.components["locality"] is None
    assert float(no_locality) > 0.0


def test_name_match_strength_grades() -> None:
    base = [draft("employer", "Meridian Freight")]
    exact = score_identity(candidate_of(*base, name="michael petrie"), SUBJECT)
    subset = score_identity(candidate_of(*base, name="michael j petrie"), SUBJECT)
    unrelated = score_identity(candidate_of(*base, name="susan calvin"), SUBJECT)

    assert exact > subset > unrelated
    assert exact.components["name"] == pytest.approx(ScoringConfig().name_exact)


def test_incompatible_locality_is_penalised_as_a_contradiction() -> None:
    elsewhere = CandidateDraft(
        name_key="michael petrie",
        locality_keys={"boston ma"},
        assertions=[draft("employer", "Meridian Freight")],
    )
    here = candidate_of(draft("employer", "Meridian Freight"))

    away = score_identity(elsewhere, SUBJECT)
    assert away < score_identity(here, SUBJECT)
    assert "incompatible_locality" in away.penalties


def test_contradictory_birth_dates_penalise_p() -> None:
    consistent = candidate_of(draft("birth_date", "1979-04-02"))
    conflicting = candidate_of(
        draft("birth_date", "1979-04-02", origin="pdl"),
        draft("birth_date", "1981-11-30", origin="linkedin"),
    )

    penalised = score_identity(conflicting, SUBJECT)
    assert "contradictory_birth_date" in penalised.penalties
    assert penalised < score_identity(consistent, SUBJECT)


# --- configuration ---------------------------------------------------------


def test_all_thresholds_come_from_config() -> None:
    candidate = candidate_of(draft("phone", "+15125550001"))
    claim = candidate.assertions[0]

    default = score_item(claim, candidate, now=NOW)
    impatient = score_item(
        claim,
        candidate,
        config=ScoringConfig(
            half_life_days_by_predicate={"phone": 1.0},
        ),
        now=NOW,
    )
    assert impatient.components["freshness"] < default.components["freshness"]


def test_scores_stay_within_bounds() -> None:
    config = ScoringConfig()
    everything_wrong = CandidateDraft(
        name_key="susan calvin",
        locality_keys={"boston ma"},
        assertions=[
            draft("birth_date", "1979-04-02", origin="pdl"),
            draft("birth_date", "1981-11-30", origin="linkedin"),
        ],
    )
    p = score_identity(everything_wrong, SUBJECT, face_distance=0.99)
    assert config.score_floor <= float(p) <= config.score_ceiling

    for assertion in everything_wrong.assertions:
        q = score_item(assertion, everything_wrong, now=NOW)
        assert config.score_floor <= float(q) <= config.score_ceiling


# --- end to end on the clean-match fixture --------------------------------


def test_fixture_06_scores_coherently() -> None:
    from app.adapters.fixture import FixtureAdapter

    adapter = FixtureAdapter("06_clean_match")
    drafts = adapter.normalize(adapter.collect(Query(name="Marcus Webb")))
    formation = form_candidates(drafts)
    candidate = formation.candidates[0]

    subject = Query(name="Marcus Webb", address="Austin, TX")
    p = score_identity(candidate, subject)
    assert 0.0 < float(p) <= 1.0
    assert p.penalties == {}

    for assertion in candidate.assertions:
        q = score_item(
            assertion,
            candidate,
            consumed_record_refs=p.consumed_record_refs,
            now=NOW,
        )
        r = derive_attribution(p, q)
        assert r == pytest.approx(float(p) * float(q))
        # Every fixture 6 claim carries an observation date.
        assert q.freshness_unknown is False


# --- reliability comes from evidence_kind, not lineage --------------------


def test_two_claims_differing_only_in_evidence_kind_give_different_q() -> None:
    """The whole point of splitting the field: reliability must move on its own."""
    observed = draft("employer", "Meridian Freight", evidence=EvidenceKind.direct_observation)
    claimed = draft("employer", "Meridian Freight", evidence=EvidenceKind.self_reported)
    copied = draft("employer", "Meridian Freight", evidence=EvidenceKind.republished)

    q_observed = score_item(observed, candidate_of(observed), now=NOW)
    q_claimed = score_item(claimed, candidate_of(claimed), now=NOW)
    q_copied = score_item(copied, candidate_of(copied), now=NOW)

    assert float(q_observed) > float(q_claimed) > float(q_copied)

    config = ScoringConfig()
    assert q_observed.components["reliability"] == pytest.approx(
        config.reliability_by_evidence_kind[EvidenceKind.direct_observation]
    )
    assert q_claimed.components["reliability"] == pytest.approx(
        config.reliability_by_evidence_kind[EvidenceKind.self_reported]
    )
    # Everything else about these claims is identical.
    for key in ("freshness", "corroboration", "attachment_certainty"):
        assert q_observed.components[key] == q_claimed.components[key]


def test_lineage_no_longer_moves_reliability() -> None:
    """source_origin is for corroboration only now."""
    known = draft("birth_date", "1979-04-02", lineage=SourceOrigin.known_origin)
    copy = draft("birth_date", "1979-04-02", lineage=SourceOrigin.suspected_copy)

    q_known = score_item(known, candidate_of(known), now=NOW)
    q_copy = score_item(copy, candidate_of(copy), now=NOW)

    assert q_known.components["reliability"] == q_copy.components["reliability"]
    assert float(q_known) == pytest.approx(float(q_copy))


def test_the_two_axes_vary_independently() -> None:
    """A first-party origin can relay something it never observed."""
    relayed = draft(
        "employer",
        "Meridian Freight",
        lineage=SourceOrigin.known_origin,
        evidence=EvidenceKind.republished,
    )
    copied_but_observed = draft(
        "employer",
        "Meridian Freight",
        lineage=SourceOrigin.suspected_copy,
        evidence=EvidenceKind.direct_observation,
    )

    q_relayed = score_item(relayed, candidate_of(relayed), now=NOW)
    q_observed = score_item(
        copied_but_observed, candidate_of(copied_but_observed), now=NOW
    )

    # Reliability follows evidence_kind even when lineage says the opposite.
    assert q_observed.components["reliability"] > q_relayed.components["reliability"]


def test_unknown_evidence_kind_is_the_default() -> None:
    """An adapter that made no judgement is not credited with having made one."""
    bare = AssertionDraft(predicate="employer", raw_value="x", normalized_value="x")
    assert bare.evidence_kind is EvidenceKind.unknown

    config = ScoringConfig()
    q = score_item(bare, candidate_of(bare), now=NOW)
    assert q.components["reliability"] == pytest.approx(
        config.reliability_by_evidence_kind[EvidenceKind.unknown]
    )


# --- q is bounded for the same reason p is -------------------------------


def test_q_never_returns_one() -> None:
    """p was honest about its limits and q was not.

    Every input to q is an uncalibrated heuristic: reliability is a table of
    guesses about publishers, freshness is a decay curve nobody fitted, and
    corroboration counts websites. The product's whole claim is that it tells
    you how far to trust what it found, so a 1.00 on an item is the one number
    it may not print.
    """
    config = ScoringConfig()
    assert config.q_ceiling < 1.0

    # The strongest item the system can construct: a distinctive predicate,
    # observed today, directly, and agreed on by three independent origins.
    today = NOW
    shared = "+15125550001"
    drafts = [
        draft("phone", shared, origin=o, observed_at=today, publisher=f"{o}.example")
        for o in ("origin_a", "origin_b", "origin_c")
    ]
    candidate = candidate_of(*drafts)

    q = score_item(drafts[0], candidate, now=today)

    assert float(q) <= config.q_ceiling
    assert float(q) < 1.0


def test_r_cannot_claim_certainty_either() -> None:
    """r = p x q, so both ceilings bind it. Nothing downstream lifts it back."""
    config = ScoringConfig()
    ceiling = config.p_ceiling * config.q_ceiling
    assert derive_attribution(config.p_ceiling, config.q_ceiling) == pytest.approx(
        ceiling
    )
    assert ceiling < 1.0
