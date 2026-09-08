"""A block is a proposal. Only evidence makes it a candidate.

The failure these tests pin: a block keyed on a name alone was becoming one
candidate with no merge step, which reached the outcome the merge rule forbids
by skipping the rule entirely. On a live run it fused an Instagram account, a
university athlete, a realtor and a professional body into one p=1.0 person.
"""

from __future__ import annotations

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import (
    CandidateDraft,
    FormationConfig,
    form_candidates,
)
from app.engine.scoring import score_identity
from app.models import GroupingBasis, SourceOrigin

SUBJECT = Query(name="Michael Petrie", context="logistics operations Austin")


def page(ref: str, origin: str, *, city: str | None = None, email: str | None = None):
    """One search result about "Michael Petrie", from one site."""
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie",
            normalized_value="michael petrie",
            record_ref=ref,
            publisher=origin,
            origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        ),
        AssertionDraft(
            predicate="profile_url",
            raw_value=f"https://{origin}/p",
            normalized_value=f"https://{origin}/p",
            record_ref=ref,
            publisher=origin,
            origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        ),
    ]
    for predicate, value in (("city", city), ("email", email)):
        if value:
            drafts.append(
                AssertionDraft(
                    predicate=predicate,
                    raw_value=value,
                    normalized_value=value.casefold(),
                    record_ref=ref,
                    publisher=origin,
                    origin_key=origin,
                    source_origin=SourceOrigin.known_origin,
                )
            )
    return drafts


# --- 1. a block is not a candidate ----------------------------------------


def test_four_records_sharing_only_a_name_form_four_candidates() -> None:
    """The live failure, in miniature: four strangers, not one person."""
    drafts = [
        *page("r1", "instagram.com"),
        *page("r2", "theclm.org"),
        *page("r3", "gofrogs.com"),
        *page("r4", "coldwellbanker.com"),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 4
    assert all(
        c.grouping_basis is GroupingBasis.name_only for c in result.candidates
    )
    # Each holds exactly its own record.
    assert sorted(len(c.assertions) for c in result.candidates) == [2, 2, 2, 2]


def test_records_sharing_a_name_and_a_distinctive_attribute_do_combine() -> None:
    """Positive evidence still combines them. Blocking proposes, evidence disposes."""
    drafts = [
        *page("r1", "a.example", email="mp@example.com"),
        *page("r2", "b.example", email="mp@example.com"),
        *page("r3", "c.example"),
    ]

    result = form_candidates(drafts)

    combined = [c for c in result.candidates if len(c.assertions) > 3]
    assert len(combined) == 1
    assert combined[0].grouping_basis is GroupingBasis.distinctive_attribute
    assert {d.origin_key for d in combined[0].assertions} == {
        "a.example",
        "b.example",
    }
    # The third stands alone, on a name alone.
    alone = [c for c in result.candidates if c is not combined[0]]
    assert len(alone) == 1
    assert alone[0].grouping_basis is GroupingBasis.name_only


def test_a_shared_name_and_locality_still_groups() -> None:
    """Name plus locality is more than a name, so the clean case survives."""
    drafts = [
        *page("r1", "a.example", city="Austin, TX"),
        *page("r2", "b.example", city="Austin, TX"),
    ]

    result = form_candidates(drafts)

    assert len(result.candidates) == 1
    assert result.candidates[0].grouping_basis is GroupingBasis.name_and_locality


def test_the_split_is_configurable() -> None:
    drafts = [*page("r1", "a.example"), *page("r2", "b.example")]

    assert len(form_candidates(drafts).candidates) == 2

    permissive = form_candidates(
        drafts, config=FormationConfig(name_only_blocks_require_evidence=False)
    )
    assert len(permissive.candidates) == 1


# --- 2. p reflects what supports the grouping ------------------------------


def test_a_name_only_grouping_scores_materially_below_the_alternatives() -> None:
    name_only = CandidateDraft(
        name_key="michael petrie",
        grouping_basis=GroupingBasis.name_only,
        assertions=page("r1", "a.example"),
    )
    anchored = CandidateDraft(
        name_key="michael petrie",
        grouping_basis=GroupingBasis.anchor,
        assertions=page("r1", "a.example"),
    )
    by_attribute = CandidateDraft(
        name_key="michael petrie",
        grouping_basis=GroupingBasis.distinctive_attribute,
        assertions=page("r1", "a.example"),
    )

    p_weak = score_identity(name_only, SUBJECT)
    p_anchor = score_identity(anchored, SUBJECT)
    p_attribute = score_identity(by_attribute, SUBJECT)

    assert float(p_weak) < float(p_anchor)
    assert float(p_weak) < float(p_attribute)
    # Materially below, not marginally: less than half.
    assert float(p_weak) < 0.5 * float(p_anchor)
    assert p_weak.grouping_basis is GroupingBasis.name_only


def test_a_name_only_grouping_cannot_reach_one() -> None:
    """No amount of corroboration rescues a candidate held together by a name.

    The four origins are made to genuinely agree — one shared email — because
    corroboration now requires agreement. Four origins merely *present* no
    longer saturate anything, which is the subject of the test below.
    """
    shared = "m.petrie@example.com"
    drafts = [
        *page("r1", "a.example", email=shared),
        *page("r2", "b.example", email=shared),
        *page("r3", "c.example", email=shared),
        *page("r4", "d.example", email=shared),
    ]
    fused = CandidateDraft(
        name_key="michael petrie",
        grouping_basis=GroupingBasis.name_only,
        assertions=drafts,
    )
    p = score_identity(fused, SUBJECT)

    # Four agreeing origins saturate corroboration, and it still is not enough.
    assert p.components["corroboration"] == 1.0
    assert p.components["base"] == 1.0
    assert float(p) < 0.5


def test_strangers_who_share_only_a_name_do_not_corroborate_each_other() -> None:
    """The inversion, at its root.

    Corroboration counted the distinct origins present and never asked whether
    two of them agreed about anything. So a pile of same-name strangers grew
    more convincing with every stranger added: the signal meant to guard
    against coincidence was being fed by it.
    """
    def score(n: int) -> float:
        drafts = [
            d
            for i in range(n)
            for d in page(f"r{i}", f"origin{i}.example")
        ]
        return score_identity(
            CandidateDraft(
                name_key="michael petrie",
                grouping_basis=GroupingBasis.name_only,
                assertions=drafts,
            ),
            SUBJECT,
        ).components["corroboration"]

    assert score(1) == 0.0
    assert score(4) == 0.0
    assert score(12) == 0.0


def test_agreement_on_a_name_is_not_agreement() -> None:
    """Two origins that share a name and nothing else corroborate nothing."""
    drafts = [*page("r1", "a.example"), *page("r2", "b.example")]
    bare = score_identity(
        CandidateDraft(name_key="michael petrie", assertions=drafts), SUBJECT
    )
    assert bare.components["corroboration"] == 0.0

    # Give the same two origins one thing to agree on, and it counts.
    agreeing = [
        *page("r1", "a.example", email="m.petrie@example.com"),
        *page("r2", "b.example", email="m.petrie@example.com"),
    ]
    corroborated = score_identity(
        CandidateDraft(name_key="michael petrie", assertions=agreeing), SUBJECT
    )
    assert corroborated.components["corroboration"] > 0.0


def test_agreement_on_a_locality_is_not_agreement() -> None:
    """A shared city is what put them in one pile; it cannot also confirm it."""
    drafts = [
        *page("r1", "a.example", city="Austin, TX"),
        *page("r2", "b.example", city="Austin, TX"),
    ]
    p = score_identity(
        CandidateDraft(name_key="michael petrie", assertions=drafts), SUBJECT
    )
    assert p.components["corroboration"] == 0.0


def test_the_basis_factor_is_configurable_and_exposed() -> None:
    from app.engine.scoring import ScoringConfig

    candidate = CandidateDraft(
        name_key="michael petrie",
        grouping_basis=GroupingBasis.name_only,
        assertions=page("r1", "a.example"),
    )
    default = score_identity(candidate, SUBJECT)
    assert default.components["grouping_basis_factor"] == pytest.approx(0.40)

    lenient = score_identity(
        candidate,
        SUBJECT,
        config=ScoringConfig(
            grouping_basis_factor={GroupingBasis.name_only: 1.0}
        ),
    )
    assert float(lenient) > float(default)


def test_an_unspecified_basis_is_neutral() -> None:
    """A hand-built candidate is not penalised for a judgement nobody made."""
    candidate = CandidateDraft(
        name_key="michael petrie", assertions=page("r1", "a.example")
    )
    p = score_identity(candidate, SUBJECT)
    assert candidate.grouping_basis is GroupingBasis.unspecified
    assert p.components["grouping_basis_factor"] == 1.0


# --- 3. the anchor leads ---------------------------------------------------


def test_the_anchor_candidate_sorts_first() -> None:
    """Even when a blind candidate scores higher on its own signals."""
    drafts = [
        *page("anchor-1", "linkedin.com"),
        *page("blind-1", "gofrogs.com", city="Fort Worth, TX"),
        *page("blind-2", "coldwellbanker.com", city="Fort Worth, TX"),
    ]
    context = {
        "anchor-1": "Logistics operations lead based in Austin.",
        # Same name, nothing else in common: these fail the anchor and cluster
        # blindly on name plus locality, which scores well on its own signals.
        "blind-1": "Outfielder, university baseball roster.",
        "blind-2": "Residential real estate agent, listings and open houses.",
    }

    result = form_candidates(drafts, subject=SUBJECT, record_context=context)

    ranked = sorted(
        result.candidates,
        key=lambda c: (c.is_anchor, float(score_identity(c, SUBJECT))),
        reverse=True,
    )
    assert ranked[0].is_anchor is True
    assert ranked[0].grouping_basis is GroupingBasis.anchor
    assert any(not c.is_anchor for c in ranked[1:])


def test_anchor_status_reaches_the_score() -> None:
    assert score_identity(
        CandidateDraft(
            name_key="michael petrie",
            grouping_basis=GroupingBasis.anchor,
            assertions=page("r1", "a.example"),
        ),
        SUBJECT,
    ).grouping_basis is GroupingBasis.anchor


# --- 4. a thin anchor match is not enough ---------------------------------


def test_a_partial_name_with_no_other_signal_does_not_anchor() -> None:
    """Half a name and nothing else is a coincidence of words."""
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie Associates",
            normalized_value="michael petrie associates",
            record_ref="thin-1",
            origin_key="unknown.example",
        )
    ]

    result = form_candidates(drafts, subject=SUBJECT, record_context={})

    match = result.anchor_matches["thin-1"]
    assert match.strength > FormationConfig().anchor_threshold
    assert match.has_link is False
    assert match.context_signal is None
    assert match.admitted is False
    assert match.rejected_because == "name only — no other signal"
    assert result.anchor_candidate is None


def test_a_link_does_not_admit_the_same_record() -> None:
    """A link used to be enough. It never should have been.

    Practically every real web result has a link, so the rule admitted almost
    everything; but the reason to drop it is not that it was loose. A page
    existing says nothing about which human it describes, and only evidence
    about the human can answer the question the anchor asks.
    """
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie Associates",
            normalized_value="michael petrie associates",
            record_ref="linked-1",
            origin_key="a.example",
        ),
        AssertionDraft(
            predicate="profile_url",
            raw_value="https://a.example/p",
            normalized_value="https://a.example/p",
            record_ref="linked-1",
            origin_key="a.example",
        ),
    ]

    result = form_candidates(drafts, subject=SUBJECT, record_context={})
    match = result.anchor_matches["linked-1"]
    assert match.has_link is True
    assert match.admitted is False
    assert match.rejected_because == "name only — no other signal"
    assert result.anchor_candidate is None


def test_the_same_record_anchors_once_it_has_context() -> None:
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie Associates",
            normalized_value="michael petrie associates",
            record_ref="ctx-1",
            origin_key="a.example",
        )
    ]
    context = {"ctx-1": "Logistics operations consultancy in Austin."}

    result = form_candidates(drafts, subject=SUBJECT, record_context=context)
    match = result.anchor_matches["ctx-1"]
    assert match.context_signal > 0
    assert match.admitted is True
    assert result.anchor_candidate is not None


def test_the_substance_requirement_is_configurable() -> None:
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie Associates",
            normalized_value="michael petrie associates",
            record_ref="thin-1",
            origin_key="a.example",
        )
    ]

    strict = form_candidates(drafts, subject=SUBJECT, record_context={})
    assert strict.anchor_candidate is None

    lax = form_candidates(
        drafts,
        config=FormationConfig(anchor_requires_independent_signal=False),
        subject=SUBJECT,
        record_context={},
    )
    assert lax.anchor_candidate is not None
