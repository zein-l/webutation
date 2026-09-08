"""The weakest door left into the anchor, stated so nobody assumes otherwise.

A record joins the anchor on the subject's name plus one other signal. Context
overlap is one of those signals, and any overlap above zero counts — so a single
coincidentally shared word satisfies the rule. These tests pin that behaviour on
purpose. They are a description, not an endorsement: if the rule is ever
tightened, they should fail and be rewritten.

A frequency-weighted version was tried and reverted. See ``_anchor_context_signal``.
"""

from __future__ import annotations

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates
from app.models import SourceOrigin

NAME = "Michael Petrie"


def record(ref: str, origin: str) -> list[AssertionDraft]:
    def draft(predicate: str, value: str) -> AssertionDraft:
        return AssertionDraft(
            predicate=predicate, raw_value=value,
            normalized_value=value.casefold(), record_ref=ref,
            publisher=origin, origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        )

    return [draft("name", NAME), draft("profile_url", f"https://{origin}/p")]


def test_one_shared_word_admits_a_record() -> None:
    """The door. A stranger's page sharing "operations" joins the anchor.

    The name requirement bounds the damage to people who really do share the
    subject's name, which is what keeps this from being the original bug. It is
    still the loosest admission rule in the system.

    One matched term is half of ``anchor_context_saturation``, so it is scored
    as the partial evidence it is rather than as a share of however many terms
    the caller happened to type.
    """
    result = form_candidates(
        record("r1", "a.example"),
        subject=Query(name=NAME, context="logistics operations Austin"),
        record_context={"r1": "Operations manager at a car dealership in Miami."},
    )

    match = result.anchor_matches["r1"]
    assert match.context_signal == 0.5
    assert match.admitted is True


def test_sharing_nothing_still_refuses() -> None:
    """Zero overlap is evidence against, and the rule does hold there.

    This one never reaches the substance check: scoring the disagreement into
    the mean drops it below the threshold first, so ``rejected_because`` stays
    None and the report explains it as a threshold decision.
    """
    result = form_candidates(
        record("r1", "a.example"),
        subject=Query(name=NAME, context="logistics operations Austin"),
        record_context={"r1": "A pastry chef known for laminated dough."},
    )

    match = result.anchor_matches["r1"]
    assert match.context_signal == 0.0
    assert match.admitted is False
    assert match.rejected_because is None
    assert result.anchor_candidate is None


def test_frequency_weighting_would_break_a_real_group() -> None:
    """Why the obvious fix is not in place.

    Eight pages genuinely about one person share their whole vocabulary. Under a
    document-frequency rule every shared term scores as common and the anchor
    loses all eight — a worse error than admitting one stranger, and the reason
    the rarity estimate has to come from outside the result set.
    """
    snippet = "A mathematician who wrote notes on the analytical engine."
    drafts = [d for i in range(8) for d in record(f"r{i}", f"site{i}.example")]
    result = form_candidates(
        drafts,
        subject=Query(name=NAME, context="mathematician analytical engine"),
        record_context={f"r{i}": snippet for i in range(8)},
    )

    assert result.anchor_candidate is not None
    assert len(result.anchor_candidate.anchor_matches) == 8
