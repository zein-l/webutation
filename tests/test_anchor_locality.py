"""A locality has to name a place before it can be evidence of one.

The first fix for "a name is never enough" required a signal that is not the
name, and offered a locality as one of them. Locality agreement was then a token
intersection, which meant "Austin, TX" agreed with "Houston, TX" on the state —
so every same-name person in one state joined a single anchor. The original bug,
one notch up, in the same place.

This file pins the comparison down. It is not incidental: the module already
excludes locality from DISTINCTIVE_PREDICATES on the grounds that "large numbers
of unrelated people share" it, and a comparator that matches on a state is
asserting exactly what that exclusion denies.
"""

from __future__ import annotations

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import (
    FormationConfig,
    _anchor_locality_signal,
    form_candidates,
)
from app.engine.scoring import score_identity
from app.models import SourceOrigin

CONFIG = FormationConfig()


class Locality:
    """Stands in for a record, which is all the comparator reads."""

    def __init__(self, key: str | None) -> None:
        self.locality_key = key


def signal(subject: str | None, record: str | None) -> float | None:
    return _anchor_locality_signal(Locality(record), subject, CONFIG)


def record(
    ref: str, origin: str, city: str | None = None, name: str = "Michael Petrie"
) -> list[AssertionDraft]:
    def draft(predicate: str, value: str) -> AssertionDraft:
        return AssertionDraft(
            predicate=predicate, raw_value=value,
            normalized_value=value.casefold(), record_ref=ref,
            publisher=origin, origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        )

    drafts = [draft("name", name), draft("profile_url", f"https://{origin}/p")]
    if city:
        drafts.append(draft("city", city))
    return drafts


# --- the settlement is what identifies a place -----------------------------


@pytest.mark.parametrize(
    ("subject", "rec"),
    [
        ("austin tx", "austin texas"),      # the postal code and the name agree
        ("austin", "austin tx"),            # one side omits the state
        ("100 main st springfield il", "springfield il"),  # a street address
        ("los angeles ca", "los angeles ca"),
        ("new york ny", "new york ny"),
    ],
)
def test_the_same_settlement_matches(subject: str, rec: str) -> None:
    assert signal(subject, rec) == CONFIG.anchor_locality_match


@pytest.mark.parametrize(
    ("subject", "rec", "why"),
    [
        ("austin tx", "houston tx", "same state, different cities"),
        ("boston ma", "austin tx", "nothing in common"),
        ("new york ny", "new orleans la", "they share only the word 'new'"),
        ("san francisco ca", "san diego ca", "they share only the word 'san'"),
        ("100 main st springfield il", "st louis mo", "they share only 'st'"),
        ("springfield il", "springfield north carolina", "different states"),
    ],
)
def test_different_settlements_do_not_match(subject: str, rec: str, why: str) -> None:
    assert signal(subject, rec) == CONFIG.anchor_locality_mismatch, why


@pytest.mark.parametrize(("subject", "rec"), [("austin tx", "tx"), ("dallas tx", "texas")])
def test_a_shared_state_alone_is_reported_as_a_shared_state(
    subject: str, rec: str
) -> None:
    """Weakly confirmatory, and deliberately below the admitting threshold."""
    value = signal(subject, rec)
    assert value == CONFIG.anchor_locality_region_only
    assert 0 < value < CONFIG.anchor_locality_admits


def test_silence_on_either_side_is_not_a_comparison() -> None:
    assert signal(None, "austin tx") is None
    assert signal("austin tx", None) is None


# --- and that is what admission turns on -----------------------------------


def test_a_shared_state_does_not_admit_a_record() -> None:
    """The regression the first fix introduced, stated directly."""
    drafts = record("r1", "a.example", city="TX")
    result = form_candidates(
        drafts, subject=Query(name="Michael Petrie", address="Austin, TX"),
        record_context={},
    )

    match = result.anchor_matches["r1"]
    assert match.locality_signal == CONFIG.anchor_locality_region_only
    assert match.admitted is False
    assert result.anchor_candidate is None


def test_six_men_in_six_texas_cities_do_not_share_an_anchor() -> None:
    """Reproduced against the first fix at p = 0.95. One of them is the subject."""
    cities = ["Austin, TX", "Houston, TX", "Dallas, TX",
              "El Paso, TX", "Lubbock, TX", "Waco, TX"]
    drafts = [
        d
        for i, city in enumerate(cities)
        for d in record(f"r{i}", f"origin{i}.example", city=city)
    ]
    subject = Query(name="Michael Petrie", address="Austin, TX")

    result = form_candidates(drafts, subject=subject, record_context={})

    anchor = result.anchor_candidate
    assert anchor is not None
    assert len(anchor.anchor_matches) == 1
    assert "r0" in anchor.anchor_matches

    for ref in ("r1", "r2", "r3", "r4", "r5"):
        assert result.anchor_matches[ref].admitted is False


def test_a_locality_that_contradicts_disqualifies_however_good_the_context() -> None:
    """A disagreement is a statement, not a weak signal.

    Diluted in the mean, a strong context term carried a record that says Boston
    into an anchor for a subject in Austin.
    """
    drafts = record("r1", "a.example", city="Boston, MA")
    result = form_candidates(
        drafts,
        subject=Query(
            name="Michael Petrie", address="Austin, TX",
            context="logistics freight",
        ),
        record_context={"r1": "logistics freight"},
    )

    match = result.anchor_matches["r1"]
    # Two matched terms, so the context is as strong as context gets. The point
    # is that even at full strength it does not carry a contradicted locality.
    assert match.context_signal == 1.0
    assert match.locality_signal == CONFIG.anchor_locality_mismatch
    assert match.admitted is False
    assert match.rejected_because == "locality contradicts the subject"


def test_a_record_without_the_subjects_name_does_not_anchor_on_context() -> None:
    """"Name plus an independent signal" requires the name to be there."""
    drafts = [
        AssertionDraft(
            predicate="profile_url", raw_value="https://a.example/p",
            normalized_value="https://a.example/p", record_ref="r1",
            origin_key="a.example", source_origin=SourceOrigin.known_origin,
        )
    ]
    result = form_candidates(
        drafts,
        subject=Query(name="Michael Petrie", context="logistics freight"),
        record_context={"r1": "logistics freight"},
    )

    match = result.anchor_matches["r1"]
    assert match.name_signal is None
    assert match.context_signal == 1.0
    assert match.admitted is False
    assert match.rejected_because == "does not carry the subject's name"


def test_a_different_persons_name_does_not_anchor_on_context() -> None:
    drafts = record("r1", "a.example", name="Someone Else")
    result = form_candidates(
        drafts,
        subject=Query(name="Michael Petrie", context="logistics"),
        record_context={"r1": "logistics"},
    )
    assert result.anchor_matches["r1"].admitted is False
    assert result.anchor_candidate is None


def test_a_genuine_locality_match_still_anchors() -> None:
    """The rule must not cost the anchor a real one."""
    drafts = record("r1", "a.example", city="Austin, Texas")
    result = form_candidates(
        drafts, subject=Query(name="Michael Petrie", address="Austin, TX"),
        record_context={},
    )

    match = result.anchor_matches["r1"]
    assert match.locality_signal == CONFIG.anchor_locality_match
    assert match.admitted is True
    assert result.anchor_candidate is not None


# --- p stops scoring the subject's name against itself ---------------------


def test_p_uses_the_name_agreement_that_was_measured() -> None:
    """An anchor's name_key is the subject's, so comparing them scored 1.00.

    The record is titled "Michael Petrie Email and Phone Number"; anchoring
    measured that at 0.85. Scoring it as a perfect name match hands p evidence
    nobody produced.
    """
    drafts = record("r1", "a.example", name="Michael Petrie Email and Phone Number")
    subject = Query(name="Michael Petrie", context="logistics")
    result = form_candidates(
        drafts, subject=subject, record_context={"r1": "logistics"}
    )

    anchor = result.anchor_candidate
    assert anchor is not None
    measured = result.anchor_matches["r1"].name_signal
    assert measured == CONFIG.anchor_name_subject_contained

    p = score_identity(anchor, subject)
    assert p.components["name"] == pytest.approx(measured)
    assert p.components["name"] != 1.0


def test_the_weakest_member_sets_the_anchors_name_signal() -> None:
    """p asks whether these are one person. The least convincing member decides."""
    drafts = [
        *record("r1", "a.example", name="Michael Petrie"),
        *record("r2", "b.example", name="About Michael Petrie"),
    ]
    subject = Query(name="Michael Petrie", context="logistics")
    result = form_candidates(
        drafts, subject=subject,
        record_context={"r1": "logistics", "r2": "logistics"},
    )

    anchor = result.anchor_candidate
    signals = [m.name_signal for m in anchor.anchor_matches.values()]
    assert min(signals) < max(signals)

    p = score_identity(anchor, subject)
    assert p.components["name"] == pytest.approx(min(signals))
