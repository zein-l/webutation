"""Context is counted to a saturation point, not divided by what was typed.

The share form asked what fraction of the caller's context terms a record
carried. That scale is unreachable — a search snippet is about 150 characters
and cannot hold six context terms; across the whole Michael Petrie corpus the
highest context signal ever observed was 0.50, while being compared against a
0.65 threshold as though 1.00 were attainable.

It also penalised precision, which is the defect these tests exist to prevent
returning: the same record, on identical evidence, scored 1.00 for a caller who
typed "Webutation" and 0.17 for one who typed "Webutation, private investigator,
insurance fraud, OSINT".

The value 2.0 is on precedent, not measurement. See the note on
``anchor_context_saturation`` before trusting it.
"""

from __future__ import annotations

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import FormationConfig, form_candidates
from app.models import SourceOrigin

CONFIG = FormationConfig()
NAME = "Michael Petrie"


def record(ref: str, origin: str = "a.example") -> list[AssertionDraft]:
    def draft(predicate: str, value: str) -> AssertionDraft:
        return AssertionDraft(
            predicate=predicate, raw_value=value,
            normalized_value=value.casefold(), record_ref=ref,
            publisher=origin, origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        )

    return [draft("name", NAME), draft("profile_url", f"https://{origin}/p")]


def signal(context: str, snippet: str, config: FormationConfig = CONFIG) -> float | None:
    result = form_candidates(
        record("r1"),
        config=config,
        subject=Query(name=NAME, context=context),
        record_context={"r1": snippet},
    )
    return result.anchor_matches["r1"].context_signal


# --- the defect that prompted the change ---------------------------------


def test_typing_more_context_does_not_weaken_the_same_evidence() -> None:
    """The clinching case: one matched term, four different descriptions.

    Under the share form this record scored 1.00, 0.50, 0.33 and 0.17 as the
    caller grew more specific, and was admitted only in the first two. The
    evidence never changed; only the caller's verbosity did.
    """
    snippet = "Founder-CEO at Webutation, Inc, based in Philadelphia, PA."
    scores = [
        signal("Webutation", snippet),
        signal("Webutation, OSINT", snippet),
        signal("Webutation, OSINT, insurance", snippet),
        signal("Webutation, private investigator, insurance fraud, OSINT", snippet),
    ]

    assert len(set(scores)) == 1, f"one matched term must score the same: {scores}"
    assert scores[0] == 0.5


def test_the_scale_is_reachable() -> None:
    """Full agreement must be attainable by a real snippet, or 1.00 is a fiction."""
    rich = "Webutation, private investigator, insurance fraud, OSINT"
    assert signal(rich, "Michael Petrie, private investigator, insurance fraud") == 1.0


# --- the counting rule ---------------------------------------------------


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ("nothing whatsoever in common here", 0.0),
        ("a note about logistics", 0.5),
        ("logistics and freight together", 1.0),
        ("logistics freight haulage all three", 1.0),
    ],
)
def test_matches_count_toward_saturation(snippet: str, expected: float) -> None:
    assert signal("logistics freight haulage", snippet) == expected


def test_saturation_is_configurable() -> None:
    snippet = "a note about logistics"
    assert signal("logistics freight", snippet, FormationConfig(anchor_context_saturation=1.0)) == 1.0
    assert signal("logistics freight", snippet, FormationConfig(anchor_context_saturation=2.0)) == 0.5
    assert signal("logistics freight", snippet, FormationConfig(anchor_context_saturation=4.0)) == 0.25


def test_a_zero_saturation_means_any_match_is_full_agreement() -> None:
    """Guards the division rather than leaving it to raise."""
    config = FormationConfig(anchor_context_saturation=0.0)
    assert signal("logistics freight", "a note about logistics", config) == 1.0
    assert signal("logistics freight", "nothing in common", config) == 0.0


def test_silence_is_still_not_disagreement() -> None:
    """None where no comparison was possible; 0.0 where one was and failed."""
    assert signal("", "some text") is None
    assert signal("logistics", "") is None
    assert signal("logistics", "unrelated words entirely") == 0.0


# --- the documented sensitivity, asserted -------------------------------


def test_the_reported_false_negative_turns_on_the_saturation_value() -> None:
    """The sensitivity recorded on the config field, kept honest by a test.

    The subject's own RocketReach page carries the name in a page title and
    matches exactly one context term. Whether that is admitted depends on a
    constant chosen by precedent rather than measured, so the dependence is
    stated here rather than left for someone to rediscover.
    """
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie Email & Phone Number | Webutation, Inc",
            normalized_value="michael petrie email & phone number | webutation, inc",
            record_ref="r1", publisher="rocketreach.co", origin_key="rocketreach.co",
            source_origin=SourceOrigin.known_origin,
        ),
        AssertionDraft(
            predicate="profile_url", raw_value="https://rocketreach.co/x",
            normalized_value="https://rocketreach.co/x", record_ref="r1",
            publisher="rocketreach.co", origin_key="rocketreach.co",
            source_origin=SourceOrigin.known_origin,
        ),
    ]
    subject = Query(
        name=NAME, context="Webutation, private investigator, insurance fraud, OSINT"
    )
    snippet = {
        "r1": "Michael Petrie, based in Philadelphia, PA, US, is currently a "
              "Founder-CEO at Webutation, Inc."
    }

    def strength(saturation: float) -> float:
        result = form_candidates(
            drafts,
            config=FormationConfig(anchor_context_saturation=saturation),
            subject=subject,
            record_context=snippet,
        )
        return result.anchor_matches["r1"].strength

    at_two, at_three = strength(2.0), strength(3.0)

    assert at_two == pytest.approx(0.70, abs=0.005)
    assert at_two > CONFIG.anchor_threshold, "admitted at the shipped value"
    assert at_three == pytest.approx(0.63, abs=0.005)
    assert at_three < CONFIG.anchor_threshold, "and not at 3.0 — the value is load-bearing"


# --- the regression saturation caused, and the rule that closes it --------


def test_context_alone_cannot_anchor_a_subject_with_no_name() -> None:
    """Wikipedia's article on private investigators is not a person.

    Saturation removed an accidental protection. Under the share form, two
    matched terms out of six scored 0.33 and never cleared the threshold; once
    matches counted toward saturation, "private investigator" and "insurance
    fraud" were two terms each and reached 1.00 with context as the only
    signal. Eleven such records joined a p=0.80 anchor, among them a Reddit
    thread and a press release about an unrelated man.
    """
    drafts = [
        AssertionDraft(
            predicate="profile_url",
            raw_value="https://en.wikipedia.org/wiki/Private_investigator",
            normalized_value="https://en.wikipedia.org/wiki/private_investigator",
            record_ref="r1", publisher="wikipedia.org", origin_key="wikipedia.org",
            source_origin=SourceOrigin.known_origin,
        )
    ]
    result = form_candidates(
        drafts,
        subject=Query(context="Webutation, private investigator, insurance fraud, OSINT"),
        record_context={"r1": "A private investigator is a person who can be hired."},
    )

    match = result.anchor_matches["r1"]
    assert match.context_signal == 1.0, "two terms still saturate the signal"
    assert match.admitted is False
    assert match.rejected_because == "no name to match, and a trade is not an identity"
    assert result.anchor_candidate is None


def test_a_face_still_anchors_a_subject_with_no_name() -> None:
    """The rule bars text, not evidence about the person.

    A photo-only search must keep working: a face is evidence about a human
    being, which is the one thing that earns admission on its own.
    """
    class Face:
        best_similarity = 0.99
        ambiguous = False

    drafts = [
        AssertionDraft(
            predicate="image_url", raw_value="https://a.example/p.png",
            normalized_value="https://a.example/p.png", record_ref="r1",
            publisher="a.example", origin_key="a.example",
            source_origin=SourceOrigin.known_origin,
        )
    ]
    result = form_candidates(
        drafts,
        subject=Query(photo_url="https://x.test/s.png"),
        record_context={},
        face_match=lambda _drafts: Face(),
    )

    assert result.anchor_matches["r1"].admitted is True
    assert result.anchor_candidate is not None


def test_the_name_bearing_case_is_unaffected() -> None:
    """The repair must not undo what the saturation change was for."""
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie Email & Phone Number | Webutation, Inc",
            normalized_value="michael petrie email & phone number | webutation, inc",
            record_ref="r1", publisher="rocketreach.co", origin_key="rocketreach.co",
            source_origin=SourceOrigin.known_origin,
        )
    ]
    result = form_candidates(
        drafts,
        subject=Query(
            name=NAME,
            context="Webutation, private investigator, insurance fraud, OSINT",
        ),
        record_context={"r1": "Founder-CEO at Webutation, Inc, Philadelphia, PA."},
    )

    assert result.anchor_matches["r1"].admitted is True
