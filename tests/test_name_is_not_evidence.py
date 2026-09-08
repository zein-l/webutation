"""A shared name may propose a comparison. It may never settle one.

This is the rule the rest of the system already obeyed and the anchor path did
not. On a name+photo run with no context, twelve records anchored on name 1.00
or name 0.85 alone, and the anchor candidate ended up holding a college baseball
roster, a real estate agent, an obituary, a university professor and an
insider-trading filing: six different men under one name, at p = 1.00.

Two things made that possible and both are tested here.

The strength is a weighted mean over the signals that could be *computed*. An
unavailable signal is dropped rather than scored zero, which is right — silence
is not dissent — but it means a record whose only comparable signal is its name
scores exactly its name similarity. An exact match on a common name reads 1.00
and looks like certainty.

And corroboration counts distinct origins that agree, so six different websites
each describing a different Michael Petrie corroborated one another to the top of
the scale. A common name scored *higher* than a rare one.
"""

from __future__ import annotations

import math

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import FormationConfig, form_candidates
from app.engine.faces import FaceEmbedding, FaceMatcher
from app.engine.scoring import ScoringConfig, score_identity
from app.models import SourceOrigin

NAME = "Michael Petrie"
NAME_KEY = "michael petrie"


def unit(*values: float) -> FaceEmbedding:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return FaceEmbedding(vector=tuple(v / norm for v in values), det_score=0.99)


SUBJECT_FACE = unit(1.0, 0.0, 0.0)
SAME_PERSON = unit(0.995, 0.1, 0.0)
STRANGER = unit(0.0, 1.0, 0.0)


def record(
    ref: str,
    origin: str,
    *,
    name: str = NAME,
    city: str | None = None,
    image: str | None = None,
    link: bool = True,
) -> list[AssertionDraft]:
    """One web result about somebody called Michael Petrie."""

    def draft(predicate: str, value: str) -> AssertionDraft:
        return AssertionDraft(
            predicate=predicate,
            raw_value=value,
            normalized_value=value.casefold(),
            record_ref=ref,
            publisher=origin,
            origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        )

    drafts = [draft("name", name)]
    if link:
        drafts.append(draft("profile_url", f"https://{origin}/p"))
    if city:
        drafts.append(draft("city", city))
    if image:
        drafts.append(draft("image_url", image))
    return drafts


def matcher(faces: dict[str, list[FaceEmbedding]]) -> FaceMatcher:
    return FaceMatcher(
        subject_photo_url="subject.jpg",
        embedder=lambda body: faces.get(body.decode(), []),
        fetch=lambda url: url.encode() if url in faces else None,
    )


# --- the four the rule was stated as ---------------------------------------


def test_a_record_matching_only_on_name_does_not_anchor() -> None:
    """The one that matters. Name 1.00, nothing else, refused."""
    drafts = record("r1", "rocketreach.example")
    result = form_candidates(
        drafts, subject=Query(name=NAME), record_context={}
    )

    match = result.anchor_matches["r1"]
    assert match.name_signal == 1.0
    assert match.strength == 1.0
    assert match.strength > FormationConfig().anchor_threshold
    assert match.admitted is False
    assert match.rejected_because == "name only — no other signal"
    assert result.anchor_candidate is None


def test_a_name_plus_a_face_anchors() -> None:
    drafts = record("r1", "rocketreach.example", image="them.jpg")
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    result = form_candidates(
        drafts,
        subject=Query(name=NAME, photo_url="https://x.test/s.png"),
        record_context={},
        face_match=faces.match_record,
    )

    anchor = result.anchor_candidate
    assert anchor is not None
    match = result.anchor_matches["r1"]
    assert match.admitted is True
    assert match.face_signal >= FormationConfig().anchor_face_similarity


def test_a_name_plus_context_anchors() -> None:
    drafts = record("r1", "rocketreach.example")
    result = form_candidates(
        drafts,
        subject=Query(name=NAME, context="logistics operations"),
        record_context={"r1": "director of logistics"},
    )

    match = result.anchor_matches["r1"]
    assert match.context_signal > 0
    assert match.admitted is True
    assert result.anchor_candidate is not None


def test_p_never_returns_one() -> None:
    """Uncalibrated heuristics do not add up to certainty.

    The page prints "an uncalibrated estimate, not a probability" directly
    beneath the figure. A 1.00 there contradicts the sentence next to it.
    """
    ceiling = ScoringConfig().p_ceiling
    assert ceiling < 1.0

    # The strongest case the system can construct: exact name, agreeing
    # locality, a near-certain face, and corroboration from many origins.
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})
    drafts = [
        d
        for i in range(8)
        for d in record(f"r{i}", f"origin{i}.example", city="Austin, TX", image="them.jpg")
    ]
    subject = Query(
        name=NAME,
        address="Austin, TX",
        context="logistics operations",
        photo_url="https://x.test/s.png",
    )
    result = form_candidates(
        drafts,
        subject=subject,
        record_context={f"r{i}": "logistics operations" for i in range(8)},
        face_match=faces.match_record,
    )

    for candidate in result.candidates:
        p = score_identity(candidate, subject)
        assert float(p) <= ceiling
        assert float(p) != 1.0


# --- the case that was actually on screen ----------------------------------


def test_six_different_men_do_not_become_one_candidate() -> None:
    """The reported bug, as reported: name+photo, no context."""
    people = [
        ("roster", "gobearcats.example"),
        ("realtor", "realtor.example"),
        ("obituary", "legacy.example"),
        ("professor", "university.example"),
        ("insider", "secform4.example"),
        ("aggregator", "rocketreach.example"),
    ]
    drafts = [d for ref, origin in people for d in record(ref, origin)]
    subject = Query(name=NAME, photo_url="https://x.test/s.png")

    result = form_candidates(drafts, subject=subject, record_context={})

    assert result.anchor_candidate is None
    for ref, _ in people:
        assert result.anchor_matches[ref].admitted is False
        assert result.anchor_matches[ref].rejected_because == (
            "name only — no other signal"
        )

    # And nothing downstream reassembles them into one confident person.
    assert max(float(score_identity(c, subject)) for c in result.candidates) < 0.5


def test_more_strangers_sharing_a_name_no_longer_raises_confidence() -> None:
    """The corroboration inversion, stated as a test.

    Corroboration counts distinct origins that agree. When the records are
    different people who merely share a name, agreeing is exactly what they do,
    so a common name used to score higher than a rare one. Adding strangers must
    not now raise anyone's score.
    """
    subject = Query(name=NAME, photo_url="https://x.test/s.png")

    def best_p(count: int) -> float:
        drafts = [
            d for i in range(count) for d in record(f"r{i}", f"origin{i}.example")
        ]
        result = form_candidates(drafts, subject=subject, record_context={})
        return max(float(score_identity(c, subject)) for c in result.candidates)

    few, many = best_p(2), best_p(12)
    assert many <= few + 1e-9
    assert many < 0.5


# --- adversarial edges around the new rule ---------------------------------


def test_a_face_below_threshold_does_not_rescue_a_name(monkeypatch) -> None:
    """A weak face is not "some" evidence that tops up a name.

    A stranger's face is evidence *against*, and a face that cannot be resolved
    is no evidence at all. Neither is a reason to admit.
    """
    drafts = record("r1", "rocketreach.example", image="them.jpg")
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [STRANGER]})

    result = form_candidates(
        drafts,
        subject=Query(name=NAME, photo_url="https://x.test/s.png"),
        record_context={},
        face_match=faces.match_record,
    )

    match = result.anchor_matches["r1"]
    assert match.face_signal < FormationConfig().anchor_face_similarity
    assert match.admitted is False
    assert result.anchor_candidate is None


def test_a_name_plus_an_agreeing_locality_anchors() -> None:
    drafts = record("r1", "rocketreach.example", city="Austin, TX")
    result = form_candidates(
        drafts, subject=Query(name=NAME, address="Austin, Texas"), record_context={}
    )

    match = result.anchor_matches["r1"]
    assert match.locality_signal == FormationConfig().anchor_locality_match
    assert match.admitted is True
    assert result.anchor_candidate is not None


def test_a_name_plus_a_contradicting_locality_does_not_anchor() -> None:
    """A locality that disagrees is worse than one that is missing."""
    drafts = record("r1", "rocketreach.example", city="Boston, MA")
    result = form_candidates(
        drafts, subject=Query(name=NAME, address="Austin, TX"), record_context={}
    )

    match = result.anchor_matches["r1"]
    assert match.locality_signal == FormationConfig().anchor_locality_mismatch
    assert match.admitted is False
    assert result.anchor_candidate is None


def test_a_missing_locality_is_not_read_as_agreement() -> None:
    """Silence is not assent. A record with no city has not said Austin."""
    drafts = record("r1", "rocketreach.example")
    result = form_candidates(
        drafts, subject=Query(name=NAME, address="Austin, TX"), record_context={}
    )

    match = result.anchor_matches["r1"]
    assert match.locality_signal is None
    assert match.admitted is False


def test_a_name_only_search_produces_no_anchor_at_all() -> None:
    """Deliberate. With only a name supplied there is nothing to verify against.

    An empty anchor says that. A confident one would not.
    """
    drafts = [
        d
        for i in range(5)
        for d in record(f"r{i}", f"origin{i}.example")
    ]
    result = form_candidates(drafts, subject=Query(name=NAME), record_context={})

    assert result.anchor_available is True
    assert result.anchor_candidate is None
    assert all(not m.admitted for m in result.anchor_matches.values())


def test_a_photo_only_run_still_anchors_on_the_face_alone() -> None:
    """The new rule must not cost the face its standing.

    A face is evidence about the human rather than about a label, which is why
    it admits with no name and no context. Nothing else earns that.
    """
    drafts = record("r1", "rocketreach.example", name="Someone Else", image="them.jpg")
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    result = form_candidates(
        drafts,
        subject=Query(photo_url="https://x.test/s.png"),
        record_context={},
        face_match=faces.match_record,
    )

    assert result.anchor_candidate is not None
    assert result.anchor_matches["r1"].admitted is True
    assert result.anchor_matches["r1"].name_signal is None


def test_the_rejection_says_which_comparison_was_missing() -> None:
    """The reason has to name the absent signal, not just refuse."""
    from app.report import _rejection

    drafts = record("r1", "rocketreach.example")
    result = form_candidates(drafts, subject=Query(name=NAME), record_context={})
    payload = _rejection("r1", result.anchor_matches["r1"], FormationConfig())

    assert payload["decided_by"] == "substance"
    assert payload["reason"] == "name only — no other signal"
    # The old wording promised a rule that no longer exists.
    assert "page behind it" not in payload["explanation"]
    assert "link" not in payload["explanation"]


# --- the cap does not quietly become the score -----------------------------


def test_the_cap_bounds_p_without_flattening_it() -> None:
    """0.95 is a ceiling, not a default. Weaker cases must still score lower."""
    config = ScoringConfig()
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    strong_subject = Query(
        name=NAME, address="Austin, TX", context="logistics",
        photo_url="https://x.test/s.png",
    )
    strong = form_candidates(
        [d for i in range(6) for d in record(f"r{i}", f"o{i}.example",
                                             city="Austin, TX", image="them.jpg")],
        subject=strong_subject,
        record_context={f"r{i}": "logistics" for i in range(6)},
        face_match=faces.match_record,
    )
    weak_subject = Query(name=NAME, context="logistics")
    weak = form_candidates(
        record("r1", "o.example"),
        subject=weak_subject,
        record_context={"r1": "logistics"},
    )

    best_strong = max(float(score_identity(c, strong_subject)) for c in strong.candidates)
    best_weak = max(float(score_identity(c, weak_subject)) for c in weak.candidates)

    assert best_strong == pytest.approx(config.p_ceiling, abs=1e-9)
    assert best_weak < best_strong


def test_the_ceiling_of_a_signal_is_not_the_ceiling_of_p() -> None:
    """Two different quantities, deliberately two different fields.

    score_ceiling is what a fully present signal is worth, and it is also the
    default grouping_basis_factor. p_ceiling is the most this system may claim.
    Collapsing them would make either impossible to change on its own.
    """
    config = ScoringConfig()
    assert config.score_ceiling == 1.0
    assert config.p_ceiling < config.score_ceiling
