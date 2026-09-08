"""A photo-only subject must produce a usable report.

This is the run that had no text anchor at all, and every text-shaped signal
was therefore absent. The report said "held by face match 0.99" and, directly
underneath, an identity confidence of 0.00 — and since r = p x q, that zero
propagated to every claim on the page.

The face evidence was there the whole time; anchoring had measured it. Scoring
simply never read it.
"""

from __future__ import annotations

import math

import pytest

from app.adapters.base import AssertionDraft, Query
from app.adapters.serpapi import GoogleLensAdapter, GoogleSearchAdapter
from app.engine.candidates import form_candidates
from app.engine.faces import FaceEmbedding, FaceMatcher
from app.engine.scoring import ScoringConfig, derive_attribution, score_identity, score_item
from app.models import SourceRunStatus

PHOTO = "https://example.test/uploads/" + "a" * 64 + ".png"


def unit(*values: float) -> FaceEmbedding:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return FaceEmbedding(vector=tuple(v / norm for v in values), det_score=0.99)


SUBJECT_FACE = unit(1.0, 0.0, 0.0)
SAME_PERSON = unit(0.995, 0.1, 0.0)
STRANGER = unit(0.0, 1.0, 0.0)


def image(ref: str, url: str, origin: str = "rocketreach.co") -> AssertionDraft:
    return AssertionDraft(
        predicate="image_url", raw_value=url, normalized_value=url,
        record_ref=ref, publisher=origin, origin_key=origin,
    )


def matcher(faces: dict[str, list[FaceEmbedding]]) -> FaceMatcher:
    return FaceMatcher(
        subject_photo_url="subject.jpg",
        embedder=lambda body: faces.get(body.decode(), []),
        fetch=lambda url: url.encode() if url in faces else None,
    )


# --- the bug, stated as the requirement asked ------------------------------


def test_a_face_match_alone_scores_p_well_above_half() -> None:
    """A near-certain face, no name, no context. p must reflect the face."""
    drafts = [image("r1", "them.jpg")]
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    formation = form_candidates(
        drafts, subject=Query(photo_url=PHOTO), face_match=faces.match_record
    )
    anchor = formation.anchor_candidate
    assert anchor is not None

    p = score_identity(anchor, Query(photo_url=PHOTO))

    assert float(p) > 0.5
    assert p.components["face"] is not None
    assert p.components["name"] is None
    assert p.components["locality"] is None


def test_the_face_is_what_drives_it() -> None:
    """Not merely above threshold: the face is the term doing the work."""
    drafts = [image("r1", "them.jpg")]
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})
    formation = form_candidates(
        drafts, subject=Query(photo_url=PHOTO), face_match=faces.match_record
    )
    p = score_identity(formation.anchor_candidate, Query(photo_url=PHOTO))

    config = ScoringConfig()
    expected = (
        p.components["face"] * config.weight_face
        + p.components["corroboration"] * config.weight_corroboration
    ) / (config.weight_face + config.weight_corroboration)
    assert float(p) == pytest.approx(expected)
    # The face contributes the large majority of the weight in play.
    assert config.weight_face > config.weight_corroboration * 3


def test_claims_no_longer_score_zero_through_p() -> None:
    """r = p x q, so a zeroed p silently zeroed every claim on the page."""
    drafts = [
        image("r1", "them.jpg"),
        AssertionDraft(
            predicate="name", raw_value="Michael Petrie", normalized_value="michael petrie",
            record_ref="r1", publisher="rocketreach.co", origin_key="rocketreach.co",
        ),
    ]
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})
    formation = form_candidates(
        drafts, subject=Query(photo_url=PHOTO), face_match=faces.match_record
    )
    anchor = formation.anchor_candidate
    p = score_identity(anchor, Query(photo_url=PHOTO))

    for assertion in anchor.assertions:
        q = score_item(assertion, anchor, consumed_record_refs=p.consumed_record_refs)
        assert derive_attribution(p, q) > 0.0


def test_a_poor_face_match_does_not_inherit_a_high_p() -> None:
    """The fix must not turn the face term into a free pass."""
    strong = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})
    weak = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [STRANGER]})

    def anchor_for(m):
        return form_candidates(
            [image("r1", "them.jpg")],
            subject=Query(photo_url=PHOTO),
            face_match=m.match_record,
        ).anchor_candidate

    good = anchor_for(strong)
    assert good is not None
    assert float(score_identity(good, Query(photo_url=PHOTO))) > 0.5
    # A stranger's face does not anchor at all, so there is no candidate to
    # inherit anything.
    assert anchor_for(weak) is None


def test_an_explicit_face_distance_still_wins() -> None:
    """A caller that measured the face itself is not overridden."""
    drafts = [image("r1", "them.jpg")]
    faces = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})
    anchor = form_candidates(
        drafts, subject=Query(photo_url=PHOTO), face_match=faces.match_record
    ).anchor_candidate

    far = score_identity(anchor, Query(photo_url=PHOTO), face_distance=0.95)
    near = score_identity(anchor, Query(photo_url=PHOTO), face_distance=0.05)
    assert float(far) < float(near)


def test_a_candidate_with_no_face_evidence_is_unchanged() -> None:
    """Text-only runs must score exactly as they did before."""
    from app.engine.candidates import CandidateDraft

    candidate = CandidateDraft(
        name_key="michael petrie",
        locality_keys={"austin tx"},
        assertions=[
            AssertionDraft(
                predicate="name", raw_value="Michael Petrie",
                normalized_value="michael petrie", record_ref="r1",
                origin_key="a.example",
            )
        ],
    )
    p = score_identity(candidate, Query(name="Michael Petrie", address="Austin, TX"))
    assert p.components["face"] is None
    assert float(p) > 0.0


# --- the adapters that cannot run say so -----------------------------------


def test_a_photo_only_run_skips_the_text_adapter() -> None:
    google = GoogleSearchAdapter(fetch=lambda params: {"organic_results": []})
    run = google.run(Query(photo_url=PHOTO))

    assert run.source_run.status is SourceRunStatus.skipped
    assert run.source_run.error_reason == "needs a name or context"
    assert "ValueError" not in run.source_run.error_reason


def test_a_text_only_run_skips_the_image_adapter() -> None:
    lens = GoogleLensAdapter(fetch=lambda params: {"visual_matches": []})
    run = lens.run(Query(name="Michael Petrie"))

    assert run.source_run.status is SourceRunStatus.skipped
    assert run.source_run.error_reason == "needs a photo"


def test_skipped_empty_and_failed_are_three_different_answers() -> None:
    skipped = GoogleSearchAdapter(fetch=lambda p: {}).run(Query(photo_url=PHOTO))
    empty = GoogleSearchAdapter(fetch=lambda p: {"organic_results": []}).run(
        Query(name="Michael Petrie")
    )
    failed = GoogleSearchAdapter(
        fetch=lambda p: (_ for _ in ()).throw(ConnectionError("reset"))
    ).run(Query(name="Michael Petrie"))

    statuses = {
        skipped.source_run.status,
        empty.source_run.status,
        failed.source_run.status,
    }
    assert len(statuses) == 3
    assert skipped.source_run.status is SourceRunStatus.skipped
    assert empty.source_run.status is SourceRunStatus.empty
    assert failed.source_run.status is SourceRunStatus.failed
