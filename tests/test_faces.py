"""Face verification, driven by a stub embedder.

No model is loaded and no image is downloaded. The embedder and the fetcher are
both injected, which is the same seam that lets InsightFace be swapped out.

The property under test throughout is the difference between "no" and "no
answer". A record with no image has said nothing about whether it is the
subject, and must not be scored as though it had said no.
"""

from __future__ import annotations

import math

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates
from app.engine.faces import (
    FaceComparison,
    FaceConfig,
    FaceEmbedding,
    FaceMatcher,
    compare,
    image_urls,
    similarity_to_distance,
)
from app.models import GroupingBasis


def unit(*values: float) -> FaceEmbedding:
    """A face embedding, normalised so similarities read as cosines."""
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return FaceEmbedding(
        vector=tuple(v / norm for v in values), det_score=0.99
    )


SUBJECT_FACE = unit(1.0, 0.0, 0.0)
SAME_PERSON = unit(0.95, 0.31, 0.0)      # cosine ~0.95 with the subject
DIFFERENT_PERSON = unit(0.0, 1.0, 0.0)   # orthogonal, cosine 0.0
BYSTANDER = unit(0.2, 0.9, 0.3)
#: The same person again, photographed differently: alike but not identical.
SAME_PERSON_AGAIN = unit(0.90, 0.42, 0.10)


def image_draft(ref: str, url: str, origin: str = "a.example") -> AssertionDraft:
    return AssertionDraft(
        predicate="image_url",
        raw_value=url,
        normalized_value=url,
        record_ref=ref,
        publisher=origin,
        origin_key=origin,
    )


def name_draft(ref: str, name: str, origin: str = "a.example") -> AssertionDraft:
    return AssertionDraft(
        predicate="name",
        raw_value=name,
        normalized_value=name.casefold(),
        record_ref=ref,
        publisher=origin,
        origin_key=origin,
    )


def matcher(faces_by_url: dict[str, list[FaceEmbedding]], **kwargs) -> FaceMatcher:
    """A FaceMatcher whose "images" are just URL keys into a dict."""
    return FaceMatcher(
        subject_photo_url="subject.jpg",
        embedder=lambda body: faces_by_url.get(body.decode(), []),
        fetch=lambda url: url.encode() if url in faces_by_url else None,
        **kwargs,
    )


# --- direction: the sign error this module exists to prevent --------------


def test_cosine_similarity_is_higher_for_more_alike_faces() -> None:
    assert compare(SUBJECT_FACE, SUBJECT_FACE) == pytest.approx(1.0)
    assert compare(SUBJECT_FACE, SAME_PERSON) > compare(
        SUBJECT_FACE, DIFFERENT_PERSON
    )
    assert compare(SUBJECT_FACE, DIFFERENT_PERSON) == pytest.approx(0.0)


def test_distance_runs_the_other_way() -> None:
    """Formation speaks distance; this module speaks similarity."""
    alike = compare(SUBJECT_FACE, SAME_PERSON)
    unalike = compare(SUBJECT_FACE, DIFFERENT_PERSON)

    assert similarity_to_distance(alike) < similarity_to_distance(unalike)
    assert similarity_to_distance(1.0) == 0.0
    assert similarity_to_distance(0.0) == 1.0


def test_a_zero_vector_yields_no_information_not_a_mismatch() -> None:
    empty = FaceEmbedding(vector=(0.0, 0.0, 0.0))
    assert compare(SUBJECT_FACE, empty) == 0.0


def test_mismatched_dimensions_raise_rather_than_compare() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        compare(SUBJECT_FACE, FaceEmbedding(vector=(1.0, 0.0)))


# --- absence of evidence ---------------------------------------------------


def test_a_record_with_no_image_yields_no_signal() -> None:
    m = matcher({"subject.jpg": [SUBJECT_FACE]})
    assert m.available is True
    assert m.match_record([name_draft("r1", "Michael Petrie")]) is None


def test_an_unreachable_image_yields_no_signal() -> None:
    m = matcher({"subject.jpg": [SUBJECT_FACE]})
    result = m.match_record([image_draft("r1", "https://gone.example/x.jpg")])
    assert result is None


def test_an_image_with_no_detectable_face_yields_no_signal() -> None:
    m = matcher({"subject.jpg": [SUBJECT_FACE], "logo.jpg": []})
    assert m.match_record([image_draft("r1", "logo.jpg")]) is None


def test_no_subject_photo_means_the_matcher_is_unavailable() -> None:
    m = FaceMatcher(subject_photo_url=None, embedder=lambda b: [], fetch=lambda u: None)
    assert m.available is False
    assert m.match_record([image_draft("r1", "face.jpg")]) is None


def test_a_subject_photo_with_no_face_is_reported_not_guessed() -> None:
    m = matcher({"subject.jpg": []})
    assert m.available is False
    assert m.subject_error == "no face detected in the subject photo"


def test_an_unfetchable_subject_photo_is_reported() -> None:
    m = FaceMatcher(
        subject_photo_url="missing.jpg",
        embedder=lambda b: [SUBJECT_FACE],
        fetch=lambda url: None,
    )
    assert m.available is False
    assert m.subject_error == "subject photo could not be fetched"


def test_none_and_zero_are_different_answers() -> None:
    """The distinction the whole module turns on."""
    m = matcher({"subject.jpg": [SUBJECT_FACE], "stranger.jpg": [DIFFERENT_PERSON]})

    no_answer = m.match_record([name_draft("r1", "Someone")])
    said_no = m.match_record([image_draft("r2", "stranger.jpg")])

    assert no_answer is None
    assert said_no is not None
    assert said_no.best_similarity == pytest.approx(0.0)


# --- comparison ------------------------------------------------------------


def test_a_matching_face_produces_a_high_similarity() -> None:
    m = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})
    result = m.match_record([image_draft("r1", "them.jpg")])

    assert result is not None
    assert result.best_similarity > 0.9
    assert result.ambiguous is False
    assert result.faces_in_image == 1
    assert result.image_url == "them.jpg"
    assert result.distance == pytest.approx(1.0 - result.best_similarity)


def test_multiple_faces_record_all_similarities_and_flag_ambiguity() -> None:
    """A group shot may match the person standing next to the subject."""
    m = matcher(
        {"subject.jpg": [SUBJECT_FACE], "group.jpg": [BYSTANDER, SAME_PERSON]}
    )
    result = m.match_record([image_draft("r1", "group.jpg")])

    assert result is not None
    assert result.ambiguous is True
    assert result.faces_in_image == 2
    assert len(result.similarities) == 2
    assert result.best_similarity == max(result.similarities)
    assert result.best_similarity > 0.9


def test_the_best_of_several_images_is_used() -> None:
    m = matcher(
        {
            "subject.jpg": [SUBJECT_FACE],
            "blurry.jpg": [DIFFERENT_PERSON],
            "clear.jpg": [SAME_PERSON],
        }
    )
    result = m.match_record(
        [image_draft("r1", "blurry.jpg"), image_draft("r1", "clear.jpg")]
    )
    assert result.best_similarity > 0.9
    assert result.image_url == "clear.jpg"


def test_images_per_record_are_capped() -> None:
    urls = {f"i{n}.jpg": [SAME_PERSON] for n in range(10)}
    drafts = [image_draft("r1", url) for url in urls]
    assert len(image_urls(drafts, FaceConfig(max_images_per_record=3))) == 3


def test_each_url_is_fetched_and_embedded_once() -> None:
    """Never re-download, and never re-embed within a run."""
    fetched: list[str] = []
    embedded: list[bytes] = []

    def fetch(url: str) -> bytes | None:
        fetched.append(url)
        return url.encode()

    def embedder(body: bytes) -> list[FaceEmbedding]:
        embedded.append(body)
        return [SUBJECT_FACE] if body == b"subject.jpg" else [SAME_PERSON]

    m = FaceMatcher(
        subject_photo_url="subject.jpg", embedder=embedder, fetch=fetch
    )
    for _ in range(3):
        m.match_record([image_draft("r1", "them.jpg")])

    assert fetched.count("them.jpg") == 1
    assert embedded.count(b"them.jpg") == 1


# --- merging ---------------------------------------------------------------


def test_candidate_distance_converts_the_sign_once() -> None:
    from app.engine.candidates import CandidateDraft

    m = matcher(
        {
            "subject.jpg": [SUBJECT_FACE],
            "left.jpg": [SAME_PERSON],
            "right.jpg": [SAME_PERSON],
        }
    )
    left = CandidateDraft(name_key="a", assertions=[image_draft("l", "left.jpg")])
    right = CandidateDraft(name_key="b", assertions=[image_draft("r", "right.jpg")])

    distance = m.candidate_distance(left, right)
    assert distance is not None
    assert distance < 0.1  # the same person, so nearly zero distance


def test_candidate_distance_is_none_without_images() -> None:
    from app.engine.candidates import CandidateDraft

    m = matcher({"subject.jpg": [SUBJECT_FACE]})
    left = CandidateDraft(name_key="a", assertions=[name_draft("l", "A")])
    right = CandidateDraft(name_key="b", assertions=[name_draft("r", "B")])
    assert m.candidate_distance(left, right) is None


def test_a_face_merge_records_the_similarity_in_its_evidence() -> None:
    from app.engine.candidates import CandidateDraft  # noqa: F401

    drafts = [
        name_draft("l", "Michael Petrie", "a.example"),
        AssertionDraft(
            predicate="city", raw_value="Austin, TX", normalized_value="austin, tx",
            record_ref="l", origin_key="a.example",
        ),
        image_draft("l", "left.jpg", "a.example"),
        name_draft("r", "Michael Petrie", "b.example"),
        AssertionDraft(
            predicate="city", raw_value="Boston, MA", normalized_value="boston, ma",
            record_ref="r", origin_key="b.example",
        ),
        image_draft("r", "right.jpg", "b.example"),
    ]
    m = matcher(
        {
            "subject.jpg": [SUBJECT_FACE],
            "left.jpg": [SAME_PERSON],
            "right.jpg": [SAME_PERSON_AGAIN],
        }
    )

    result = form_candidates(drafts, face_distance=m.candidate_distance)

    assert len(result.candidates) == 1
    (evidence,) = result.candidates[0].merge_evidence
    # The report speaks similarity; the rule is written in distance. Both are
    # stated so neither has to be inferred.
    assert "matched on face" in evidence
    assert "similarity" in evidence
    assert "face distance" in evidence
    similarity = float(evidence.split("similarity ")[1].split()[0])
    assert 0.9 < similarity < 1.0
    assert result.candidates[0].grouping_basis is GroupingBasis.face


# --- anchoring on a face alone --------------------------------------------


SUBJECT_QUERY = Query(photo_url="subject.jpg")


def test_a_face_match_anchors_a_record_with_no_name_and_no_context() -> None:
    """A face is direct evidence about the person. A name is only a label."""
    drafts = [image_draft("r1", "them.jpg")]
    m = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    result = form_candidates(
        drafts, subject=SUBJECT_QUERY, face_match=m.match_record
    )

    assert result.anchor_available is True
    anchor = result.anchor_candidate
    assert anchor is not None
    match = result.anchor_matches["r1"]
    assert match.admitted is True
    assert match.name_signal is None
    assert match.context_signal is None
    assert match.face_signal > 0.9
    assert "face" in match.basis


def test_a_non_matching_face_does_not_anchor() -> None:
    drafts = [image_draft("r1", "stranger.jpg")]
    m = matcher({"subject.jpg": [SUBJECT_FACE], "stranger.jpg": [DIFFERENT_PERSON]})

    result = form_candidates(
        drafts, subject=SUBJECT_QUERY, face_match=m.match_record
    )
    assert result.anchor_candidate is None
    assert result.anchor_matches["r1"].face_signal == pytest.approx(0.0)


def test_an_ambiguous_face_match_is_flagged_on_the_anchor() -> None:
    drafts = [image_draft("r1", "group.jpg")]
    m = matcher(
        {"subject.jpg": [SUBJECT_FACE], "group.jpg": [BYSTANDER, SAME_PERSON]}
    )

    result = form_candidates(
        drafts, subject=SUBJECT_QUERY, face_match=m.match_record
    )
    match = result.anchor_matches["r1"]
    assert match.admitted is True
    assert match.face_ambiguous is True
    assert "ambiguous" in match.basis


def test_a_photo_only_subject_now_has_an_anchor_when_faces_are_available() -> None:
    """Previously anchoring was skipped entirely for a photo-only query."""
    drafts = [image_draft("r1", "them.jpg"), name_draft("r1", "Someone Else")]
    m = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    without_faces = form_candidates(drafts, subject=SUBJECT_QUERY)
    with_faces = form_candidates(
        drafts, subject=SUBJECT_QUERY, face_match=m.match_record
    )

    assert without_faces.anchor_available is False
    assert with_faces.anchor_available is True
    assert with_faces.anchor_candidate is not None


def test_the_anchor_similarity_threshold_is_configurable() -> None:
    from app.engine.candidates import FormationConfig

    drafts = [image_draft("r1", "them.jpg")]
    m = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [BYSTANDER]})

    similarity = compare(SUBJECT_FACE, BYSTANDER)
    assert 0.0 < similarity < 0.55

    strict = form_candidates(
        drafts, subject=SUBJECT_QUERY, face_match=m.match_record
    )
    assert strict.anchor_candidate is None

    lenient = form_candidates(
        drafts,
        config=FormationConfig(anchor_face_similarity=similarity - 0.01),
        subject=SUBJECT_QUERY,
        face_match=m.match_record,
    )
    assert lenient.anchor_candidate is not None


def test_a_face_match_overrides_the_thin_substance_rule() -> None:
    """No context and no link, but a face. That is enough and only that is."""
    drafts = [
        name_draft("r1", "Michael Petrie Associates"),
        image_draft("r1", "them.jpg"),
    ]
    m = matcher({"subject.jpg": [SUBJECT_FACE], "them.jpg": [SAME_PERSON]})

    subject = Query(name="Michael Petrie", photo_url="subject.jpg")
    result = form_candidates(
        drafts, subject=subject, record_context={}, face_match=m.match_record
    )

    match = result.anchor_matches["r1"]
    assert match.has_link is False
    assert match.context_signal is None
    assert match.admitted is True
    assert match.rejected_because is None
