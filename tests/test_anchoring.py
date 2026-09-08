"""Subject anchoring in candidate formation.

Without an anchor, formation clusters blindly and eight search results about
one person become eight candidates. The subject is a hypothesis: a name and a
context were supplied, and results should be judged against them.

Anchoring is not merge-by-name. Two records join the anchor because each
matched the subject independently, which is a stronger claim than agreeing
with each other. Records that fail stay visible as their own candidates,
because showing which same-name people were rejected, and why, is the point.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import (
    CandidateFormationConfig,
    FormationConfig,
    form_candidates,
    normalize_name_key,
)

JAN = datetime(2024, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2024, 2, 1, tzinfo=timezone.utc)

ANCHOR_SUBJECT = Query(name="Ada Lovelace", context="mathematician analytical engine")


def search_result(
    ref: str, title: str, *, origin: str, city: str | None = None
) -> list[AssertionDraft]:
    """One search result, shaped the way the SerpAPI adapter emits them."""
    head = title
    for separator in (" - ", " | ", ", "):
        if separator in head:
            head = head.split(separator, 1)[0]

    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value=title,
            normalized_value=normalize_name_key(head),
            record_ref=ref,
            publisher=origin,
            origin_key=origin,
        ),
        AssertionDraft(
            predicate="profile_url",
            raw_value=f"https://{origin}/p",
            normalized_value=f"https://{origin}/p",
            record_ref=ref,
            publisher=origin,
            origin_key=origin,
        ),
    ]
    if city:
        drafts.append(
            AssertionDraft(
                predicate="city",
                raw_value=city,
                normalized_value=city.casefold(),
                record_ref=ref,
                publisher=origin,
                origin_key=origin,
            )
        )
    return drafts


def plain_record(
    publisher: str, origin: str, observed_at: datetime, **fields: str
) -> list[AssertionDraft]:
    ref = f"{publisher}/{origin}"
    return [
        AssertionDraft(
            predicate=predicate,
            raw_value=value,
            normalized_value=value.casefold(),
            record_ref=ref,
            publisher=publisher,
            origin_key=origin,
            observed_at=observed_at,
        )
        for predicate, value in fields.items()
    ]


# --- the failure anchoring exists to fix -----------------------------------


def test_eight_matching_results_form_one_anchor_candidate() -> None:
    titles = [
        "Ada Lovelace - Wikipedia",
        "About Ada Lovelace",
        "Ada Lovelace | Biography and Facts",
        "Ada Lovelace, the first programmer",
        "Ada Lovelace - Computer History Museum",
        "Remembering Ada Lovelace",
        "Ada Lovelace Day",
        "Ada Lovelace - Britannica",
    ]
    drafts: list[AssertionDraft] = []
    context: dict[str, str] = {}
    for index, title in enumerate(titles):
        ref = f"result-{index}"
        drafts.extend(search_result(ref, title, origin=f"site{index}.example"))
        context[ref] = "A mathematician who wrote notes on the analytical engine."

    blind = form_candidates(drafts)
    anchored = form_candidates(drafts, subject=ANCHOR_SUBJECT, record_context=context)

    # Blind clustering makes every result its own person. Records sharing a
    # name-only block key no longer combine, because a shared name is not
    # evidence, so the fragmentation is now complete and honest rather than
    # partly hidden by name keys that happened to collide.
    assert len(blind.candidates) == 8

    assert anchored.anchor_available is True
    assert len(anchored.candidates) == 1

    candidate = anchored.anchor_candidate
    assert candidate is not None
    assert candidate.is_anchor is True
    assert len(candidate.assertions) == len(drafts)
    assert len(candidate.anchor_matches) == 8
    assert anchored.unattached == []


# --- the wrong person stays visible ----------------------------------------


def test_a_different_person_with_the_same_name_stays_separate() -> None:
    """Rejecting the wrong Michael Petrie is the feature, not a side effect."""
    subject = Query(name="Michael Petrie", context="logistics operations Austin")

    drafts = [
        *search_result("right-1", "Michael Petrie - Meridian Freight", origin="a.example"),
        *search_result("right-2", "Michael Petrie | Operations", origin="b.example"),
        *search_result("wrong-1", "Michael Petrie - Cardiology", origin="c.example"),
    ]
    context = {
        "right-1": "Operations lead in logistics, based in Austin.",
        "right-2": "Logistics operations manager, Austin.",
        "wrong-1": "Consultant cardiologist at a Boston teaching hospital.",
    }

    result = form_candidates(drafts, subject=subject, record_context=context)

    anchor = result.anchor_candidate
    assert anchor is not None
    assert set(anchor.anchor_matches) == {"right-1", "right-2"}

    # The other person is retained and visible, not discarded.
    others = [c for c in result.candidates if not c.is_anchor]
    assert len(others) == 1
    assert {d.record_ref for d in others[0].assertions} == {"wrong-1"}

    # Nothing was lost anywhere.
    kept = sum(len(c.assertions) for c in result.candidates) + len(result.unattached)
    assert kept == len(drafts)


def test_the_rejection_reason_is_recorded() -> None:
    """A rejected record's score is the explanation a reviewer needs."""
    subject = Query(name="Michael Petrie", context="logistics operations Austin")
    drafts = search_result("wrong-1", "Michael Petrie - Cardiology", origin="c.example")
    context = {"wrong-1": "Consultant cardiologist at a Boston teaching hospital."}

    result = form_candidates(drafts, subject=subject, record_context=context)

    rejected = result.anchor_matches["wrong-1"]
    assert result.anchor_candidate is None
    assert rejected.strength <= FormationConfig().anchor_threshold
    # The name matched perfectly. The context is what rejected it.
    assert rejected.name_signal == 1.0
    assert rejected.context_signal == 0.0


# --- photo-only queries have no text anchor --------------------------------


def test_a_photo_only_subject_falls_back_to_the_blind_path_unchanged() -> None:
    drafts = [
        *plain_record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *plain_record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]

    baseline = form_candidates(drafts)
    photo_only = form_candidates(
        drafts, subject=Query(photo_url="https://example.com/face.jpg")
    )

    assert photo_only.anchor_available is False
    assert photo_only.anchor_matches == {}
    assert photo_only.anchor_candidate is None
    assert len(photo_only.candidates) == len(baseline.candidates) == 2
    assert not any(c.is_anchor for c in photo_only.candidates)


def test_no_subject_at_all_is_identical_to_before() -> None:
    drafts = [
        *plain_record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *plain_record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]
    without = form_candidates(drafts)
    explicit_none = form_candidates(drafts, subject=None, record_context=None)

    assert without.anchor_available is explicit_none.anchor_available is False
    assert len(without.candidates) == len(explicit_none.candidates) == 2


# --- per-record strength ---------------------------------------------------


def test_anchor_match_strength_is_recorded_per_record() -> None:
    drafts = [
        *search_result("r1", "Ada Lovelace - Wikipedia", origin="a.example"),
        *search_result("r2", "Charles Babbage - Wikipedia", origin="b.example"),
    ]
    context = {
        "r1": "A mathematician known for the analytical engine notes.",
        "r2": "An inventor who designed the difference engine.",
    }

    result = form_candidates(drafts, subject=ANCHOR_SUBJECT, record_context=context)

    assert set(result.anchor_matches) == {"r1", "r2"}
    for match in result.anchor_matches.values():
        assert 0.0 <= match.strength <= 1.0
        assert match.basis

    strong, weak = result.anchor_matches["r1"], result.anchor_matches["r2"]
    assert strong.strength > weak.strength
    assert strong.name_signal == 1.0
    assert weak.name_signal == 0.0
    assert "name" in strong.basis
    assert "context" in strong.basis


def test_a_record_with_no_snippet_is_scored_on_name_but_does_not_anchor() -> None:
    """Missing context is dropped from the mean, not counted against.

    Dropping it is right for the arithmetic and fatal for the decision: the
    mean of one signal is that signal, so a record with nothing but a matching
    name scores a perfect 1.00. It is scored, and it is refused.
    """
    drafts = search_result("r1", "Ada Lovelace - Wikipedia", origin="a.example")

    result = form_candidates(drafts, subject=ANCHOR_SUBJECT, record_context={})

    match = result.anchor_matches["r1"]
    assert match.context_signal is None
    assert match.name_signal == 1.0
    assert match.strength == 1.0
    assert match.admitted is False
    assert match.rejected_because == "name only — no other signal"
    assert result.anchor_candidate is None


def test_a_title_containing_the_subject_name_scores_but_does_not_anchor() -> None:
    """"About Ada Lovelace" is the subject's name, which is not the subject."""
    drafts = search_result("r1", "About Ada Lovelace", origin="a.example")
    result = form_candidates(drafts, subject=ANCHOR_SUBJECT, record_context={})

    match = result.anchor_matches["r1"]
    assert match.name_signal == FormationConfig().anchor_name_subject_contained
    assert match.admitted is False
    assert result.anchor_candidate is None


def test_a_title_containing_the_subject_name_anchors_with_context() -> None:
    """The same record, once something other than the name agrees."""
    drafts = search_result("r1", "About Ada Lovelace", origin="a.example")
    result = form_candidates(
        drafts,
        subject=ANCHOR_SUBJECT,
        record_context={"r1": "notes on the analytical engine"},
    )

    match = result.anchor_matches["r1"]
    assert match.name_signal == FormationConfig().anchor_name_subject_contained
    assert match.context_signal > 0
    assert match.admitted is True
    assert result.anchor_candidate is not None


# --- anchoring does not weaken the existing rules --------------------------


def test_anchoring_does_not_reintroduce_merge_by_name() -> None:
    """Non-anchor blocks still never merge on a shared name alone."""
    subject = Query(name="Ada Lovelace", context="mathematician")
    drafts = [
        *plain_record("a.example", "origin_a", JAN, name="Michael Petrie", city="Austin, TX"),
        *plain_record("b.example", "origin_b", FEB, name="Michael Petrie", city="Boston, MA"),
    ]

    result = form_candidates(drafts, subject=subject, record_context={})

    assert result.anchor_candidate is None
    assert len(result.candidates) == 2


def test_anchored_records_are_not_merged_with_each_other() -> None:
    """Each matched the subject independently. That is the whole distinction."""
    drafts = [
        *search_result("r1", "Ada Lovelace - Wikipedia", origin="a.example", city="London"),
        *search_result("r2", "Ada Lovelace - Britannica", origin="b.example", city="Paris"),
    ]
    # Each carries a context term of its own, so each earns its place in the
    # anchor independently. Anchoring on the shared name alone is refused.
    result = form_candidates(
        drafts,
        subject=ANCHOR_SUBJECT,
        record_context={
            "r1": "mathematician, analytical engine",
            "r2": "the analytical engine notes",
        },
    )

    anchor = result.anchor_candidate
    assert anchor is not None
    # Two localities, and no merge evidence, because no merge took place.
    assert anchor.locality_keys == {"london", "paris"}
    assert anchor.merge_evidence == []


# --- configuration ---------------------------------------------------------


def test_anchor_threshold_is_configurable() -> None:
    # A partial name, so the threshold is what decides. A record carrying none
    # of the subject's name is refused before the threshold is consulted.
    drafts = search_result("r1", "Lovelace - Wikipedia", origin="a.example")
    context = {"r1": "A mathematician who studied the analytical engine."}

    strict = form_candidates(
        drafts,
        config=FormationConfig(anchor_threshold=0.95),
        subject=ANCHOR_SUBJECT,
        record_context=context,
    )
    assert strict.anchor_candidate is None

    lenient = form_candidates(
        drafts,
        config=FormationConfig(anchor_threshold=0.2),
        subject=ANCHOR_SUBJECT,
        record_context=context,
    )
    assert lenient.anchor_candidate is not None


def test_the_config_alias_is_the_same_class() -> None:
    assert CandidateFormationConfig is FormationConfig
