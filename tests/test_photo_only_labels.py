"""A run must not describe a comparison that never happened.

On a search with a photograph and nothing else, every candidate that is not the
anchor was labelled "Other person, same name" and "Held by: a shared name
only". No name was supplied, so no name was ever compared — and the rejection
list said so, reporting all 59 of them as decided on face similarity. The two
halves of the same report contradicted each other.

The grouping basis genuinely reads "name_only", which is where the wording came
from. But that describes how records were clustered against *each other*, by
their own name keys, and says nothing about the subject. Rendering it as a
shared name asserts a match to a name the caller never gave.
"""

from __future__ import annotations

import json

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates
from app.models import GroupingBasis, SourceOrigin
from app.report import _plain_basis, build_report


def record(ref: str, name: str, origin: str) -> list[AssertionDraft]:
    def draft(predicate: str, value: str) -> AssertionDraft:
        return AssertionDraft(
            predicate=predicate, raw_value=value,
            normalized_value=value.casefold(), record_ref=ref,
            publisher=origin, origin_key=origin,
            source_origin=SourceOrigin.known_origin,
        )

    return [draft("name", name), draft("profile_url", f"https://{origin}/p")]


PHOTO_ONLY = Query(photo_url="https://x.test/subject.png")
NAMED = Query(name="Michael Petrie")


def photo_only_report():
    """The real shape: unrelated pages returned by a reverse image search."""
    drafts = [
        *record("r1", "Michael G Sondag CFP", "wealth.example"),
        *record("r2", "Dr. Jose A. Jaller, MD", "clinic.example"),
        *record("r3", "XING", "xing.example"),
    ]
    formation = form_candidates(drafts, subject=PHOTO_ONLY, record_context={})
    return build_report(
        subject=PHOTO_ONLY,
        runs=[],
        formation=formation,
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": len(drafts)},
    )


# --- the contradiction, stated directly ----------------------------------


def test_no_candidate_claims_a_shared_name_when_no_name_was_given() -> None:
    report = photo_only_report()
    assert report["candidates"], "the fixture must produce candidates"
    for candidate in report["candidates"]:
        assert candidate["held_by"] != "a shared name only"
        assert "shared name" not in candidate["held_by"]


def test_the_whole_report_never_says_the_name_matched() -> None:
    """Belt and braces: no wording anywhere may imply a name comparison."""
    report = photo_only_report()
    text = json.dumps(report).lower()
    for phrase in ("shared name", "same name", "sharing this name"):
        assert phrase not in text, f"{phrase!r} appears with no name supplied"


def test_the_report_says_which_comparisons_were_possible() -> None:
    report = photo_only_report()
    assert report["compared"] == {
        "name": False,
        "context": False,
        "locality": False,
        "face": True,
    }


def test_a_lone_record_says_it_was_matched_against_nothing() -> None:
    candidates = photo_only_report()["candidates"]
    assert all(
        c["held_by"] == "a single record, matched against nothing" for c in candidates
    ), [c["held_by"] for c in candidates]


# --- and the named case is untouched -------------------------------------


def test_a_named_search_still_reports_a_shared_name() -> None:
    """The wording is only wrong when the comparison did not happen."""
    drafts = [
        *record("r1", "Michael Petrie", "a.example"),
        *record("r2", "Michael Petrie", "b.example"),
    ]
    formation = form_candidates(drafts, subject=NAMED, record_context={})
    report = build_report(
        subject=NAMED, runs=[], formation=formation,
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": 4},
    )

    assert report["compared"]["name"] is True
    assert any("shared name" in c["held_by"] for c in report["candidates"])


def test_the_basis_follows_the_candidates_own_name_score() -> None:
    """Not whether the run had a name — whether this candidate matched one.

    A mixed search supplies a name and still compares it to nothing on most
    candidates, because a reverse image search returns pages about whoever the
    picture resembles. Keying on the subject was too coarse: it silenced the
    false claim on photo-only runs and left it standing on mixed ones.
    """
    from app.engine.candidates import CandidateDraft

    candidate = CandidateDraft(
        name_key="somebody else",
        grouping_basis=GroupingBasis.name_only,
        assertions=record("r1", "Somebody Else", "a.example"),
    )
    # No name score: the claim is unsupported whatever the run supplied.
    assert _plain_basis(candidate, PHOTO_ONLY, None) != "a shared name only"
    assert _plain_basis(candidate, NAMED, 0.0) != "a shared name only"
    # A real name agreement earns the wording.
    assert _plain_basis(candidate, NAMED, 1.0) == "a shared name only"


# --- and the same claim on a mixed run, which is where it survived -------


def mixed_report():
    """A named, contexted search whose other candidates came from the photo.

    This is the case the photo-only test could not see. A name *was* supplied,
    so a run-level "was a name compared" flag says yes — while a reverse image
    search returns pages about whoever the picture resembles, and those score
    0.00 against the name. In one live run 79 of 116 non-anchor candidates were
    in exactly this position and every one was labelled as sharing the name.
    """
    drafts = [
        # The subject.
        *record("r1", "Michael Petrie", "linkedin.example"),
        # Lens results: real people, no name in common with the subject.
        *record("r2", "Prof Mark Walterfang", "cabrini.com.au"),
        *record("r3", "Meet the PPG Team", "copiers.example"),
        *record("r4", "My Bio", "personal.example"),
    ]
    subject = Query(
        name="Michael Petrie",
        context="Webutation, private investigator, insurance fraud, OSINT",
        photo_url="https://x.test/subject.png",
    )
    formation = form_candidates(
        drafts, subject=subject,
        record_context={"r1": "Webutation private investigator"},
    )
    return build_report(
        subject=subject, runs=[], formation=formation,
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": len(drafts)},
    )


def test_a_candidate_with_no_name_score_never_claims_a_shared_name() -> None:
    """The per-candidate rule: the run had a name, this candidate did not match it."""
    report = mixed_report()
    unmatched = [
        c for c in report["candidates"]
        if not c["signals"].get("name") and str(c["is_anchor"]).lower() != "true"
    ]
    assert unmatched, "the fixture must produce candidates with no name agreement"
    for candidate in unmatched:
        assert "shared name" not in candidate["held_by"], candidate["display_name"]
        assert "same name" not in candidate["stamp"], candidate["display_name"]


def test_the_stamp_names_what_decided_each_candidate() -> None:
    report = mixed_report()
    for candidate in report["candidates"]:
        stamp = candidate["stamp"]
        if str(candidate["is_anchor"]).lower() == "true":
            assert stamp == "The person searched for"
        elif candidate["signals"].get("name"):
            assert stamp == "Other person, same name"
        else:
            assert stamp in ("Other record, no face match", "Other record, not matched")


def test_a_genuine_namesake_is_still_called_one() -> None:
    """The wording is only wrong when the comparison did not happen."""
    drafts = [
        *record("r1", "Michael Petrie", "a.example"),
        *record("r2", "Michael Petrie", "b.example"),
    ]
    subject = Query(name="Michael Petrie", context="Webutation")
    report = build_report(
        subject=subject, runs=[],
        formation=form_candidates(drafts, subject=subject, record_context={}),
        spend={"searches": 1, "live_calls": 0, "from_cache": 1, "drafts": 4},
    )
    named = [c for c in report["candidates"] if c["signals"].get("name")]
    assert named, "same-name candidates must still exist"
    for candidate in named:
        assert candidate["stamp"] == "Other person, same name"


def test_every_candidate_carries_a_stamp() -> None:
    """A missing stamp sends the view to its fallback, which claims nothing."""
    for report in (photo_only_report(), mixed_report()):
        for candidate in report["candidates"]:
            assert candidate.get("stamp"), candidate["display_name"]


# --- and the section heading above them ---------------------------------


def test_the_section_heading_does_not_claim_a_shared_name_for_all() -> None:
    """"116 other people sharing this name" when 37 shared it.

    The same defect as the cards, one level up: a run-level flag said a name
    was supplied and the heading described every candidate beneath it that way.
    """
    report = mixed_report()
    others = report["other_candidates"]

    assert others["total"] == others["same_name"] + others["no_face_match"] + others["unmatched"]
    if others["same_name"] < others["total"]:
        assert "sharing this name" not in others["label"] or "—" in others["label"], (
            f"heading claims a shared name for all of them: {others['label']!r}"
        )


def test_the_heading_counts_agree_with_the_cards() -> None:
    """Counted from the stamps, so the two cannot drift apart."""
    for report in (photo_only_report(), mixed_report()):
        others = report["other_candidates"]
        cards = [c for c in report["candidates"] if str(c["is_anchor"]).lower() != "true"]
        assert others["total"] == len(cards)
        assert others["same_name"] == sum(
            1 for c in cards if c["stamp"] == "Other person, same name"
        )
        assert others["unmatched"] == sum(
            1 for c in cards if c["stamp"] == "Other record, not matched"
        )


def test_a_photo_only_heading_never_mentions_a_name() -> None:
    """The blunt scan already covers this, stated once more where it is decided."""
    label = photo_only_report()["other_candidates"]["label"]
    for phrase in ("sharing this name", "same name", "shared name"):
        assert phrase not in label, label


def test_a_heading_for_one_population_says_it_plainly() -> None:
    """A split is only worth showing when there is a split."""
    from app.report import _others_summary

    all_named = [{"is_anchor": False, "stamp": "Other person, same name"}] * 4
    assert _others_summary(all_named)["label"] == "4 other people sharing this name"

    one = [{"is_anchor": False, "stamp": "Other person, same name"}]
    assert _others_summary(one)["label"] == "1 other person sharing this name"


def test_a_mixed_heading_shows_the_split() -> None:
    from app.report import _others_summary

    mixed = (
        [{"is_anchor": False, "stamp": "Other person, same name"}] * 37
        + [{"is_anchor": False, "stamp": "Other record, not matched"}] * 79
    )
    label = _others_summary(mixed)["label"]
    assert label.startswith("116 other candidates")
    assert "37 sharing this name" in label
    assert "79 that matched the subject on nothing" in label


def test_the_heading_says_what_was_not_matched() -> None:
    """"79 that matched on nothing" read as a fact about the records.

    A reader took it to mean those records carry no attributes, rather than
    that they failed every comparison against the subject. The subject is named
    where the phrase would otherwise be about the record itself.
    """
    from app.report import _others_summary

    mixed = (
        [{"is_anchor": False, "stamp": "Other person, same name"}] * 37
        + [{"is_anchor": False, "stamp": "Other record, not matched"}] * 79
    )
    label = _others_summary(mixed)["label"]
    assert "79 that matched the subject on nothing" in label
    assert "matched on nothing" not in label.replace("matched the subject on nothing", "")

    only_unmatched = [{"is_anchor": False, "stamp": "Other record, not matched"}] * 12
    assert _others_summary(only_unmatched)["label"] == (
        "12 other records that did not match the subject"
    )


def test_the_face_wording_stays_short() -> None:
    """A face match has one obvious counterpart, so naming it adds nothing.

    Spelling out the subject in every clause made a three-way label repeat it
    twice. The phrase that needed it was the one that read as a property of the
    record.
    """
    from app.report import _others_summary

    faces = [{"is_anchor": False, "stamp": "Other record, no face match"}] * 59
    assert _others_summary(faces)["label"] == "59 records whose faces did not match"

    three_way = (
        [{"is_anchor": False, "stamp": "Other person, same name"}] * 2
        + [{"is_anchor": False, "stamp": "Other record, no face match"}] * 3
        + [{"is_anchor": False, "stamp": "Other record, not matched"}] * 4
    )
    label = _others_summary(three_way)["label"]
    assert label.count("the subject") == 1, label
