"""Name keys must not turn one page into two people.

A search title is a page title. "Michael Petrie's Post" and "Michael Petrie on
LinkedIn" describe one person, and a name key that keeps the tail spawns a
second candidate from a single record.
"""

from __future__ import annotations

import pytest

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates, normalize_name_key

PERSON = "michael petrie"


@pytest.mark.parametrize(
    "title",
    [
        "Michael Petrie",
        "Michael Petrie's Post",
        "Michael Petrie on LinkedIn",
        "Michael Petrie | LinkedIn",
        "Michael Petrie - LinkedIn",
        "Michael Petrie - Wikipedia",
        "Michael Petrie Profile",
        "Michael Petrie Profile Page",
        "Michael Petrie's",
        "MICHAEL PETRIE",
        "Michael Petrie, Jr.",
    ],
)
def test_page_title_variants_all_reduce_to_one_name_key(title: str) -> None:
    assert normalize_name_key(title) == PERSON


def test_one_record_never_produces_two_distinct_name_keys() -> None:
    """The bug this fixes: one LinkedIn page becoming two people.

    A record carries both the extracted name and the full page title. Both are
    consulted, so both must reduce to the same key.
    """
    record = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie's Post",
            normalized_value="michael petrie's post",
            record_ref="serpapi:google:https://linkedin.com/posts/mp",
            origin_key="linkedin.com",
        ),
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie on LinkedIn",
            normalized_value="michael petrie on linkedin",
            record_ref="serpapi:google:https://linkedin.com/posts/mp",
            origin_key="linkedin.com",
        ),
    ]

    keys = {normalize_name_key(d.normalized_value) for d in record}
    keys |= {normalize_name_key(d.raw_value) for d in record}
    assert keys == {PERSON}

    # And the record therefore forms exactly one candidate.
    result = form_candidates(record)
    assert len(result.candidates) == 1
    assert result.candidates[0].name_key == PERSON


def test_the_post_and_the_profile_share_one_name_key() -> None:
    """The fix is about keys, not about grouping.

    Both records now block under one key instead of inventing a second person
    from a possessive. They still stay two candidates, because a shared name is
    not evidence that two pages describe the same human. Anchoring, or a shared
    attribute, is what may unite them later.
    """
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie",
            normalized_value="michael petrie",
            record_ref="ref-profile",
            origin_key="linkedin.com",
        ),
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie's Post",
            normalized_value="michael petrie's post",
            record_ref="ref-post",
            origin_key="linkedin.com",
        ),
    ]

    result = form_candidates(drafts)
    assert {c.name_key for c in result.candidates} == {PERSON}
    assert len(result.candidates) == 2


# --- the guards that keep this from eating real names ---------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # A surname that is also a platform word must survive.
        ("Michael Page", "michael page"),
        ("Sarah Post", "sarah post"),
        ("James Page", "james page"),
        # Middle initials are not possessive remnants.
        ("Michael S Petrie", "michael s petrie"),
        ("Marcus A. Webb", "marcus a webb"),
        # A single-token name is never stripped away.
        ("Sanders", "sanders"),
        ("LinkedIn", "linkedin"),
        # A real title that happens to end in a tail word keeps its shape when
        # stripping would leave too little behind.
        ("Ada Lovelace Day", "ada lovelace day"),
        # No apostrophe means no possessive. "Petries" may be a surname,
        # and guessing otherwise loses a real name to save a page title.
        ("Michael Petries Post", "michael petries"),
    ],
)
def test_stripping_never_eats_the_name_itself(title: str, expected: str) -> None:
    assert normalize_name_key(title) == expected


def test_anchoring_still_matches_a_tailed_title() -> None:
    """The fix must not cost the anchor its match.

    What is under test here is the name key: "Michael Petrie's Post" has to
    normalise to the subject's name. The record is given a matching context
    term as well, because a name on its own no longer admits anything.
    """
    drafts = [
        AssertionDraft(
            predicate="name",
            raw_value="Michael Petrie's Post",
            normalized_value="michael petrie's post",
            record_ref="ref-post",
            origin_key="linkedin.com",
        ),
        AssertionDraft(
            predicate="profile_url",
            raw_value="https://linkedin.com/posts/mp",
            normalized_value="https://linkedin.com/posts/mp",
            record_ref="ref-post",
            origin_key="linkedin.com",
        ),
    ]
    subject = Query(name="Michael Petrie", context="logistics")

    result = form_candidates(
        drafts, subject=subject, record_context={"ref-post": "logistics"}
    )
    match = result.anchor_matches["ref-post"]
    assert match.name_signal == 1.0
    assert result.anchor_candidate is not None
