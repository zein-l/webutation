"""How a candidate is titled in the report.

The heading is the first thing an investigator reads, and it has to be a
person. Search results supply page titles, and the most search-optimised title
is usually the longest one, so choosing by length hands the heading to whoever
wrote the best SEO rather than to whoever the evidence points at.
"""

from __future__ import annotations

import pytest

from app.report import _display_name


def name_row(raw: str, r: float) -> dict:
    return {"predicate": "name", "raw_value": raw, "r": r}


SEO_TITLE = "Michael Petrie Email & Phone Number | Enolla Consulting"
BARE_NAME = "Michael Petrie"


def test_a_bare_name_wins_over_a_long_seo_title() -> None:
    """The case this exists for."""
    rows = [
        name_row(SEO_TITLE, 0.15),
        name_row(BARE_NAME, 0.21),
        {"predicate": "profile_url", "raw_value": "https://x/y", "r": 0.9},
    ]
    assert _display_name(rows) == BARE_NAME


def test_the_bare_name_wins_even_when_the_seo_title_scores_higher_alone() -> None:
    """Score leads, so a better-evidenced record takes the heading."""
    rows = [name_row(SEO_TITLE, 0.40), name_row(BARE_NAME, 0.41)]
    assert _display_name(rows) == BARE_NAME


def test_among_equal_scores_the_shortest_trimmed_name_wins() -> None:
    rows = [
        name_row("Michael Petrie - Webutation, Inc", 0.21),
        name_row("Michael Petrie's Post", 0.21),
        name_row(SEO_TITLE, 0.21),
    ]
    # All three trim at a title separator; the shortest survivor is the person.
    assert _display_name(rows) == BARE_NAME


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Michael Petrie - Wikipedia", "Michael Petrie"),
        ("Michael Petrie | LinkedIn", "Michael Petrie"),
        ("Michael Petrie – Profile", "Michael Petrie"),
        ("Michael Petrie — Investigations", "Michael Petrie"),
        ("Michael Petrie · Overview", "Michael Petrie"),
        ("Michael Petrie", "Michael Petrie"),
    ],
)
def test_page_title_tails_are_trimmed(title: str, expected: str) -> None:
    assert _display_name([name_row(title, 0.5)]) == expected


def test_a_name_with_a_comma_is_not_cut_at_the_comma() -> None:
    """"Webb, Marcus" is a name, not a title with a tail."""
    assert _display_name([name_row("Webb, Marcus", 0.5)]) == "Webb, Marcus"


def test_only_an_seo_title_is_available() -> None:
    """The page furniture goes too, not just the separator.

    A photo-only subject can leave one record standing, and if that record is
    an aggregator's search-optimised title then trimming at the separator alone
    leaves the candidate headed "Michael Petrie Email & Phone Number".
    """
    assert _display_name([name_row(SEO_TITLE, 0.3)]) == BARE_NAME


def test_the_heading_survives_every_score_being_zero() -> None:
    """r is zero for every row on a photo-only run, so it cannot break a tie."""
    rows = [
        {"predicate": "name", "raw_value": "M. Petrie Consulting Services Group",
         "r": 0.0, "face": 0.10},
        {"predicate": "name", "raw_value": SEO_TITLE, "r": 0.0, "face": 0.99},
    ]
    # The face behind the record decides, then the shortest plausible name.
    assert _display_name(rows) == BARE_NAME


def test_page_furniture_is_only_trimmed_from_the_end() -> None:
    from app.report import _plausible_person_name

    assert _plausible_person_name("Michael Petrie Email & Phone Number") == BARE_NAME
    assert _plausible_person_name("Michael Petrie Contact Details") == BARE_NAME
    # Never below a first and last name, however suggestive the words.
    assert _plausible_person_name("Michael Page") == "Michael Page"
    assert _plausible_person_name("Sarah Post") == "Sarah Post"
    assert _plausible_person_name("Ada Records") == "Ada Records"


def test_no_name_rows_gives_no_heading() -> None:
    assert _display_name([{"predicate": "city", "raw_value": "Austin", "r": 0.4}]) is None
    assert _display_name([]) is None


def test_a_name_row_with_no_printed_value_is_skipped() -> None:
    rows = [
        {"predicate": "name", "raw_value": None, "r": 0.9},
        name_row(BARE_NAME, 0.2),
    ]
    assert _display_name(rows) == BARE_NAME


def test_the_live_case_no_longer_titles_a_person_with_a_page_title() -> None:
    """Regression: the exact rows that produced the wrong heading on screen."""
    rows = [
        name_row("Michael Petrie - Webutation, Inc", 0.21),
        name_row("Michael Petrie's Post", 0.21),
        name_row(SEO_TITLE + " ...", 0.15),
    ]
    heading = _display_name(rows)
    assert heading == BARE_NAME
    assert "Enolla" not in heading
    assert "|" not in heading
