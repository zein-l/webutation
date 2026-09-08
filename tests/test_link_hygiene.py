"""Links, publishers and the basis line, as a live run actually delivered them.

Every value here was taken from a real SerpAPI response or the report it
produced. Three faults, all in what the adapter stored rather than in scoring:

* 252 URLs arrived with their escapes intact and were unusable, one of them on
  a record the report had placed in the anchor;
* Google redirect stubs became record identities, so four wrapped copies of
  pages already collected escaped deduplication — and because Google returns a
  different snippet per row, the same page was scored twice at two strengths;
* 72 assertions stored a full URL as the publisher, so each distinct URL
  counted as a distinct publisher in the corroboration summary.
"""

from __future__ import annotations

import pytest

from app.adapters.base import Query
from app.adapters.serpapi import (
    GoogleSearchAdapter,
    clean_link,
    is_redirect_stub,
    unescape_json_escapes,
)

# Verbatim from the run.
ESCAPED = "https://www.theclm.org/persondetails?id" + chr(92) + "u003d11376"
STUB = "/goto?url=CAESqQEB6zswFdtEZCUYHllG5YPO2ZW_E0VVsf-KSIL9rkT8_mREquyiZukU0hwm"
REAL = "https://www.linkedin.com/in/michaelpetrie"


# --- escapes -------------------------------------------------------------


def test_an_escaped_equals_is_restored() -> None:
    assert chr(92) + "u003d" in ESCAPED, "the fixture must carry a real escape"
    assert unescape_json_escapes(ESCAPED) == (
        "https://www.theclm.org/persondetails?id=11376"
    )


def test_a_clean_url_is_left_alone() -> None:
    assert unescape_json_escapes(REAL) == REAL
    assert clean_link(REAL) == REAL


def test_other_escapes_are_restored_too() -> None:
    assert unescape_json_escapes("a\u0026b") == "a&b"
    assert unescape_json_escapes("x\u003fy") == "x?y"


# --- redirect stubs ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        STUB,
        "/url?q=https://example.com",
        "https://www.google.com/url?sa=t&url=CAESZQ",
    ],
)
def test_redirect_stubs_are_recognised(url: str) -> None:
    assert is_redirect_stub(url)
    assert clean_link(url) is None


@pytest.mark.parametrize("url", [REAL, "https://rocketreach.co/michael-petrie-email_16265680"])
def test_real_pages_are_not_mistaken_for_stubs(url: str) -> None:
    assert not is_redirect_stub(url)
    assert clean_link(url) == url


def test_nothing_is_not_a_link() -> None:
    assert clean_link(None) is None
    assert clean_link("") is None
    assert clean_link("   ") is None
    assert clean_link(42) is None


# --- through the adapter -------------------------------------------------


def payload(results: list[dict]) -> dict:
    return {"organic_results": results}


def run(results: list[dict]):
    adapter = GoogleSearchAdapter(fetch=lambda params: payload(results))
    return adapter.run(Query(name="Michael Petrie"))


def test_the_stored_url_is_usable() -> None:
    drafts = run([{"title": "Michael Petrie", "link": ESCAPED, "source": "CLM"}]).drafts
    urls = [d.raw_value for d in drafts if d.predicate == "profile_url"]
    assert urls == ["https://www.theclm.org/persondetails?id=11376"]
    assert chr(92) + "u003d" not in urls[0], "no literal escape may survive"


def test_a_wrapped_duplicate_does_not_become_a_second_record() -> None:
    """The exact shape that put four extra members in one anchor."""
    result = run([
        {"title": "Michael Petrie", "link": REAL, "source": "LinkedIn",
         "snippet": "Founder/CEO @Webutation"},
        # Google's wrapper around the same page, with a different snippet.
        {"title": "Michael Petrie", "link": STUB, "source": "LinkedIn",
         "snippet": "a different snippet for the same page"},
    ])
    refs = {d.record_ref for d in result.drafts}
    assert len(refs) == 1, "the wrapped row must not mint a second record"
    assert all("/goto" not in r for r in refs)


def test_a_result_with_no_link_at_all_is_still_kept() -> None:
    """No link is honest; a stub is a claim of one. They are not the same."""
    result = run([{"title": "Michael Petrie", "source": "Somewhere"}])
    assert result.drafts, "a linkless result still carries a name"
    assert all("/goto" not in d.record_ref for d in result.drafts)


def test_snippets_follow_the_cleaned_identity() -> None:
    """Context is keyed on the record ref, so it has to use the same one."""
    adapter = GoogleSearchAdapter(
        fetch=lambda params: payload(
            [{"title": "Michael Petrie", "link": ESCAPED, "snippet": "Webutation"}]
        )
    )
    result = adapter.run(Query(name="Michael Petrie"))
    refs = {d.record_ref for d in result.drafts}
    assert set(result.record_context) == refs


# --- publishers ----------------------------------------------------------


def test_a_url_publisher_is_reduced_to_its_domain() -> None:
    """Verbatim from the run: 72 assertions stored a URL here."""
    drafts = run([{
        "title": "Michael Petrie",
        "link": "https://gohuskies.com/sports/baseball/roster/michael-petrie/13497",
        "source": "https://gohuskies.com/sports/baseball/roster/michael-petrie/13497",
    }]).drafts
    assert {d.publisher for d in drafts} == {"gohuskies.com"}


def test_two_pages_on_one_site_report_one_publisher() -> None:
    """The inflation this fixes: distinct URLs read as distinct outlets."""
    drafts = run([
        {"title": "A", "link": "https://gohuskies.com/a",
         "source": "https://gohuskies.com/a"},
        {"title": "B", "link": "https://gohuskies.com/b",
         "source": "https://gohuskies.com/b"},
    ]).drafts
    assert {d.publisher for d in drafts} == {"gohuskies.com"}


def test_a_real_publisher_name_is_kept() -> None:
    """theclm.org says less to a reviewer than the name does."""
    drafts = run([{
        "title": "Michael Petrie",
        "link": "https://www.theclm.org/persondetails?id=11376",
        "source": "Claims and Litigation Management Alliance",
    }]).drafts
    assert {d.publisher for d in drafts} == {"Claims and Litigation Management Alliance"}


def test_a_bare_host_publisher_is_also_reduced() -> None:
    drafts = run([{"title": "X", "link": "https://example.com/p",
                   "source": "www.example.com"}]).drafts
    assert {d.publisher for d in drafts} == {"example.com"}
