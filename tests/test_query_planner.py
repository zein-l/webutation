"""Query expansion, and the deduplication that makes it safe.

Expansion buys recall. It is only affordable because the same page found by
several queries collapses back to one record, so the cost is searches rather
than duplicated evidence.
"""

from __future__ import annotations

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import form_candidates, normalize_name_key
from app.engine.query_planner import (
    QueryPlanConfig,
    context_terms,
    deduplicate_drafts,
    describe_plan,
    organisation_name,
    plan_queries,
    query_text,
)

SUBJECT = Query(
    name="Michael Petrie",
    context="Webutation, private investigator, insurance fraud, OSINT, New York",
)


# --- the plan --------------------------------------------------------------


def test_the_plan_covers_the_five_variant_shapes() -> None:
    texts = describe_plan(plan_queries(SUBJECT))

    assert texts[0] == "Michael Petrie"
    assert texts[1] == '"Michael Petrie"'
    assert '"Michael Petrie" Webutation' in texts
    assert "Michael Petrie private investigator" in texts
    assert "Michael Petrie OSINT" in texts


def test_the_count_is_capped_by_config() -> None:
    """Each query is a paid search, so recall is bought deliberately."""
    assert len(plan_queries(SUBJECT)) == QueryPlanConfig().max_queries == 6
    assert len(plan_queries(SUBJECT, QueryPlanConfig(max_queries=3))) == 3
    assert len(plan_queries(SUBJECT, QueryPlanConfig(max_queries=1))) == 1


def test_the_broadest_query_comes_first() -> None:
    """A cap must not cut the query most likely to find anything."""
    (only,) = plan_queries(SUBJECT, QueryPlanConfig(max_queries=1))
    assert query_text(only) == "Michael Petrie"


def test_no_duplicate_queries_are_planned() -> None:
    texts = describe_plan(plan_queries(SUBJECT))
    assert len(texts) == len(set(texts))


def test_a_name_with_no_context_still_plans_two_queries() -> None:
    texts = describe_plan(plan_queries(Query(name="Ada Lovelace")))
    assert texts == ["Ada Lovelace", '"Ada Lovelace"']


def test_context_without_a_name_searches_the_context() -> None:
    texts = describe_plan(plan_queries(Query(context="Webutation, OSINT")))
    assert texts == ["Webutation", "OSINT"]


def test_a_photo_only_subject_has_nothing_to_expand() -> None:
    assert plan_queries(Query(photo_url="https://example.com/face.jpg")) == []


def test_the_organisation_is_recognised_and_searched_alone() -> None:
    assert organisation_name(SUBJECT.context, QueryPlanConfig()) == "Webutation"

    lean = plan_queries(
        SUBJECT,
        QueryPlanConfig(max_queries=6, include_context_terms=False),
    )
    assert "Webutation" in describe_plan(lean)


def test_a_field_of_work_is_not_mistaken_for_an_organisation() -> None:
    config = QueryPlanConfig()
    assert organisation_name("private investigator, insurance fraud", config) is None
    assert organisation_name("osint, background check", config) is None


def test_context_terms_are_distinct_and_meaningful() -> None:
    terms = context_terms(SUBJECT.context, QueryPlanConfig())
    assert "Webutation" in terms
    assert "insurance fraud" in terms
    assert len(terms) == len(set(terms))
    # Short and empty fragments do not become queries.
    assert all(len(t) >= QueryPlanConfig().min_term_length for t in terms)


# --- the same page found by several queries stays one record ---------------


def result_drafts(record_ref: str, title: str, origin: str) -> list[AssertionDraft]:
    return [
        AssertionDraft(
            predicate="name",
            raw_value=title,
            normalized_value=normalize_name_key(title),
            record_ref=record_ref,
            publisher=origin,
            origin_key=origin,
        ),
        AssertionDraft(
            predicate="profile_url",
            raw_value=f"https://{origin}/p",
            normalized_value=f"https://{origin}/p",
            record_ref=record_ref,
            publisher=origin,
            origin_key=origin,
        ),
    ]


def test_the_same_page_found_by_three_queries_stays_one_record() -> None:
    """The property that makes expansion affordable."""
    page = result_drafts("serpapi:google:https://linkedin.com/in/mp", "Michael Petrie", "linkedin.com")
    from_three_queries = [*page, *page, *page]

    once = form_candidates(page)
    thrice = form_candidates(from_three_queries)

    assert len(thrice.candidates) == len(once.candidates) == 1

    candidate = thrice.candidates[0]
    assert {d.record_ref for d in candidate.assertions} == {
        "serpapi:google:https://linkedin.com/in/mp"
    }
    # One origin, however many queries surfaced it. Corroboration is unmoved.
    assert {d.origin_key for d in candidate.assertions} == {"linkedin.com"}


def test_repeated_discovery_does_not_inflate_corroboration() -> None:
    from app.engine.scoring import score_identity

    page = result_drafts("ref-1", "Michael Petrie", "linkedin.com")
    subject = Query(name="Michael Petrie")

    once = form_candidates(page).candidates[0]
    thrice = form_candidates([*page, *page, *page]).candidates[0]

    p_once = score_identity(once, subject)
    p_thrice = score_identity(thrice, subject)

    assert float(p_once) == float(p_thrice)
    assert p_thrice.components["corroboration"] == 0.0


def test_deduplicate_drafts_collapses_repeats_but_keeps_distinct_claims() -> None:
    page = result_drafts("ref-1", "Michael Petrie", "linkedin.com")
    other = result_drafts("ref-2", "Michael Petrie", "theclm.org")

    merged = deduplicate_drafts([*page, *page, *other])

    assert len(merged) == len(page) + len(other)
    assert {d.record_ref for d in merged} == {"ref-1", "ref-2"}


def test_deduplication_preserves_order() -> None:
    page = result_drafts("ref-1", "Michael Petrie", "linkedin.com")
    other = result_drafts("ref-2", "Michael Petrie", "theclm.org")
    merged = deduplicate_drafts([*page, *other, *page])
    assert [d.record_ref for d in merged] == ["ref-1", "ref-1", "ref-2", "ref-2"]
