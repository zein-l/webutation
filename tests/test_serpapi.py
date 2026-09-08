"""SerpAPI adapters, driven entirely by saved responses.

No test here touches the network. The fetch callable is injected, which is the
same seam that lets every call go through the cache in production.

The most important test in this file is the one asserting that nothing is
emitted from snippet text. A search snippet reads like structured data and is
not; turning it into claims would manufacture evidence that no source gave.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.adapters.base import AdapterRun, Query, RawResponse
from app.adapters.serpapi import (
    GoogleLensAdapter,
    GoogleSearchAdapter,
    SerpApiConfig,
    YandexImagesAdapter,
    registrable_domain,
)
from app.models import AccessCategory, EvidenceKind, SourceOrigin, SourceRunStatus

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "serpapi"

SUBJECT = Query(name="Marcus Webb", address="Austin, TX")


def saved(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def replaying(name: str):
    """A fetch callable that replays one saved response."""
    payload = saved(name)
    return lambda params: payload


def raising(exc: Exception):
    def _fetch(params: dict) -> dict:
        raise exc

    return _fetch


@pytest.fixture
def google() -> GoogleSearchAdapter:
    return GoogleSearchAdapter(fetch=replaying("google_sample"))


@pytest.fixture
def run(google: GoogleSearchAdapter) -> AdapterRun:
    return google.run(SUBJECT)


# --- registrable domain ----------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.linkedin.com/in/marcus-webb-atx", "linkedin.com"),
        ("https://techcrunch.com/2025/04/11/meridian/", "techcrunch.com"),
        ("https://www.imdb.com/name/nm9911223/", "imdb.com"),
        # Subdomains collapse: one organisation is one origin.
        ("https://sub.domain.example.co.uk/page", "example.co.uk"),
        ("https://news.bbc.co.uk/story", "bbc.co.uk"),
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
        ("https://careers.linkedin.com/x", "linkedin.com"),
        ("http://EXAMPLE.COM/Path", "example.com"),
        # Nothing nameable: lineage must fall to unknown rather than guess.
        ("https://192.168.0.1/page", None),
        ("https://localhost/page", None),
        ("not a url", None),
        ("", None),
        (None, None),
    ],
)
def test_registrable_domain(url: str | None, expected: str | None) -> None:
    assert registrable_domain(url) == expected


# --- per-result mapping ----------------------------------------------------


def test_origin_keys_are_domains_not_urls(run: AdapterRun) -> None:
    origins = {d.origin_key for d in run.drafts}
    assert "linkedin.com" in origins
    assert "techcrunch.com" in origins
    assert "spokeo.com" in origins
    assert "imdb.com" in origins
    # A full URL never appears as an origin.
    assert not any("/" in (o or "") for o in origins)


def test_record_ref_is_one_per_result(run: AdapterRun) -> None:
    """One search result is one record, derived from its link."""
    linkedin = [d for d in run.drafts if d.origin_key == "linkedin.com"]
    assert len({d.record_ref for d in linkedin}) == 1
    assert linkedin[0].record_ref == (
        "serpapi:google:https://www.linkedin.com/in/marcus-webb-atx"
    )
    # Each result is its own record.
    assert len({d.record_ref for d in run.drafts}) == len(
        {d.origin_key for d in run.drafts}
    )


def test_record_ref_falls_back_to_a_hash_of_the_block(
    google: GoogleSearchAdapter,
) -> None:
    payload = {"organic_results": [{"position": 1, "title": "No Link Here"}]}
    raw = RawResponse(payload=payload, fetched_at=None, artifact_ref="x", source="s")
    (draft,) = google.normalize(raw)
    assert draft.record_ref.startswith("serpapi:google:sha256:")
    # Stable across repeat collections of identical bytes.
    assert google.normalize(raw)[0].record_ref == draft.record_ref


def test_publisher_prefers_the_source_field_then_the_domain(
    run: AdapterRun,
) -> None:
    by_origin = {d.origin_key: d.publisher for d in run.drafts}
    assert by_origin["linkedin.com"] == "LinkedIn"
    assert by_origin["techcrunch.com"] == "TechCrunch"


def test_publisher_falls_back_to_the_domain(google: GoogleSearchAdapter) -> None:
    payload = {
        "organic_results": [
            {"position": 1, "title": "A Page", "link": "https://example.org/a"}
        ]
    }
    raw = RawResponse(payload=payload, fetched_at=None, artifact_ref="x", source="s")
    drafts = google.normalize(raw)
    assert {d.publisher for d in drafts} == {"example.org"}


def test_access_category_is_public_web(run: AdapterRun) -> None:
    assert {d.access_category for d in run.drafts} == {AccessCategory.PUBLIC_WEB}


def test_lineage_is_known_only_when_the_domain_is_identifiable(
    google: GoogleSearchAdapter, run: AdapterRun
) -> None:
    assert {d.source_origin for d in run.drafts} == {SourceOrigin.known_origin}

    payload = {"organic_results": [{"position": 1, "title": "Nowhere"}]}
    raw = RawResponse(payload=payload, fetched_at=None, artifact_ref="x", source="s")
    (draft,) = google.normalize(raw)
    assert draft.origin_key is None
    assert draft.source_origin is SourceOrigin.unknown


def test_evidence_kind_is_mapped_by_domain(run: AdapterRun) -> None:
    by_origin = {d.origin_key: d.evidence_kind for d in run.drafts}
    assert by_origin["linkedin.com"] is EvidenceKind.self_reported
    assert by_origin["techcrunch.com"] is EvidenceKind.secondhand
    assert by_origin["spokeo.com"] is EvidenceKind.republished


def test_unrecognised_domains_get_unknown_evidence_kind(
    google: GoogleSearchAdapter,
) -> None:
    payload = {
        "organic_results": [
            {"position": 1, "title": "A Page", "link": "https://obscure.example/a"}
        ]
    }
    raw = RawResponse(payload=payload, fetched_at=None, artifact_ref="x", source="s")
    (name, url) = google.normalize(raw)
    assert name.evidence_kind is EvidenceKind.unknown
    # Lineage is known, reliability is not. The two axes stay independent.
    assert name.source_origin is SourceOrigin.known_origin


def test_evidence_kind_is_never_inferred_from_lineage(run: AdapterRun) -> None:
    """Every draft is known_origin, yet reliability differs across them."""
    assert len({d.source_origin for d in run.drafts}) == 1
    assert len({d.evidence_kind for d in run.drafts}) > 1


# --- dates -----------------------------------------------------------------


def test_observed_at_is_set_only_when_a_result_carries_a_date(
    run: AdapterRun,
) -> None:
    dated = [d for d in run.drafts if d.origin_key == "techcrunch.com"]
    assert all(d.observed_at is not None for d in dated)
    assert dated[0].observed_at.year == 2025
    assert dated[0].observed_at.month == 4

    undated = [d for d in run.drafts if d.origin_key == "linkedin.com"]
    assert all(d.observed_at is None for d in undated)


def test_fetch_time_is_never_substituted_for_an_observation_date(
    run: AdapterRun,
) -> None:
    """Null is the normal case, and freshness must read unknown because of it."""
    from app.engine.candidates import CandidateDraft
    from app.engine.scoring import score_item

    undated = next(d for d in run.drafts if d.origin_key == "linkedin.com")
    candidate = CandidateDraft(name_key="marcus a webb", assertions=list(run.drafts))
    q = score_item(undated, candidate)

    assert undated.observed_at is None
    assert q.freshness_unknown is True
    assert run.raw.fetched_at is not None


def test_a_relative_date_is_not_resolved_to_approximately_now(
    google: GoogleSearchAdapter,
) -> None:
    payload = {
        "organic_results": [
            {
                "position": 1,
                "title": "A Page",
                "link": "https://example.org/a",
                "date": "3 days ago",
            }
        ]
    }
    raw = RawResponse(payload=payload, fetched_at=None, artifact_ref="x", source="s")
    assert all(d.observed_at is None for d in google.normalize(raw))


# --- conservative extraction ----------------------------------------------


def test_only_three_predicates_are_ever_emitted(run: AdapterRun) -> None:
    assert {d.predicate for d in run.drafts} <= {"name", "profile_url", "image_url"}


def test_no_assertion_is_emitted_from_snippet_text(run: AdapterRun) -> None:
    """The snippets name an employer, a city and an age. None becomes a claim.

    This is the test that keeps the system honest: those strings read like
    structured data and are prose written for a human reader.
    """
    emitted = {d.predicate for d in run.drafts}
    for invented in ("employer", "job_title", "city", "address", "age", "birth_date"):
        assert invented not in emitted

    snippets = " ".join(run.record_context.values()).casefold()
    assert "meridian freight" in snippets
    assert "austin" in snippets
    assert "age 47" in snippets

    # Nothing from those snippets reached a value anywhere.
    values = " ".join(
        (d.normalized_value or "") for d in run.drafts if d.predicate != "profile_url"
    )
    assert "meridian freight systems" not in values
    assert "austin, tx" not in values
    assert "47" not in values


def test_snippets_are_kept_as_context_against_their_record(
    run: AdapterRun,
) -> None:
    ref = "serpapi:google:https://www.linkedin.com/in/marcus-webb-atx"
    assert ref in run.record_context
    assert "Meridian Freight Systems" in run.record_context[ref]
    assert ref in {d.record_ref for d in run.drafts}


def test_display_name_is_the_title_head_with_the_full_title_retained(
    run: AdapterRun,
) -> None:
    name = next(
        d
        for d in run.drafts
        if d.predicate == "name" and d.origin_key == "linkedin.com"
    )
    assert name.normalized_value == "marcus a. webb"
    # The full title is not discarded, so the split can be revisited.
    assert name.raw_value == "Marcus A. Webb - Operations Manager at Meridian Freight"


# --- inline images ---------------------------------------------------------


def test_inline_images_emit_image_urls_with_the_source_page_as_publisher(
    run: AdapterRun,
) -> None:
    images = [d for d in run.drafts if d.predicate == "image_url"]
    assert len(images) == 2  # original and thumbnail
    assert {d.publisher for d in images} == {"IMDb"}
    assert {d.origin_key for d in images} == {"imdb.com"}
    assert {d.record_ref for d in images} == {
        "serpapi:google:https://www.imdb.com/name/nm9911223/"
    }
    assert any("marcus-webb-full.jpg" in (d.raw_value or "") for d in images)
    assert any("encrypted-tbn0" in (d.raw_value or "") for d in images)


# --- run outcomes ----------------------------------------------------------


def test_a_populated_response_is_found_with_a_count(run: AdapterRun) -> None:
    assert run.source_run.status is SourceRunStatus.found
    assert run.source_run.result_count == 5  # 4 organic + 1 inline image
    assert run.source_run.error_reason is None
    assert run.drafts


def test_a_zero_result_response_is_empty_not_failed() -> None:
    adapter = GoogleSearchAdapter(fetch=replaying("google_zero_results"))
    result = adapter.run(SUBJECT)

    assert result.source_run.status is SourceRunStatus.empty
    assert result.source_run.result_count == 0
    assert result.source_run.error_reason is None
    assert result.drafts == []
    # The source did answer, so the response is still there.
    assert result.raw is not None


def test_an_exception_is_failed_with_a_reason() -> None:
    adapter = GoogleSearchAdapter(fetch=raising(ConnectionError("connection reset")))
    result = adapter.run(SUBJECT)

    assert result.source_run.status is SourceRunStatus.failed
    assert "ConnectionError" in result.source_run.error_reason
    assert "connection reset" in result.source_run.error_reason
    assert result.source_run.result_count is None
    assert result.raw is None


def test_a_non_200_is_failed_with_a_reason() -> None:
    import requests

    adapter = GoogleSearchAdapter(
        fetch=raising(requests.HTTPError("401 Client Error: Unauthorized"))
    )
    result = adapter.run(SUBJECT)

    assert result.source_run.status is SourceRunStatus.failed
    assert "401" in result.source_run.error_reason


def test_an_api_level_error_is_failed_not_empty() -> None:
    """An invalid key is a failure, however politely the API reports it."""
    adapter = GoogleSearchAdapter(fetch=replaying("google_api_error"))
    result = adapter.run(SUBJECT)

    assert result.source_run.status is SourceRunStatus.failed
    assert "Invalid API key" in result.source_run.error_reason


def test_empty_and_failed_never_collapse() -> None:
    empty = GoogleSearchAdapter(fetch=replaying("google_zero_results")).run(SUBJECT)
    failed = GoogleSearchAdapter(fetch=raising(RuntimeError("boom"))).run(SUBJECT)

    assert empty.source_run.status is not failed.source_run.status
    assert empty.source_run.error_reason is None
    assert failed.source_run.error_reason is not None


# --- the three adapters ----------------------------------------------------


def test_the_three_engines_and_their_parameters() -> None:
    captured: list[dict] = []

    def capture(params: dict) -> dict:
        captured.append(params)
        return {"organic_results": []}

    photo = Query(photo_url="https://example.com/face.jpg")
    GoogleSearchAdapter(fetch=capture).run(SUBJECT)
    GoogleLensAdapter(fetch=capture).run(photo)
    # Enabled explicitly: this asserts the engine's parameters, not whether
    # this deployment calls it.
    YandexImagesAdapter(
        config=SerpApiConfig(disabled_engines=frozenset()), fetch=capture
    ).run(photo)

    assert [p["engine"] for p in captured] == ["google", "google_lens", "yandex_images"]
    assert captured[0]["q"] == "Marcus Webb Austin, TX"
    assert captured[1]["url"] == "https://example.com/face.jpg"
    assert captured[2]["url"] == "https://example.com/face.jpg"


def test_image_adapters_are_skipped_without_a_photo() -> None:
    """Not applicable, not failed. Nothing went wrong; nothing was asked."""
    for adapter_class in (GoogleLensAdapter, YandexImagesAdapter):
        result = adapter_class(
            config=SerpApiConfig(disabled_engines=frozenset()), fetch=replaying("google_sample")
        ).run(
            Query(name="Marcus Webb")
        )
        assert result.source_run.status is SourceRunStatus.skipped
        assert result.source_run.error_reason == "needs a photo"
        assert result.drafts == []


def test_a_text_adapter_is_skipped_without_text() -> None:
    """The mirror case: a photo-only subject gives web search nothing."""
    result = GoogleSearchAdapter(fetch=replaying("google_sample")).run(
        Query(photo_url="https://example.com/face.jpg")
    )
    assert result.source_run.status is SourceRunStatus.skipped
    assert result.source_run.error_reason == "needs a name or context"


def test_no_raw_exception_text_reaches_the_report() -> None:
    """A reader should never meet a Python class name standing on its own."""
    skipped = GoogleSearchAdapter(fetch=replaying("google_sample")).run(Query())
    assert not skipped.source_run.error_reason.startswith("ValueError")

    unreachable = GoogleLensAdapter(fetch=replaying("google_sample")).run(
        Query(photo_url="http://localhost:8000/uploads/a.png")
    )
    assert unreachable.source_run.status is SourceRunStatus.failed
    assert not unreachable.source_run.error_reason.startswith("ValueError")
    assert "photo not publicly reachable" in unreachable.source_run.error_reason


def test_visual_matches_are_read_by_the_image_adapters() -> None:
    payload = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Marcus Webb",
                "link": "https://www.imdb.com/name/nm9911223/",
                "source": "IMDb",
                "thumbnail": "https://example.com/t.jpg",
                "original": "https://example.com/o.jpg",
            }
        ]
    }
    adapter = GoogleLensAdapter(fetch=lambda params: payload)
    result = adapter.run(Query(photo_url="https://example.com/face.jpg"))

    assert result.source_run.status is SourceRunStatus.found
    assert result.source_run.result_count == 1
    images = [d for d in result.drafts if d.predicate == "image_url"]
    assert len(images) == 2
    assert {d.origin_key for d in result.drafts} == {"imdb.com"}


# --- cache integration -----------------------------------------------------


def test_every_call_goes_through_the_cache_by_default() -> None:
    """The default fetch is the cache, not a bare request."""
    from app.cache import cached_get

    assert GoogleSearchAdapter()._fetch is cached_get
    assert GoogleLensAdapter()._fetch is cached_get
    assert YandexImagesAdapter()._fetch is cached_get


def test_artifact_ref_is_the_cache_key_for_the_request(
    google: GoogleSearchAdapter,
) -> None:
    from app.cache import cache_key

    raw = google.collect(SUBJECT)
    expected = cache_key(
        {"engine": "google", "q": "Marcus Webb Austin, TX", "num": 10}
    )
    assert raw.artifact_ref == expected
    assert len(raw.artifact_ref) == 64


# --- configuration ---------------------------------------------------------


def test_the_evidence_map_is_configurable() -> None:
    config = SerpApiConfig(
        evidence_kind_by_domain={"linkedin.com": EvidenceKind.direct_observation}
    )
    adapter = GoogleSearchAdapter(config=config, fetch=replaying("google_sample"))
    result = adapter.run(SUBJECT)

    by_origin = {d.origin_key: d.evidence_kind for d in result.drafts}
    assert by_origin["linkedin.com"] is EvidenceKind.direct_observation
    assert by_origin["techcrunch.com"] is EvidenceKind.unknown


# --- engines this deployment does not call --------------------------------


def test_yandex_is_off_by_default_and_reports_not_applicable() -> None:
    """Off because it refuses this deployment's photo URLs, not because it is bad.

    Since failures are no longer cached, leaving it on buys the same refusal on
    every run. Declining is reported as skipped, not failed: nothing went wrong.
    """
    called: list[dict] = []
    run = YandexImagesAdapter(fetch=lambda p: called.append(p) or {}).run(
        Query(photo_url="https://example.com/face.jpg")
    )

    assert run.source_run.status is SourceRunStatus.skipped
    assert called == [], "a disabled engine must not reach the network"
    assert "re-enable" in run.source_run.error_reason.lower()
    assert run.drafts == []


def test_lens_is_still_on_by_default() -> None:
    called: list[dict] = []
    GoogleLensAdapter(fetch=lambda p: called.append(p) or {"visual_matches": []}).run(
        Query(photo_url="https://example.com/face.jpg")
    )
    assert len(called) == 1


def test_a_disabled_engine_can_be_re_enabled() -> None:
    """The flag is the whole mechanism — no code change to turn it back on."""
    called: list[dict] = []
    run = YandexImagesAdapter(
        config=SerpApiConfig(disabled_engines=frozenset()),
        fetch=lambda p: called.append(p) or {"visual_matches": []},
    ).run(Query(photo_url="https://example.com/face.jpg"))

    assert run.source_run.status is not SourceRunStatus.skipped
    assert len(called) == 1


def test_any_engine_can_be_disabled_not_just_yandex() -> None:
    """A general switch, so this is not a Yandex special case."""
    called: list[dict] = []
    run = GoogleSearchAdapter(
        config=SerpApiConfig(disabled_engines=frozenset({"google"})),
        fetch=lambda p: called.append(p) or {"organic_results": []},
    ).run(SUBJECT)

    assert run.source_run.status is SourceRunStatus.skipped
    assert called == []
