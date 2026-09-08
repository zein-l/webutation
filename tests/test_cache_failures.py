"""A failed search must not be stored as though it had succeeded.

SerpAPI answers a failed search with HTTP 200 and an ``error`` member, so
``raise_for_status`` passes and the body parses. Every such response was written
to the cache and served from it forever after: Yandex reporting that it could
not fetch the subject's photo — a fact about this deployment's public URL, not
about the photograph — became a permanent "failed" that survived fixing the
cause. ``cached_image`` already refuses to cache failures, and states why. This
is the same rule applied to the search cache.
"""

from __future__ import annotations

import json

import pytest

from app import cache as cache_module
from app.cache import cached_get, purge_failed, response_error

SUCCESS = {"organic_results": [{"title": "A result", "link": "https://a.example"}]}
FAILURE = {
    "search_metadata": {"status": "Success"},
    "error": "The URL does not refer to an image, or the image is not publicly accessible.",
}


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.content = json.dumps(payload).encode("utf-8")
        self.headers = {"Content-Type": "application/json"}

    def raise_for_status(self) -> None:
        """SerpAPI returns 200 for a failed search, so this never fires."""


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache_module, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    return tmp_path


def stub_network(monkeypatch, payload: dict) -> list[int]:
    """Serve ``payload`` and count how many times the network was reached."""
    calls: list[int] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return FakeResponse(payload)

    monkeypatch.setattr(cache_module.requests, "get", fake_get)
    return calls


# --- detection -----------------------------------------------------------


def test_an_error_member_is_recognised_as_a_failure() -> None:
    assert response_error(FAILURE) == FAILURE["error"]
    assert response_error(SUCCESS) is None
    assert response_error({"error": ""}) is None
    assert response_error({"error": "   "}) is None
    assert response_error(None) is None
    assert response_error(["error"]) is None


# --- writing -------------------------------------------------------------


def test_a_successful_search_is_still_cached(cache_dir, monkeypatch) -> None:
    calls = stub_network(monkeypatch, SUCCESS)
    params = {"engine": "google", "q": "anything"}

    assert cached_get(params) == SUCCESS
    assert cached_get(params) == SUCCESS

    assert len(calls) == 1, "the second call must be served from disk"
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_a_failed_search_is_not_written(cache_dir, monkeypatch) -> None:
    calls = stub_network(monkeypatch, FAILURE)
    params = {"engine": "yandex_images", "url": "https://x.test/p.png"}

    first = cached_get(params)

    assert first == FAILURE, "the caller still sees the failure and reports it"
    assert list(cache_dir.glob("*.json")) == [], "nothing may reach the disk"
    assert len(calls) == 1


def test_a_failure_is_retried_rather_than_remembered(cache_dir, monkeypatch) -> None:
    """The point of the rule: a source fixed tomorrow works tomorrow."""
    calls = stub_network(monkeypatch, FAILURE)
    params = {"engine": "yandex_images", "url": "https://x.test/p.png"}

    cached_get(params)
    cached_get(params)
    assert len(calls) == 2, "a failure must not satisfy the next request"

    # The deployment is fixed and the same search now succeeds.
    calls_ok = stub_network(monkeypatch, SUCCESS)
    assert cached_get(params) == SUCCESS
    assert len(calls_ok) == 1
    assert len(list(cache_dir.glob("*.json"))) == 1

    # And from here it is a normal cached success.
    assert cached_get(params) == SUCCESS
    assert len(calls_ok) == 1


def test_a_failure_does_not_displace_a_good_entry(cache_dir, monkeypatch) -> None:
    """A later failure must not overwrite or archive a result already held."""
    params = {"engine": "google", "q": "anything"}
    stub_network(monkeypatch, SUCCESS)
    cached_get(params)

    stub_network(monkeypatch, FAILURE)
    assert cached_get(params, force_refresh=True) == FAILURE

    stored = list(cache_dir.glob("*.json"))
    assert len(stored) == 1
    envelope = json.loads(stored[0].read_text(encoding="utf-8"))
    assert json.loads(envelope["response_body"]) == SUCCESS
    assert not (cache_dir / "history").exists(), "nothing was displaced"


# --- purging -------------------------------------------------------------


def write_entry(cache_dir, key: str, payload: dict) -> None:
    (cache_dir / f"{key}.json").write_text(
        json.dumps({"params": {}, "response_body": json.dumps(payload)}),
        encoding="utf-8",
    )


def test_purge_lists_before_it_deletes(cache_dir) -> None:
    write_entry(cache_dir, "good", SUCCESS)
    write_entry(cache_dir, "bad", FAILURE)

    found = purge_failed(dry_run=True)

    assert [p.stem for p, _ in found] == ["bad"]
    assert found[0][1] == FAILURE["error"]
    assert len(list(cache_dir.glob("*.json"))) == 2, "a dry run deletes nothing"


def test_purge_removes_only_the_failures(cache_dir) -> None:
    write_entry(cache_dir, "good", SUCCESS)
    write_entry(cache_dir, "bad", FAILURE)

    purge_failed(dry_run=False)

    remaining = {p.stem for p in cache_dir.glob("*.json")}
    assert remaining == {"good"}
    assert purge_failed(dry_run=True) == [], "nothing left to purge"


def test_purge_ignores_entries_it_cannot_read(cache_dir) -> None:
    """A malformed file is not a failure, and must not be deleted on a guess."""
    (cache_dir / "broken.json").write_text("{not json", encoding="utf-8")
    write_entry(cache_dir, "bad", FAILURE)

    purge_failed(dry_run=False)

    assert (cache_dir / "broken.json").exists()
    assert not (cache_dir / "bad.json").exists()
