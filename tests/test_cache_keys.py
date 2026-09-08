"""Cache keys identify what was asked for, not which link asked it.

A reverse image search is keyed on the photograph. A signed upload URL mints a
fresh token every time the file is uploaded, so keying on the whole URL made
the same photograph a cache miss on every run and bought the same search again
at full price.
"""

from __future__ import annotations

import pytest

from app.adapters.base import Query
from app.adapters.serpapi import (
    GoogleLensAdapter,
    SerpApiConfig,
    YandexImagesAdapter,
)
from app.cache import cache_key, stable_for_key

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
BASE = f"https://lemon-turkeys-bathe.loca.lt/uploads/{DIGEST}.png"


def signed(url: str, token: str, expires: int) -> str:
    return f"{url}?expires={expires}&token={token}"


def lens_key(url: str) -> str:
    return cache_key({"engine": "google_lens", "url": url})


# --- the same photograph is the same lookup -------------------------------


def test_a_fresh_token_for_the_same_photo_is_the_same_key() -> None:
    first = signed(BASE, "711d8138890f3243180346b22facdb85", 1788710346)
    second = signed(BASE, "deadbeefdeadbeefdeadbeefdeadbeef", 1788799999)

    assert first != second
    assert lens_key(first) == lens_key(second)


def test_a_changed_tunnel_hostname_is_still_the_same_photo() -> None:
    """A tunnel restart hands out a new host. The image has not changed."""
    moved = f"https://some-other-name.loca.lt/uploads/{DIGEST}.png?expires=1&token=x"
    assert lens_key(signed(BASE, "t", 1)) == lens_key(moved)


def test_a_different_photo_is_a_different_lookup() -> None:
    other = f"https://lemon-turkeys-bathe.loca.lt/uploads/{OTHER_DIGEST}.png?expires=1&token=x"
    assert lens_key(signed(BASE, "t", 1)) != lens_key(other)


def test_an_upload_url_reduces_to_its_digest() -> None:
    assert stable_for_key(signed(BASE, "t", 1)) == f"upload:{DIGEST}"


# --- other URLs keep their meaning ----------------------------------------


def test_an_unsigned_url_is_left_alone() -> None:
    url = "https://example.com/photo.jpg?size=large"
    assert stable_for_key(url) == url
    assert lens_key(url) != lens_key("https://example.com/photo.jpg?size=small")


def test_a_foreign_signed_url_loses_only_its_credentials() -> None:
    a = "https://bucket.s3.amazonaws.com/p.jpg?v=2&X-Amz-Signature=aaa&X-Amz-Date=1"
    b = "https://bucket.s3.amazonaws.com/p.jpg?v=2&X-Amz-Signature=bbb&X-Amz-Date=2"
    assert lens_key(a) == lens_key(b)
    # The part that says which object is kept.
    assert "v=2" in stable_for_key(a)
    assert lens_key(a) != lens_key(
        "https://bucket.s3.amazonaws.com/p.jpg?v=3&X-Amz-Signature=aaa"
    )


@pytest.mark.parametrize(
    "value", ["Michael Petrie", "", "not a url", 10, None, {"nested": 1}]
)
def test_non_urls_pass_through_untouched(value) -> None:
    assert stable_for_key(value) == value


def test_a_text_query_is_unaffected() -> None:
    """Google search params carry no URL, so nothing about them changes."""
    a = cache_key({"engine": "google", "q": "Michael Petrie", "num": 10})
    b = cache_key({"engine": "google", "q": "Michael Petrie", "num": 10})
    c = cache_key({"engine": "google", "q": "Michael Petrie OSINT", "num": 10})
    assert a == b != c


# --- through the adapters --------------------------------------------------


@pytest.mark.parametrize("adapter_class", [GoogleLensAdapter, YandexImagesAdapter])
def test_a_repeat_run_of_one_photo_costs_nothing(adapter_class) -> None:
    """The point of the change: the second run must hit the same entry."""
    seen: list[dict] = []
    # Every engine enabled: this is about cache keys, not about which engines
    # this deployment calls.
    adapter = adapter_class(
        config=SerpApiConfig(disabled_engines=frozenset()),
        fetch=lambda params: seen.append(params) or {"visual_matches": []},
    )

    first = adapter.collect(Query(photo_url=signed(BASE, "token-one", 111)))
    second = adapter.collect(Query(photo_url=signed(BASE, "token-two", 222)))

    # The request still carries the real, signed URL each time.
    assert seen[0]["url"] != seen[1]["url"]
    assert "token=token-one" in seen[0]["url"]
    # But both resolve to one stored artifact.
    assert first.artifact_ref == second.artifact_ref
