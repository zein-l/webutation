"""The subject's own photo is read from disk, not fetched back off the internet.

The photo is handed to the face pipeline as the same signed public URL that
SerpAPI is given, because a search engine can only look at an image it can
reach. Following that URL ourselves means asking the public internet for a file
already on this disk, and on a tunnelled host that round trip is the least
reliable step in the run.

It is also the most expensive one to lose. A read timeout there yields no
subject face, which yields no face comparison for any record, which is exactly
the signal the identity score leans on hardest.
"""

from __future__ import annotations

import pytest

from app.cache import _local_path, cached_image
from app.uploads import UPLOADS_DIR, public_url, store_image

BASE = "https://webutation-siu-demo.loca.lt"


@pytest.fixture(autouse=True)
def stable_signing_key(monkeypatch):
    monkeypatch.setenv("UPLOAD_SIGNING_KEY", "test-key-not-a-secret")


@pytest.fixture
def published(monkeypatch):
    """An upload on disk, plus the public URL it is published under."""
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype="uint8")
    ).save(buffer, format="PNG")
    body = buffer.getvalue()

    stored = store_image(body, "image/png")
    yield stored, public_url(stored.filename), body

    (UPLOADS_DIR / stored.filename).unlink(missing_ok=True)


def exploding_fetch(*args, **kwargs):
    raise AssertionError("the network was used for a file already on disk")


def test_our_own_upload_resolves_to_the_file_on_disk(published) -> None:
    stored, url, _ = published
    assert _local_path(url) == UPLOADS_DIR / stored.filename


def test_reading_it_needs_no_network(published, monkeypatch) -> None:
    """The one that matters: a flaky tunnel cannot cost us the subject face."""
    import app.cache

    _, url, body = published
    monkeypatch.setattr(app.cache.requests, "get", exploding_fetch)

    assert cached_image(url) == body


def test_a_stranger_hosting_the_same_path_is_still_fetched(published) -> None:
    """Only our own origin is short-circuited; anything else is a real URL."""
    stored, _, _ = published
    elsewhere = f"https://someone-else.example/uploads/{stored.filename}"
    assert _local_path(elsewhere) is None


def test_another_path_on_our_own_host_is_not_an_upload(published) -> None:
    assert _local_path(f"{BASE}/static/logo.png") is None


def test_a_traversal_in_the_upload_path_resolves_to_nothing(published) -> None:
    assert _local_path(f"{BASE}/uploads/../../.env") is None
    assert _local_path(f"{BASE}/uploads/%2e%2e%2f.env") is None


def test_a_name_that_is_not_a_digest_resolves_to_nothing(published) -> None:
    assert _local_path(f"{BASE}/uploads/portrait.png") is None
    assert _local_path(f"{BASE}/uploads/{'a' * 64}.exe") is None


def test_an_upload_we_do_not_hold_resolves_to_nothing(published) -> None:
    assert _local_path(f"{BASE}/uploads/{'b' * 64}.png") is None


def test_with_no_public_base_nothing_is_treated_as_ours(published, monkeypatch) -> None:
    stored, url, _ = published
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert _local_path(url) is None


def test_the_scheme_has_to_match_too(published, monkeypatch) -> None:
    """http://host and https://host are different origins."""
    stored, _, _ = published
    assert _local_path(f"http://webutation-siu-demo.loca.lt/uploads/{stored.filename}") is None


def test_ordinary_remote_images_are_unaffected(monkeypatch) -> None:
    """Records' images still come off the network, cached as before."""
    import app.cache

    class Response:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nremote"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(app.cache.requests, "get", lambda *a, **k: Response())
    assert cached_image("https://records.example/photo-not-ours.png") == Response.content
