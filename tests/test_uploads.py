"""Subject photo upload, and the honest failure when it cannot be published.

The awkward fact this feature exists around: reverse image search happens on
SerpAPI's servers, so the photo has to be fetchable from the public internet.
A path on disk is not, and neither is localhost. Half these tests are about
saying so clearly rather than failing quietly.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.adapters.base import Query
from app.adapters.serpapi import (
    GoogleLensAdapter,
    SerpApiConfig,
    YandexImagesAdapter,
)

#: These tests are about what a reverse image adapter does, not about which
#: engines this deployment happens to call, so they enable every engine.
ALL_ENGINES = SerpApiConfig(disabled_engines=frozenset())
from app.api import app
from app.models import SourceRunStatus
from app.uploads import (
    DEFAULT_RETENTION_SECONDS,
    MAX_UPLOAD_BYTES,
    UPLOADS_DIR,
    UploadRejected,
    is_publicly_reachable,
    public_url,
    sign,
    signed_path,
    store_image,
    stored_path,
    stored_uploads,
    sweep_expired,
    sweep_interval_seconds,
    retention_seconds,
    delete_upload,
    expired_uploads,
    unreachable_reason,
    verify,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def stable_signing_key(monkeypatch):
    monkeypatch.setenv("UPLOAD_SIGNING_KEY", "test-key-not-a-secret")


@pytest.fixture(autouse=True)
def no_public_base(monkeypatch):
    """Default to unconfigured, which is the case that must degrade well."""
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)


def png_bytes(width: int = 64, height: int = 64) -> bytes:
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (40, 90, 160)
    return cv2.imencode(".png", image)[1].tobytes()


def upload(client: TestClient, body: bytes, content_type: str, name: str = "p.png"):
    return client.post(
        "/subjects/photo",
        files={"file": (name, io.BytesIO(body), content_type)},
    )


# --- a valid upload returns a fetchable URL -------------------------------


def test_a_valid_upload_returns_a_fetchable_url(client, monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://webutation.example")

    response = upload(client, png_bytes(), "image/png")
    assert response.status_code == 200
    body = response.json()

    assert body["publicly_reachable"] is True
    assert body["public_url"].startswith("https://webutation.example/uploads/")
    assert "token=" in body["public_url"] and "expires=" in body["public_url"]
    assert body["reason"] is None
    assert len(body["digest"]) == 64

    # And the signed path actually serves the bytes back.
    fetched = client.get(body["path"])
    assert fetched.status_code == 200
    assert fetched.content == png_bytes()
    assert fetched.headers["Cache-Control"] == "private, no-store"
    assert fetched.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_the_url_is_unguessable_without_the_token(client) -> None:
    body = upload(client, png_bytes(), "image/png").json()
    assert client.get(f"/uploads/{body['filename']}").status_code == 422
    assert (
        client.get(
            f"/uploads/{body['filename']}", params={"expires": 1 << 40, "token": "x" * 32}
        ).status_code
        == 404
    )


def test_the_same_photo_stores_once(client) -> None:
    first = upload(client, png_bytes(), "image/png").json()
    second = upload(client, png_bytes(), "image/png").json()
    assert first["digest"] == second["digest"]
    assert first["filename"] == second["filename"]


# --- a non-image is rejected ----------------------------------------------


def test_a_non_image_is_rejected(client) -> None:
    """A content type is a claim by the uploader. Decoding is the check."""
    response = upload(client, b"this is not an image at all", "image/png")
    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]


def test_an_unaccepted_content_type_is_rejected(client) -> None:
    response = upload(client, png_bytes(), "application/pdf", name="x.pdf")
    assert response.status_code == 400
    assert "not accepted" in response.json()["detail"]


def test_an_empty_upload_is_rejected(client) -> None:
    response = upload(client, b"", "image/png")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_an_oversized_upload_is_rejected() -> None:
    with pytest.raises(UploadRejected, match="over the"):
        store_image(b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")


def test_a_renamed_executable_does_not_become_an_image() -> None:
    with pytest.raises(UploadRejected, match="could not be decoded"):
        store_image(b"MZ\x90\x00" + b"\x00" * 500, "image/jpeg")


# --- an expired token 404s -------------------------------------------------


def test_an_expired_token_404s(client) -> None:
    body = upload(client, png_bytes(), "image/png").json()
    filename = body["filename"]

    expired_at = int(time.time()) - 1
    response = client.get(
        f"/uploads/{filename}",
        params={"expires": expired_at, "token": sign(filename, expired_at)},
    )
    assert response.status_code == 404


def test_verify_rejects_expiry_and_tampering() -> None:
    now = 1_000_000
    valid_until = now + 600

    assert verify("a.png", valid_until, sign("a.png", valid_until), now=now) is True
    # Expired.
    assert verify("a.png", now - 1, sign("a.png", now - 1), now=now) is False
    # Signed for a different file.
    assert verify("b.png", valid_until, sign("a.png", valid_until), now=now) is False
    # Expiry extended without re-signing.
    assert verify("a.png", valid_until + 1, sign("a.png", valid_until), now=now) is False


def test_a_traversal_name_resolves_to_nothing() -> None:
    assert stored_path("../../.env") is None
    assert stored_path("not-a-digest.png") is None
    assert stored_path("a" * 64 + ".exe") is None


# --- PUBLIC_BASE_URL unset: failed, and never empty -----------------------


@pytest.mark.parametrize(
    "photo_url",
    [
        "D:/image001.png",
        "file:///D:/image001.png",
        "http://localhost:8000/uploads/abc.png",
        "http://127.0.0.1:8000/uploads/abc.png",
        "http://192.168.1.20:8000/uploads/abc.png",
    ],
)
@pytest.mark.parametrize(
    "adapter_class", [GoogleLensAdapter, YandexImagesAdapter]
)
def test_unreachable_photos_report_failed_not_empty(
    adapter_class, photo_url
) -> None:
    """The engine never saw the photo. Saying "empty" would claim it looked."""
    adapter = adapter_class(
        config=ALL_ENGINES, fetch=lambda params: {"visual_matches": []}
    )
    run = adapter.run(Query(name="Michael Petrie", photo_url=photo_url))

    assert run.source_run.status is SourceRunStatus.failed
    assert run.source_run.status is not SourceRunStatus.empty
    assert "photo not publicly reachable" in run.source_run.error_reason
    assert run.drafts == []


@pytest.mark.parametrize(
    "adapter_class", [GoogleLensAdapter, YandexImagesAdapter]
)
def test_no_photo_at_all_is_skipped_rather_than_failed(adapter_class) -> None:
    """A photo nobody can fetch is a failure. No photo is not.

    The first was asked and could not be answered; the second was never asked.
    Reporting them alike sends someone to fix a tunnel that is not the problem.
    """
    adapter = adapter_class(
        config=ALL_ENGINES, fetch=lambda params: {"visual_matches": []}
    )
    run = adapter.run(Query(name="Michael Petrie", photo_url=None))

    assert run.source_run.status is SourceRunStatus.skipped
    assert run.source_run.status is not SourceRunStatus.failed
    assert run.source_run.error_reason == "needs a photo"


def test_a_reachable_photo_lets_the_adapter_run() -> None:
    adapter = GoogleLensAdapter(fetch=lambda params: {"visual_matches": []})
    run = adapter.run(
        Query(photo_url="https://webutation.example/uploads/abc.png")
    )
    # Reached the engine, which genuinely returned nothing.
    assert run.source_run.status is SourceRunStatus.empty
    assert run.source_run.error_reason is None


def test_upload_says_reverse_image_search_is_unavailable(client) -> None:
    body = upload(client, png_bytes(), "image/png").json()

    assert body["publicly_reachable"] is False
    assert body["public_url"] is None
    assert "photo not publicly reachable" in body["reason"]
    # The photo was fine. Saying "no photo supplied" would send someone to
    # re-upload a file that was never the problem.
    assert "PUBLIC_BASE_URL is not set" in body["reason"]
    assert "no photo supplied" not in body["reason"]
    assert "PUBLIC_BASE_URL" in body["reverse_image_search"]
    # The upload is still usable locally, which is the point of saying so.
    assert body["path"].startswith("/uploads/")


def test_a_localhost_base_url_is_not_treated_as_public(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://localhost:8000")
    assert public_url("abc.png") is None


def test_reachability_verdicts() -> None:
    assert is_publicly_reachable("https://webutation.example/u/a.png") is True
    assert is_publicly_reachable("http://localhost:8000/a.png") is False
    assert is_publicly_reachable("D:/a.png") is False
    assert unreachable_reason("https://example.com/a.png") is None
    assert "localhost resolves only" in unreachable_reason("http://localhost/a.png")


# --- health ----------------------------------------------------------------


def test_health_reports_whether_reverse_search_can_work(client, monkeypatch) -> None:
    unset = client.get("/health").json()
    assert unset["reverse_image_search"] is False
    assert "PUBLIC_BASE_URL" in unset["reason"]

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://webutation.example")
    configured = client.get("/health").json()
    assert configured["reverse_image_search"] is True
    assert configured["reason"] is None


def test_signed_path_expires_when_told_to() -> None:
    path = signed_path("abc.png", ttl_seconds=120, now=1_000_000)
    assert "expires=1000120" in path


def test_a_localhost_base_url_names_localhost_as_the_problem(
    client, monkeypatch
) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://localhost:8000")
    body = upload(client, png_bytes(), "image/png").json()

    assert body["publicly_reachable"] is False
    assert "localhost resolves only" in body["reason"]


# --- retention -------------------------------------------------------------


def age_file(path, seconds: int) -> None:
    """Backdate a file's mtime, which is what retention is measured from."""
    import os

    past = time.time() - seconds
    os.utime(path, (past, past))


def test_the_default_retention_window_is_twenty_four_hours() -> None:
    assert DEFAULT_RETENTION_SECONDS == 24 * 60 * 60
    assert retention_seconds() == DEFAULT_RETENTION_SECONDS


def test_retention_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("UPLOAD_RETENTION_SECONDS", "60")
    assert retention_seconds() == 60
    # Nonsense falls back rather than disabling retention entirely.
    monkeypatch.setenv("UPLOAD_RETENTION_SECONDS", "not-a-number")
    assert retention_seconds() == DEFAULT_RETENTION_SECONDS
    monkeypatch.setenv("UPLOAD_RETENTION_SECONDS", "0")
    assert retention_seconds() == DEFAULT_RETENTION_SECONDS


def test_the_sweep_never_runs_less_often_than_the_window(monkeypatch) -> None:
    monkeypatch.setenv("UPLOAD_RETENTION_SECONDS", "60")
    monkeypatch.setenv("UPLOAD_SWEEP_INTERVAL_SECONDS", "999999")
    assert sweep_interval_seconds() == 60


def test_an_expired_upload_is_removed_by_the_sweep(client) -> None:
    body = upload(client, png_bytes(31, 31), "image/png").json()
    path = UPLOADS_DIR / body["filename"]
    assert path.exists()

    # Still inside the window: untouched.
    assert sweep_expired() == []
    assert path.exists()

    age_file(path, DEFAULT_RETENTION_SECONDS + 60)
    assert path in expired_uploads()

    removed = sweep_expired()
    assert body["filename"] in removed
    assert not path.exists()


def test_the_sweep_leaves_photos_inside_the_window(client) -> None:
    body = upload(client, png_bytes(32, 32), "image/png").json()
    path = UPLOADS_DIR / body["filename"]
    age_file(path, DEFAULT_RETENTION_SECONDS - 60)
    try:
        assert sweep_expired() == []
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_the_sweep_never_deletes_the_signing_key(client) -> None:
    key_file = UPLOADS_DIR / ".signing-key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(b"not-a-photo")
    age_file(key_file, DEFAULT_RETENTION_SECONDS * 10)

    sweep_expired()
    assert key_file.exists()
    assert key_file not in stored_uploads()


def test_re_uploading_restarts_the_retention_clock(client) -> None:
    body = upload(client, png_bytes(33, 33), "image/png").json()
    path = UPLOADS_DIR / body["filename"]
    age_file(path, DEFAULT_RETENTION_SECONDS + 60)
    assert path in expired_uploads()

    # The same bytes again. Somebody wants this photo now.
    upload(client, png_bytes(33, 33), "image/png")
    try:
        assert path not in expired_uploads()
        assert sweep_expired() == []
    finally:
        path.unlink(missing_ok=True)


def test_a_swept_photo_no_longer_serves(client) -> None:
    body = upload(client, png_bytes(34, 34), "image/png").json()
    assert client.get(body["path"]).status_code == 200

    age_file(UPLOADS_DIR / body["filename"], DEFAULT_RETENTION_SECONDS + 60)
    sweep_expired()

    # The token is still valid; the file is gone. 404 either way.
    assert client.get(body["path"]).status_code == 404


# --- explicit deletion -----------------------------------------------------


def test_delete_removes_the_file_and_subsequent_fetches_404(client) -> None:
    body = upload(client, png_bytes(35, 35), "image/png").json()
    path = UPLOADS_DIR / body["filename"]
    assert client.get(body["path"]).status_code == 200

    response = client.delete(f"/subjects/photo/{body['digest']}")
    assert response.status_code == 200
    assert response.json()["deleted"] == [body["filename"]]

    assert not path.exists()
    # The signed URL has not expired, and still serves nothing.
    assert client.get(body["path"]).status_code == 404


def test_deleting_something_absent_is_a_404(client) -> None:
    assert client.delete(f"/subjects/photo/{'a' * 64}").status_code == 404


def test_delete_rejects_anything_that_is_not_a_digest(client) -> None:
    for bad in ("../../.env", "short", "z" * 64):
        assert client.delete(f"/subjects/photo/{bad}").status_code == 404
    assert delete_upload("../../.env") == []


def test_health_reports_the_retention_window(client) -> None:
    body = client.get("/health").json()
    assert body["retention_seconds"] == DEFAULT_RETENTION_SECONDS
    assert body["sweep_interval_seconds"] <= body["retention_seconds"]
    assert "photos_stored" in body


def test_the_upload_response_states_when_the_file_dies(client) -> None:
    body = upload(client, png_bytes(36, 36), "image/png").json()
    try:
        assert body["deleted_after_seconds"] == DEFAULT_RETENTION_SECONDS
        # The link dies long before the file does, and both are bounded.
        assert body["expires_at"] - int(time.time()) < body["deleted_after_seconds"]
    finally:
        (UPLOADS_DIR / body["filename"]).unlink(missing_ok=True)


def test_the_sweep_runs_on_startup(monkeypatch) -> None:
    """A process down longer than the window must not serve stale photos."""
    calls: list[int] = []
    monkeypatch.setattr("app.api.sweep_expired", lambda: calls.append(1) or [])
    monkeypatch.setattr("app.api.sweep_interval_seconds", lambda: 3600)

    with TestClient(app):
        pass
    assert calls, "the lifespan sweep did not run at startup"
