"""Storing an uploaded subject photo and making it reachable.

The constraint this module exists for: SerpAPI's Lens and Yandex engines fetch
the image from *their* servers. A path on the operator's disk, and equally a
URL on localhost, is invisible to them. So an upload has to become a real,
publicly resolvable URL before reverse image search can run at all.

Two consequences shape everything here.

* **The URL must be unguessable and short-lived.** It points at a photograph of
  a person, published to the open internet so a third party can fetch it. A
  signed token in the query bounds both who can retrieve it and for how long.
* **Failing to be reachable is not the same as finding nothing.** When no
  public base URL is configured, the reverse-image adapters must say they could
  not run, never that they ran and saw nothing. :func:`unreachable_reason`
  gives them the words.

An upload is still worth making when the base URL is missing: the photo is
compared locally against every image the other sources turn up, so face
verification works even when reverse image search cannot.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from app.cache import CACHE_DIR

UPLOADS_DIR = CACHE_DIR / "uploads"

#: Content types accepted, and the extension each is stored under. Deliberately
#: short: every additional format is another decoder to trust.
ALLOWED_CONTENT_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

#: How long a signed URL stays valid. Long enough for a search to fetch it,
#: short enough that a leaked link expires before it is useful.
DEFAULT_TTL_SECONDS = 15 * 60

#: How long an uploaded photograph is kept on disk before it is deleted.
#: Distinct from the URL lifetime above and much longer: the link stops working
#: in minutes, the file survives long enough to re-run a search against it.
#: Both are bounded on purpose. A photograph of a person accumulating
#: indefinitely because nobody chose a number is not a neutral default.
DEFAULT_RETENTION_SECONDS = 24 * 60 * 60

#: How often the background sweep runs. Retention is a promise about the
#: longest a file may survive, so the sweep has to be frequent enough that the
#: promise is roughly true rather than aspirational.
DEFAULT_SWEEP_INTERVAL_SECONDS = 15 * 60

#: Hosts that resolve only on the machine running this, and are therefore
#: useless to a third-party fetcher however correct the URL looks.
_LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


@dataclass(frozen=True)
class StoredImage:
    """One accepted upload."""

    digest: str
    extension: str
    byte_count: int
    content_type: str

    @property
    def filename(self) -> str:
        return f"{self.digest}{self.extension}"

    @property
    def path(self) -> Path:
        return UPLOADS_DIR / self.filename


class UploadRejected(Exception):
    """The upload was not a usable image. The message says why."""


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------


def signing_key() -> bytes:
    """The HMAC key for upload URLs.

    Taken from ``UPLOAD_SIGNING_KEY`` when set. Otherwise one is generated and
    kept on disk, so restarting the server does not silently invalidate every
    URL already handed out. A per-process random key would do exactly that.
    """
    from_env = os.environ.get("UPLOAD_SIGNING_KEY")
    if from_env:
        return from_env.encode("utf-8")

    key_file = UPLOADS_DIR / ".signing-key"
    if key_file.exists():
        return key_file.read_bytes()

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_bytes(32)
    key_file.write_bytes(generated)
    return generated


def sign(filename: str, expires_at: int) -> str:
    """Token authorising one filename until one moment."""
    message = f"{filename}:{expires_at}".encode("utf-8")
    return hmac.new(signing_key(), message, hashlib.sha256).hexdigest()[:32]


def verify(filename: str, expires_at: int, token: str, now: int | None = None) -> bool:
    """Constant-time check of a token, and of whether it has expired."""
    now = int(time.time()) if now is None else now
    if expires_at < now:
        return False
    return hmac.compare_digest(sign(filename, expires_at), token)


def signed_path(
    filename: str, ttl_seconds: int = DEFAULT_TTL_SECONDS, now: int | None = None
) -> str:
    """Path plus query that will serve this file until the token expires."""
    now = int(time.time()) if now is None else now
    expires_at = now + ttl_seconds
    return f"/uploads/{filename}?expires={expires_at}&token={sign(filename, expires_at)}"


# --------------------------------------------------------------------------
# Reachability
# --------------------------------------------------------------------------


def retention_seconds() -> int:
    """How long an upload is kept. ``UPLOAD_RETENTION_SECONDS`` overrides."""
    raw = os.environ.get("UPLOAD_RETENTION_SECONDS", "").strip()
    if not raw:
        return DEFAULT_RETENTION_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_RETENTION_SECONDS
    return value if value > 0 else DEFAULT_RETENTION_SECONDS


def sweep_interval_seconds() -> int:
    """How often to sweep, never longer than the retention window itself."""
    raw = os.environ.get("UPLOAD_SWEEP_INTERVAL_SECONDS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_SWEEP_INTERVAL_SECONDS
    except ValueError:
        value = DEFAULT_SWEEP_INTERVAL_SECONDS
    if value <= 0:
        value = DEFAULT_SWEEP_INTERVAL_SECONDS
    return max(1, min(value, retention_seconds()))


def public_base_url() -> str:
    """The externally resolvable origin this service is published under."""
    return os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")


def unreachable_reason(url: str | None) -> str | None:
    """Why a third party could not fetch this URL, or None if it could.

    The message is the one the reverse-image adapters report, so it says what
    is wrong rather than that something is.
    """
    if not url:
        return "photo not publicly reachable: no photo supplied"

    if "://" not in url or url.startswith("file://"):
        return (
            "photo not publicly reachable: a local file cannot be fetched by "
            "the search engine, which retrieves the image from its own servers"
        )

    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return "photo not publicly reachable: no host in the URL"
    if host in _LOCAL_HOSTNAMES:
        return (
            "photo not publicly reachable: localhost resolves only on this "
            "machine. Set PUBLIC_BASE_URL to an externally resolvable origin"
        )
    try:
        address = ip_address(host)
    except ValueError:
        return None
    if address.is_loopback or address.is_private or address.is_link_local:
        return (
            f"photo not publicly reachable: {host} is not routable from the "
            "public internet. Set PUBLIC_BASE_URL to an externally "
            "resolvable origin"
        )
    return None


def is_publicly_reachable(url: str | None) -> bool:
    return unreachable_reason(url) is None


def public_url(
    filename: str, ttl_seconds: int = DEFAULT_TTL_SECONDS, now: int | None = None
) -> str | None:
    """Absolute signed URL, or None when no public origin is configured.

    None is returned rather than a localhost URL, because a URL that looks
    usable and is not is worse than an honest absence.
    """
    base = public_base_url()
    if not base:
        return None
    candidate = f"{base}{signed_path(filename, ttl_seconds, now)}"
    return candidate if is_publicly_reachable(candidate) else None


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def _decodes_as_image(body: bytes) -> bool:
    """Confirm the bytes really are an image, not just labelled as one.

    A content type is a claim by the uploader. Decoding is the check.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - depends on environment
        raise UploadRejected(
            "cannot verify the image: opencv is not installed"
        ) from None
    buffer = np.frombuffer(body, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) is not None


def store_image(body: bytes, content_type: str | None) -> StoredImage:
    """Validate and store one uploaded image, keyed by its own digest.

    Content-addressed, so the same photo uploaded twice occupies one file and
    yields one URL. Raises :class:`UploadRejected` with a usable message on
    anything that is not a real image of an accepted type and size.
    """
    if not body:
        raise UploadRejected("the uploaded file is empty")
    if len(body) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"the image is {len(body)} bytes, over the "
            f"{MAX_UPLOAD_BYTES} byte limit"
        )

    normalized = (content_type or "").split(";")[0].strip().lower()
    if normalized not in ALLOWED_CONTENT_TYPES:
        raise UploadRejected(
            f"content type {normalized or 'missing'!r} is not accepted; "
            f"use one of {', '.join(sorted(set(ALLOWED_CONTENT_TYPES)))}"
        )

    if not _decodes_as_image(body):
        raise UploadRejected(
            "the file could not be decoded as an image, whatever its content "
            "type claimed"
        )

    digest = hashlib.sha256(body).hexdigest()
    extension = ALLOWED_CONTENT_TYPES[normalized]
    stored = StoredImage(
        digest=digest,
        extension=extension,
        byte_count=len(body),
        content_type=normalized,
    )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if stored.path.exists():
        # Same bytes, so nothing to write, but the retention clock restarts.
        # Somebody asking for this photo again should not have it swept out
        # from under them because an identical copy was stored yesterday.
        os.utime(stored.path, None)
    else:
        tmp = stored.path.with_suffix(stored.extension + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(stored.path)
    return stored


# --------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------


def stored_uploads() -> list[Path]:
    """Every stored photo. Excludes the signing key and partial writes."""
    if not UPLOADS_DIR.exists():
        return []
    extensions = set(ALLOWED_CONTENT_TYPES.values())
    return [
        path
        for path in UPLOADS_DIR.iterdir()
        if path.is_file() and path.suffix in extensions and len(path.stem) == 64
    ]


def expired_uploads(now: float | None = None, ttl: int | None = None) -> list[Path]:
    """Photos older than the retention window."""
    now = time.time() if now is None else now
    ttl = retention_seconds() if ttl is None else ttl
    cutoff = now - ttl
    expired = []
    for path in stored_uploads():
        try:
            if path.stat().st_mtime < cutoff:
                expired.append(path)
        except OSError:
            continue
    return expired


def sweep_expired(now: float | None = None, ttl: int | None = None) -> list[str]:
    """Delete photos past the retention window. Returns what was removed.

    Errors on individual files are swallowed: a sweep that stops at the first
    locked file leaves everything after it in place, which is the opposite of
    what a retention window is for.
    """
    removed: list[str] = []
    for path in expired_uploads(now=now, ttl=ttl):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    return removed


def delete_upload(digest: str) -> list[str]:
    """Remove one photo by digest, whatever extension it was stored under.

    Returns the filenames removed, empty when there was nothing to remove.
    Deleting a photograph is the safe direction, so this is deliberately
    permissive about being called for something already gone.
    """
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        return []
    removed: list[str] = []
    for extension in sorted(set(ALLOWED_CONTENT_TYPES.values())):
        path = UPLOADS_DIR / f"{digest.lower()}{extension}"
        if path.exists():
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                continue
    return removed


def path_for_digest(digest: str) -> Path | None:
    """The stored file for a digest, whatever extension it landed under.

    Face comparison reads the photo from disk rather than over HTTP. The two
    concerns are separate: whether SerpAPI can fetch a public URL says nothing
    about whether this process can open its own file, and tying them together
    means a dead tunnel silently disables face matching too.
    """
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        return None
    for extension in sorted(set(ALLOWED_CONTENT_TYPES.values())):
        path = UPLOADS_DIR / f"{digest.lower()}{extension}"
        if path.exists():
            return path
    return None


def stored_path(filename: str) -> Path | None:
    """The file this name refers to, or None if the name is not one of ours.

    The name must be a digest plus an accepted extension, which leaves nothing
    for a path traversal to work with.
    """
    candidate = Path(filename).name
    if candidate != filename:
        return None
    stem, _, extension = candidate.rpartition(".")
    extension = f".{extension}"
    if extension not in set(ALLOWED_CONTENT_TYPES.values()):
        return None
    if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem):
        return None
    path = UPLOADS_DIR / candidate
    return path if path.exists() else None
