"""Disk-backed cache for SerpAPI search requests.

Every response is stored verbatim so that a cached run is byte-identical to
the original network fetch, and so the recorded checksum stays verifiable
against the file on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_URL = "https://serpapi.com/search"
CACHE_DIR = Path("cache")
HISTORY_DIR = CACHE_DIR / "history"
REQUEST_TIMEOUT = 30

# Never written to disk and never part of a cache key.
_SECRET_KEYS = ("api_key",)

#: One of our own signed upload URLs. The digest in the path identifies the
#: image; everything around it identifies this particular link to it.
_UPLOAD_PATH = re.compile(r"/uploads/([0-9a-f]{64})\.[a-z0-9]+$")

#: Query parameters that authorise a request rather than describe what it asks
#: for. A signed upload URL mints a fresh token on every upload, so keying a
#: reverse image search on the whole URL makes the same photograph a cache miss
#: every time and buys the same search again at full price.
_AUTHORISING_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        "token",
        "expires",
        "signature",
        "sig",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-date",
        "x-goog-signature",
        "se",
        "st",
    }
)


def stable_for_key(value: Any) -> Any:
    """What a parameter identifies, with the part that merely authorises it removed.

    Reverse image search is keyed on which image was searched for, not on which
    link happened to point at it. One of our uploads reduces to its digest, so
    the same photograph hits the same cache entry across re-uploads, restarts
    and a changed tunnel hostname. Any other signed URL keeps its shape and
    loses only its credentials.

    Only the cache key is affected. The request still carries the real URL, or
    the search engine could not fetch anything.
    """
    if not isinstance(value, str) or "://" not in value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value

    match = _UPLOAD_PATH.search(parts.path)
    if match:
        return f"upload:{match.group(1)}"

    kept = [
        (name, item)
        for name, item in parse_qsl(parts.query, keep_blank_values=True)
        if name.lower() not in _AUTHORISING_QUERY_PARAMS
    ]
    if len(kept) == len(parse_qsl(parts.query, keep_blank_values=True)):
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), ""))


def response_error(payload: object) -> str | None:
    """The failure SerpAPI reported inside an otherwise successful response.

    SerpAPI answers a failed search with HTTP 200 and an ``error`` member, so
    ``raise_for_status`` sees nothing wrong and the body parses cleanly. Without
    this check a failure is indistinguishable from a result and gets stored as
    one.
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    return None


def _cache_key(params: dict) -> str:
    """SHA-256 of the params, keys sorted, secrets removed.

    ``sort_keys`` recurses, and ``ensure_ascii`` pins the encoding, so the
    same logical params always hash to the same digest. URL values are reduced
    to what they identify first: see :func:`stable_for_key`.
    """
    public = {
        k: stable_for_key(v) for k, v in params.items() if k not in _SECRET_KEYS
    }
    canonical = json.dumps(
        public, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_key(params: dict) -> str:
    """Public name for the cache key of a request.

    Adapters need it to record an artifact_ref on the drafts they emit, so a
    claim can be traced back to the stored bytes it came from.
    """
    return _cache_key(params)


def _archive_displaced(path: Path, key: str) -> Path | None:
    """Copy a soon-to-be-overwritten envelope into the history tree.

    A refresh would otherwise destroy the evidence it replaces. The displaced
    envelope is filed under ``cache/history/{key}/`` and named for the moment
    it was originally fetched, so every response ever served stays auditable.

    Returns the archive path, or ``None`` if there was nothing to displace.
    """
    if not path.exists():
        return None

    raw = path.read_text(encoding="utf-8")
    try:
        stamp = json.loads(raw)["fetched_at"]
    except (ValueError, KeyError, TypeError):
        stamp = None

    # Colons are illegal in Windows filenames, so an ISO timestamp cannot be
    # used as-is. Fall back to the current time if the envelope is unreadable.
    try:
        moment = datetime.fromisoformat(stamp) if stamp else datetime.now(timezone.utc)
    except ValueError:
        moment = datetime.now(timezone.utc)
    name = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    target_dir = HISTORY_DIR / key
    target_dir.mkdir(parents=True, exist_ok=True)

    # Microsecond precision makes a clash near-impossible, but never clobber
    # an existing archive entry if one somehow shares the name.
    target = target_dir / f"{name}.json"
    counter = 1
    while target.exists():
        target = target_dir / f"{name}-{counter}.json"
        counter += 1

    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(target)
    return target


def cached_get(params: dict, force_refresh: bool = False) -> dict:
    """Return the SerpAPI response for ``params``, fetching only on a miss.

    On a hit the stored body is parsed and returned with no network call. On a
    miss the API is queried and the raw body is written to ``cache/{key}.json``
    inside an envelope carrying the request params, the fetch time, and the
    body checksum.

    ``force_refresh`` skips the cache read and forces a network call, then
    writes the result as usual. The cache key is unchanged, so the refreshed
    response replaces the entry it bypassed rather than accumulating beside it.
    """
    key = _cache_key(params)
    path = CACHE_DIR / f"{key}.json"

    if path.exists() and not force_refresh:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        return json.loads(envelope["response_body"])

    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "SERPAPI_KEY is not set. Copy .env.example to .env and fill it in."
        )

    request_params = {k: v for k, v in params.items() if k not in _SECRET_KEYS}
    response = requests.get(
        SERPAPI_URL,
        params={**request_params, "api_key": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    # Hash the bytes as received, then decode as UTF-8 (required for JSON by
    # RFC 8259) so the stored text and the recorded digest describe the same
    # sequence of bytes.
    body_bytes = response.content
    body_text = body_bytes.decode("utf-8")
    payload = json.loads(body_text)

    # A failure is not cached, for the reason ``cached_image`` gives: a source
    # that is unreachable today may work tomorrow, and a permanent negative
    # entry hides that. Yandex refusing to fetch a photo is a fact about this
    # deployment's public URL, not about the photograph, and once stored it
    # reports "failed" forever — including after the deployment is fixed. The
    # caller still gets the payload and still reports the failure; only the
    # writing to disk is skipped, so the next run asks again.
    failure = response_error(payload)
    if failure is not None:
        return payload

    envelope: dict[str, Any] = {
        "params": request_params,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "response_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "response_body": body_text,
    }

    # Preserve whatever this write is about to replace. Copy rather than move,
    # so a failure below leaves the live entry intact.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _archive_displaced(path, key)

    # Write through a temp file so an interrupted run cannot leave a
    # half-written entry that later reads would treat as a valid hit.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)

    return payload


def purge_failed(dry_run: bool = True) -> list[tuple[Path, str]]:
    """Delete cache entries that stored a failed search.

    Entries written before failures were recognised still sit on disk and are
    still served, so refusing to write new ones does not clear the old. Returns
    the entries found as ``(path, error)``; with ``dry_run`` they are listed and
    left alone.

    Purging is deliberately a separate, explicit step rather than something a
    read does on its own. A read that treated a stored failure as a miss would
    re-buy the same failing search on every run, which for a source that is
    persistently broken is a standing charge. This way the sequence is: fix the
    deployment, purge, and the next run tries again.
    """
    found: list[tuple[Path, str]] = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = json.loads(envelope["response_body"])
        except (ValueError, KeyError, OSError):
            continue
        error = response_error(payload)
        if error is None:
            continue
        found.append((path, error))
        if not dry_run:
            path.unlink()
    return found


IMAGE_CACHE_DIR = CACHE_DIR / "images"

#: Many image hosts reject requests that do not identify themselves, so a
#: missing User-Agent shows up as an unreachable image rather than a refusal.
IMAGE_USER_AGENT = "webutation-prototype/0.1 (research; contact via repo)"


def _local_path(url: str) -> Path | None:
    """The local file this reference names, or None if it is a real URL."""
    if url.startswith("file://"):
        return Path(url_to_path(url))
    if "://" in url:
        return _own_upload_path(url)
    candidate = Path(url)
    return candidate if candidate.exists() else None


def _own_upload_path(url: str) -> Path | None:
    """The upload this URL points at, when the URL is one we published.

    The subject's photo is given to the face pipeline as the same signed public
    URL that SerpAPI is sent, because a search engine can only fetch a photo it
    can reach. Following that URL ourselves means asking the public internet for
    a file already sitting on this disk, and on a tunnelled host that round trip
    is the least reliable step in the run: one read timeout and the subject has
    no face, which silently costs every face comparison in the report.

    Only our own origin is accepted, and the name still has to pass the upload
    store's own validation, so this reads a file we wrote and nothing else.
    """
    from app.uploads import public_base_url, stored_path

    base = public_base_url()
    if not base:
        return None

    parts = urlsplit(url)
    base_parts = urlsplit(base)
    if (parts.scheme, parts.netloc) != (base_parts.scheme, base_parts.netloc):
        return None
    if not parts.path.startswith("/uploads/"):
        return None

    return stored_path(parts.path[len("/uploads/") :])


def url_to_path(url: str) -> str:
    """Strip a file:// prefix, tolerating the Windows three-slash form."""
    stripped = url[len("file://") :]
    return stripped[1:] if stripped[:1] == "/" and stripped[2:3] == ":" else stripped


def image_cache_key(url: str) -> str:
    """Cache key for an image, derived from its URL alone."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def cached_image(url: str, force_refresh: bool = False) -> bytes | None:
    """Return the bytes of an image, fetching only once per URL.

    Returns ``None`` when the image cannot be fetched. A failure is not cached:
    an unreachable host today may be reachable tomorrow, and a permanent
    negative entry would hide that. The cost of retrying is one request; the
    cost of a wrong permanent "no" is a face that never gets compared.

    Stored beside the bytes is a sidecar recording the URL, the fetch time and
    the digest, so a cached image is as traceable as a cached search response.
    """
    # A local file needs no cache and no network. Face verification can work
    # from disk even when reverse image search cannot, because SerpAPI needs a
    # URL it can reach and a local path is not one.
    local = _local_path(url)
    if local is not None:
        try:
            return local.read_bytes()
        except OSError:
            return None

    key = image_cache_key(url)
    blob = IMAGE_CACHE_DIR / f"{key}.bin"

    if blob.exists() and not force_refresh:
        return blob.read_bytes()

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": IMAGE_USER_AGENT,
                # Tunnel edges serve a click-through page to requests that
                # look like a browser. This says we are not one.
                "Bypass-Tunnel-Reminder": "true",
            },
        )
        response.raise_for_status()
        body = response.content
    except Exception:
        return None

    if not body:
        return None

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = blob.with_suffix(".bin.tmp")
    tmp.write_bytes(body)
    tmp.replace(blob)

    sidecar = IMAGE_CACHE_DIR / f"{key}.json"
    sidecar.write_text(
        json.dumps(
            {
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "content_type": response.headers.get("Content-Type"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return body


if __name__ == "__main__":  # pragma: no cover - operator tool
    import argparse

    parser = argparse.ArgumentParser(description="Cache maintenance.")
    parser.add_argument(
        "--purge-failed",
        action="store_true",
        help="delete cached entries whose stored response is a failure",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; without it the entries are only listed",
    )
    args = parser.parse_args()

    if args.purge_failed:
        entries = purge_failed(dry_run=not args.apply)
        verb = "Deleted" if args.apply else "Would delete"
        for path, error in entries:
            print(f"{verb} {path.name[:16]}  {error[:88]}")
        print(f"{verb.lower()} {len(entries)} failed entr{'y' if len(entries) == 1 else 'ies'}")
    else:
        parser.print_help()
