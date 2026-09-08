"""HTTP surface. Currently one thing: getting a subject photo onto the web.

Reverse image search is the only part of this system that finds a person by
their face rather than by their name, and it is the part a local file cannot
reach. These two routes exist to close that gap: one accepts a photograph, the
other serves it back under a signed, expiring URL that SerpAPI's servers can
actually fetch.

Publishing a photograph of a person to the open internet is not a neutral act,
so the URL is unguessable and short-lived by construction rather than by
policy, and the response says plainly whether the photo is reachable and what
follows if it is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.cache import IMAGE_USER_AGENT
from app.meminfo import memory_report, storage_report
from app.ratelimit import GUARD, client_key
from app.report import ADAPTERS, available_fixtures, get_run, start_run
from app.uploads import (
    DEFAULT_TTL_SECONDS,
    MAX_UPLOAD_BYTES,
    UploadRejected,
    delete_upload,
    public_base_url,
    public_url,
    retention_seconds,
    signed_path,
    store_image,
    stored_path,
    stored_uploads,
    sweep_expired,
    sweep_interval_seconds,
    unreachable_reason,
    verify,
)

log = logging.getLogger(__name__)


async def _sweep_forever() -> None:
    """Delete photos past the retention window, on a loop.

    Runs once at startup so a process that has been down longer than the
    window does not serve stale photos while waiting for its first interval.
    Errors are logged and the loop continues: a sweep that dies on one bad
    file would silently stop enforcing retention altogether.
    """
    while True:
        try:
            removed = sweep_expired()
            if removed:
                log.info("retention sweep removed %d photo(s)", len(removed))
        except Exception:  # noqa: BLE001 - the loop must outlive one failure
            log.exception("retention sweep failed")
        await asyncio.sleep(sweep_interval_seconds())


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(_sweep_forever())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Webutation",
    description="Subject photo intake for reverse image search.",
    lifespan=lifespan,
)


def _allowed_origins() -> list[str]:
    """Origins the browser app is served from.

    Empty in development: Vite proxies the API, so the page and the API share an
    origin and the browser never sends a preflight. A deployment serves the
    static site from a different host, which makes every call cross-origin.

    Listed explicitly rather than "*", because these routes accept an upload and
    a token; a wildcard would let any page on the internet spend this
    deployment's search budget through a visitor's browser.
    """
    import os

    raw = os.environ.get("ALLOWED_ORIGINS", "")
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


_origins = _allowed_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        # X-Run-Token is the guard on the only route that spends money; without
        # it here the browser's preflight refuses the header and every search
        # fails with a CORS error rather than a 401.
        allow_headers=["Content-Type", "X-Run-Token"],
    )


@app.post("/subjects/photo")
async def upload_subject_photo(
    file: UploadFile = File(...),
    ttl_seconds: int = Query(DEFAULT_TTL_SECONDS, ge=60, le=24 * 60 * 60),
) -> JSONResponse:
    """Accept a subject photo and return the URL a search engine can fetch.

    ``public_url`` is null when no externally resolvable origin is configured.
    That is reported rather than papered over with a localhost URL, because a
    URL that looks usable and is not would surface later as a search that
    mysteriously found nothing.

    An upload is still worth making in that case: the photo is compared locally
    against every image the other sources return, so face verification runs
    even when reverse image search cannot.
    """
    body = await file.read()
    try:
        stored = store_image(body, file.content_type)
    except UploadRejected as rejected:
        raise HTTPException(status_code=400, detail=str(rejected)) from rejected

    now = int(time.time())
    absolute = public_url(stored.filename, ttl_seconds, now=now)

    # The photo was fine; the deployment is what cannot publish it. Saying
    # "no photo supplied" here would send someone to re-upload a good file.
    base = public_base_url()
    if absolute:
        reason = None
    elif not base:
        reason = (
            "photo not publicly reachable: PUBLIC_BASE_URL is not set, so no "
            "URL exists that a search engine could fetch this from"
        )
    else:
        reason = unreachable_reason(f"{base}/uploads/{stored.filename}")

    return JSONResponse(
        {
            "digest": stored.digest,
            "filename": stored.filename,
            "bytes": stored.byte_count,
            "content_type": stored.content_type,
            # Always usable by this process, for local face comparison.
            "path": signed_path(stored.filename, ttl_seconds, now=now),
            # Usable by SerpAPI only when a public origin is configured.
            "public_url": absolute,
            "publicly_reachable": absolute is not None,
            "reason": reason,
            "expires_at": now + ttl_seconds,
            # The URL dies in minutes; the file dies in hours. Both bounded.
            "deleted_after_seconds": retention_seconds(),
            "reverse_image_search": (
                "available"
                if absolute
                else "unavailable until PUBLIC_BASE_URL names an origin "
                "reachable from the public internet"
            ),
        }
    )


@app.delete("/subjects/photo/{digest}")
def delete_subject_photo(digest: str) -> JSONResponse:
    """Remove an uploaded photo now, rather than waiting for retention.

    Returns 404 when there is nothing to remove, so a caller can tell deletion
    from a typo. Deleting is the safe direction, so no confirmation is
    required beyond naming the digest.
    """
    removed = delete_upload(digest)
    if not removed:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"deleted": removed, "digest": digest.lower()})


@app.get("/uploads/{filename}")
def serve_upload(
    filename: str,
    expires: int = Query(...),
    token: str = Query(...),
) -> FileResponse:
    """Serve an uploaded photo to anyone holding a valid, unexpired token.

    An expired or wrong token is answered with 404 rather than 403: whether a
    given digest exists is itself information about who has been searched for.
    """
    if not verify(filename, expires, token):
        raise HTTPException(status_code=404, detail="not found")

    path = stored_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(
        path,
        headers={
            # Signed and expiring, so it must not be cached by anything in
            # between and must not be indexed.
            "Cache-Control": "private, no-store",
            "X-Robots-Tag": "noindex, nofollow",
            # Best effort against a tunnel edge serving an interstitial in
            # place of the image. Honestly: this is a *response* header, and
            # the check a tunnel makes is on the *request*, which belongs to
            # whichever engine is fetching. It costs nothing and may help a
            # proxy in between; it cannot make Yandex send anything.
            "Bypass-Tunnel-Reminder": "true",
            "Server": IMAGE_USER_AGENT,
        },
    )


@app.get("/health")
def health() -> dict:
    """Whether this deployment can support reverse image search at all."""
    base = public_base_url()
    limits = GUARD.snapshot()
    reason = unreachable_reason(f"{base}/uploads/x.jpg") if base else None
    reachable = bool(base) and reason is None
    return {
        "status": "ok",
        "public_base_url": base or None,
        "reverse_image_search": reachable,
        "reason": None if reachable else (reason or "PUBLIC_BASE_URL is not set"),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "retention_seconds": retention_seconds(),
        "sweep_interval_seconds": sweep_interval_seconds(),
        "photos_stored": len(stored_uploads()),
        # So an operator can see the budget without reading the logs.
        "limits": limits,
        # Peak resident size against the container's ceiling. Reported because
        # a run that vanishes looks the same whether the process crashed, was
        # redeployed, or was killed for memory, and only this tells them apart.
        "memory": memory_report(),
        # Which commit is actually serving. Verifying a deploy by watching the
        # process restart proves only that something restarted; after two
        # rounds of "is the fix live yet?" this says so directly. Render sets
        # RENDER_GIT_COMMIT; anywhere else it is null rather than guessed.
        "commit": (os.environ.get("RENDER_GIT_COMMIT") or "")[:7] or None,
        # Whether the cache survives a deploy, read from the mount table
        # rather than inferred from configuration.
        "storage": storage_report(),
    }


@app.get("/fixtures")
def list_fixtures() -> JSONResponse:
    """Edge cases a reviewer can trigger without spending a search."""
    return JSONResponse({"fixtures": available_fixtures()})


@app.post("/runs")
def create_run(
    request: Request, payload: dict = Body(default_factory=dict)
) -> JSONResponse:
    """Start a search. Returns immediately; poll the run for progress.

    Either a photo or some text is enough. Asking for neither is the one thing
    that cannot be searched, so it is refused with that reason.

    This is the only route that spends money, so it is the only one that is
    guarded. A shared token decides who may ask; a daily cap decides how much
    may be spent whatever the token. See :mod:`app.ratelimit`.
    """
    denied = GUARD.check_token(request.headers.get("x-run-token"))
    if denied:
        raise HTTPException(status_code=401, detail=denied)

    caller = client_key(request)
    denied = GUARD.check_quota(caller)
    if denied:
        raise HTTPException(status_code=429, detail=denied)
    fixture = (payload.get("fixture") or "").strip() or None
    name = (payload.get("name") or "").strip()
    address = (payload.get("address") or "").strip()
    context = (payload.get("context") or "").strip()
    photo_url = (payload.get("photo_url") or "").strip()

    if not fixture and not any((name, address, context, photo_url)):
        raise HTTPException(
            status_code=400,
            detail="Add a photo or a name to search on. Either one is enough.",
        )

    if fixture and fixture not in {f["id"] for f in available_fixtures()}:
        raise HTTPException(status_code=404, detail=f"No fixture named {fixture!r}")

    requested = payload.get("adapters") or list(ADAPTERS)
    unknown = [a for a in requested if a not in ADAPTERS]
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Unknown source(s): {', '.join(unknown)}"
        )

    # Counted only once the request is known to be a real, runnable search,
    # so a malformed payload cannot exhaust the day's budget.
    GUARD.record(caller)

    run = start_run(
        photo_digest=(payload.get("photo_digest") or "").strip() or None,
        name=name,
        address=address,
        context=context,
        photo_url=photo_url,
        adapters=requested,
        fixture=fixture,
        max_queries=int(payload.get("max_queries") or 6),
    )
    return JSONResponse({"run_id": run.id, "status": run.status}, status_code=202)


@app.get("/runs/{run_id}")
def read_run(run_id: str) -> JSONResponse:
    """Progress while it runs, and the report once it is done."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No run with that id")
    return JSONResponse(run.snapshot())
