"""SerpAPI adapters: Google web search, Google Lens, Yandex reverse image.

All three share one client, one response shape and one mapping, which is why
adding the second and third costs almost nothing. Every call goes through
:func:`app.cache.cached_get`, so a repeat run is free and every draft can be
traced back to stored bytes.

**Extraction here is deliberately thin.** A search result structurally contains
a title, a URL and sometimes an image. It does not contain an employer, a job
title or a city; those appear inside a prose snippet written for humans. Parsing
them out would manufacture claims that no source actually made, and they would
then be scored, corroborated and reported as though a source had. So snippets
are kept as context against the record and never become assertions.

What is emitted per result: the displayed name, the page URL, and any image
URL. Nothing else.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

from app.adapters.base import (
    AdapterNotApplicable,
    AdapterRun,
    AssertionDraft,
    Query,
    RawResponse,
    SourceRunDraft,
)
from app.cache import cache_key, cached_get
from app.uploads import unreachable_reason
from app.models import AccessCategory, EvidenceKind, SourceOrigin, SourceRunStatus

#: Suffixes needing three labels to reach the registrable domain. Short and
#: incomplete by design: a full public suffix list is a dependency, and being
#: wrong here splits one origin into two rather than merging two into one.
_MULTI_LABEL_SUFFIXES: frozenset[str] = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "ne.jp", "or.jp",
        "com.au", "net.au", "org.au", "co.nz", "com.br", "com.mx", "com.ar",
        "co.in", "co.za", "com.sg", "com.tr", "co.kr", "com.cn", "com.hk",
    }
)

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

#: Separators a page title uses between the subject and the rest of the title.
_TITLE_SEPARATORS: tuple[str, ...] = (" - ", " – ", " — ", " | ", " · ", " :: ")


def registrable_domain(url: str | None) -> str | None:
    """The registrable domain of a URL: linkedin.com, not the full address.

    Returns None when no domain can be read, which is what makes lineage
    ``unknown`` rather than a guess. Bare hosts, IP addresses and malformed
    URLs all fall into that case.
    """
    if not url:
        return None
    try:
        host = urlsplit(url if "//" in url else f"//{url}").hostname
    except ValueError:
        return None
    if not host:
        return None

    host = host.lower().strip(".")
    if _IPV4.match(host) or ":" in host:
        return None
    if host.startswith("www."):
        host = host[4:]

    labels = host.split(".")
    if len(labels) < 2:
        return None
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _default_evidence_map() -> dict[str, EvidenceKind]:
    """Domain to how that domain came to know what it publishes."""
    social_and_profile = [
        "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
        "github.com", "medium.com", "about.me", "behance.net", "dribbble.com",
        "youtube.com", "tiktok.com", "reddit.com", "substack.com", "threads.net",
    ]
    news_and_press = [
        "nytimes.com", "washingtonpost.com", "wsj.com", "bbc.co.uk", "cnn.com",
        "reuters.com", "apnews.com", "bloomberg.com", "forbes.com", "ft.com",
        "theguardian.com", "techcrunch.com", "wired.com", "npr.org", "axios.com",
    ]
    aggregators = [
        "spokeo.com", "whitepages.com", "beenverified.com", "intelius.com",
        "peoplefinders.com", "radaris.com", "mylife.com", "truepeoplesearch.com",
        "zoominfo.com", "rocketreach.co", "peopledatalabs.com", "pipl.com",
        "crunchbase.com", "apollo.io", "signalhire.com", "fastpeoplesearch.com",
    ]
    return {
        **{d: EvidenceKind.self_reported for d in social_and_profile},
        **{d: EvidenceKind.secondhand for d in news_and_press},
        **{d: EvidenceKind.republished for d in aggregators},
    }


@dataclass(frozen=True)
class SerpApiConfig:
    """Every judgement the adapters make, in one place."""

    #: How a domain comes to know what it publishes. Read for reliability only,
    #: never for lineage: the two axes stay separate all the way down.
    evidence_kind_by_domain: Mapping[str, EvidenceKind] = field(
        default_factory=_default_evidence_map
    )
    #: An unrecognised domain is unknown, never assumed.
    default_evidence_kind: EvidenceKind = EvidenceKind.unknown

    #: SerpAPI answers 200 with an error string when a query simply matched
    #: nothing. That is an empty result, not a failure, and the two must not
    #: collapse. Anything else in an error field is a genuine failure.
    empty_error_markers: tuple[str, ...] = (
        "hasn't returned any results",
        "has not returned any results",
        "did not return any results",
        "no results found",
    )

    num_results: int = 10
    title_separators: tuple[str, ...] = _TITLE_SEPARATORS
    access_category: AccessCategory = AccessCategory.PUBLIC_WEB

    #: Engines not to call at all, by engine name.
    #:
    #: Yandex is off because it will not fetch this deployment's photo URL. It
    #: answers every reverse image search with "The URL does not refer to an
    #: image, or the image is not publicly accessible", while Google Lens
    #: fetches the same URL without complaint.
    #:
    #: This was first seen through a development tunnel and blamed on the
    #: tunnel. That was wrong: tested against the deployed onrender.com domain,
    #: on a URL confirmed to return 200 image/png one second earlier, Yandex
    #: gave the same refusal. The host was never the cause.
    #:
    #: Untested hypothesis: the URL carries a signed query string
    #: (?expires=…&token=…) and Yandex may want a bare image URL. Finding out
    #: costs a live search, and the test would mean publishing an unsigned,
    #: unexpiring URL to a photograph of a real person — which is the one thing
    #: the upload design exists to refuse. Since failures are no longer cached,
    #: leaving it on buys the refusal on every run.
    #:
    #: This is a deployment judgement, not a verdict on the source: Yandex has
    #: different recall from Lens and is worth having if someone works out why.
    #: ``disabled_engines=frozenset()`` turns it back on.
    disabled_engines: frozenset[str] = frozenset({"yandex_images"})


def _display_name(title: str | None, config: SerpApiConfig) -> str | None:
    """The leading segment of a page title.

    Splitting on a title separator is a display convention, not snippet
    parsing: "Marcus Webb - Operations Manager at Meridian" is two fields the
    site joined for rendering. The full title is kept as the raw value, so
    nothing is discarded and the split can be revisited.
    """
    if not title:
        return None
    head = title
    for separator in config.title_separators:
        if separator in head:
            head = head.split(separator, 1)[0]
    head = " ".join(head.split())
    return head or None


def _failure_reason(exc: Exception) -> str:
    """A sentence for the report, never a raw traceback fragment.

    Messages this code raises are already written for a reader, so they pass
    through unchanged. Anything thrown from below is wrapped in prose, with the
    exception class kept because an operator chasing a real fault needs it.
    """
    if isinstance(exc, ValueError):
        return str(exc)
    return f"the source could not be reached ({type(exc).__name__}): {exc}"


class _SerpApiAdapter:
    """Shared client and mapping. Subclasses only choose the engine and params."""

    engine: str = ""
    #: A general search returns many origins, so there is no adapter-level
    #: origin to fall back on. Leaving this None keeps unnamed lineage unknown
    #: rather than crediting it to the adapter that happened to find it.
    default_origin: str | None = None

    def __init__(
        self,
        config: SerpApiConfig | None = None,
        fetch: Callable[[dict], dict] = cached_get,
    ) -> None:
        """``fetch`` is injected so tests can replay saved responses."""
        self.config = config or SerpApiConfig()
        self._fetch = fetch
        self.name = f"serpapi:{self.engine}"

    # --- request ----------------------------------------------------------

    def _params(self, query: Query) -> dict:
        raise NotImplementedError

    def collect(self, query: Query) -> RawResponse:
        """Fetch through the cache and return the response unmodified.

        A disabled engine declines before any request is built, so it reports
        as not applicable rather than as a failure: nothing went wrong, this
        deployment simply does not ask it.
        """
        if self.engine in self.config.disabled_engines:
            raise AdapterNotApplicable(
                "disabled for this deployment: the reverse image host rejects "
                "photo URLs served from here, so the search is a paid failure. "
                "Re-enable once uploads are served from a real domain."
            )
        params = self._params(query)
        payload = self._fetch(params)
        return RawResponse(
            payload=payload,
            fetched_at=datetime.now(timezone.utc),
            # The cache key, so a draft points at the stored bytes it came from.
            artifact_ref=cache_key(params),
            source=self.name,
        )

    # --- per-result mapping ------------------------------------------------

    def _record_ref(self, result: Mapping, index: int) -> str:
        """One search result is one record.

        Derived from the link, which is what identifies a result. When a result
        carries no link, a hash of the block itself is stable across repeat
        collections of identical bytes and unique within the response.
        """
        link = result.get("link") or result.get("source_page")
        if link:
            return f"{self.name}:{link}"
        blob = json.dumps(result, sort_keys=True, ensure_ascii=True, default=str)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return f"{self.name}:sha256:{digest}:{index}"

    def _lineage(self, domain: str | None) -> SourceOrigin:
        """Identifiable domain means a nameable origin. Otherwise unknown.

        Nothing is inferred about copying. Detecting that one site republished
        another needs a corpus, and guessing here would silently change
        corroboration counts.
        """
        return SourceOrigin.known_origin if domain else SourceOrigin.unknown

    def _evidence_kind(self, domain: str | None) -> EvidenceKind:
        if not domain:
            return self.config.default_evidence_kind
        return self.config.evidence_kind_by_domain.get(
            domain, self.config.default_evidence_kind
        )

    def _observed_at(self, result: Mapping) -> datetime | None:
        """Only when the result genuinely carries a date.

        Search results usually do not, so null is the normal case and freshness
        will correctly read as unknown. The fetch time is never substituted: a
        page found today says nothing about when its claim was recorded.
        """
        raw = result.get("date")
        if not isinstance(raw, str) or not raw.strip():
            return None
        text = raw.strip()
        for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            # Relative dates such as "3 days ago" are not resolved here; an
            # unparsed date stays unknown rather than becoming approximately now.
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _drafts_for_result(
        self, result: Mapping, index: int, *, image_keys: tuple[str, ...] = ()
    ) -> list[AssertionDraft]:
        """The three things a result structurally contains, and nothing more."""
        link = result.get("link") or result.get("source_page")
        domain = registrable_domain(link)
        record_ref = self._record_ref(result, index)
        publisher = result.get("source") or result.get("source_name") or domain

        common = {
            "record_ref": record_ref,
            "publisher": publisher,
            "origin_key": domain,
            "source_origin": self._lineage(domain),
            "evidence_kind": self._evidence_kind(domain),
            "observed_at": self._observed_at(result),
            "access_category": self.config.access_category,
        }

        drafts: list[AssertionDraft] = []

        title = result.get("title")
        name = _display_name(title if isinstance(title, str) else None, self.config)
        if name:
            drafts.append(
                AssertionDraft(
                    predicate="name",
                    raw_value=title,
                    normalized_value=name.casefold(),
                    **common,
                )
            )

        if link:
            drafts.append(
                AssertionDraft(
                    predicate="profile_url",
                    raw_value=link,
                    normalized_value=link.casefold(),
                    **common,
                )
            )

        for key in image_keys:
            image = result.get(key)
            if isinstance(image, str) and image:
                drafts.append(
                    AssertionDraft(
                        predicate="image_url",
                        raw_value=image,
                        normalized_value=image.casefold(),
                        **common,
                    )
                )

        return drafts

    # --- response shape ----------------------------------------------------

    def _result_blocks(self, payload: Mapping) -> list[tuple[Mapping, tuple[str, ...]]]:
        """Results this engine returns, paired with their image fields."""
        blocks: list[tuple[Mapping, tuple[str, ...]]] = []
        for result in payload.get("organic_results") or []:
            if isinstance(result, Mapping):
                blocks.append((result, ("thumbnail",)))
        # Inline images carry a thumbnail, an original, and the page they were
        # found on. All three matter: the images feed face comparison later and
        # the page is the publisher that showed them.
        for result in payload.get("inline_images") or []:
            if isinstance(result, Mapping):
                blocks.append((result, ("original", "thumbnail")))
        return blocks

    def normalize(self, raw: RawResponse) -> list[AssertionDraft]:
        """Parse a response into drafts. No network, no snippet parsing."""
        drafts: list[AssertionDraft] = []
        for index, (result, image_keys) in enumerate(
            self._result_blocks(raw.payload)
        ):
            drafts.extend(
                self._drafts_for_result(result, index, image_keys=image_keys)
            )
        return drafts

    def record_context(self, raw: RawResponse) -> dict[str, str]:
        """Snippets, kept against their record and never turned into claims."""
        context: dict[str, str] = {}
        for index, (result, _) in enumerate(self._result_blocks(raw.payload)):
            snippet = result.get("snippet")
            if isinstance(snippet, str) and snippet.strip():
                context[self._record_ref(result, index)] = snippet.strip()
        return context

    # --- run outcome -------------------------------------------------------

    def _payload_error(self, payload: Mapping) -> str | None:
        error = payload.get("error")
        return error.strip() if isinstance(error, str) and error.strip() else None

    def run(self, query: Query) -> AdapterRun:
        """Collect and normalize, reporting the outcome instead of raising.

        Three distinct results, never two: found with a count, empty when the
        source answered with nothing, failed when it did not answer at all.
        """
        ran_at = datetime.now(timezone.utc)
        try:
            raw = self.collect(query)
        except AdapterNotApplicable as declined:
            return AdapterRun(
                source_run=SourceRunDraft(
                    source=self.name,
                    status=SourceRunStatus.skipped,
                    error_reason=str(declined),
                    ran_at=ran_at,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed source is a result
            return AdapterRun(
                source_run=SourceRunDraft(
                    source=self.name,
                    status=SourceRunStatus.failed,
                    error_reason=_failure_reason(exc),
                    ran_at=ran_at,
                )
            )

        error = self._payload_error(raw.payload)
        if error is not None:
            matched_empty = any(
                marker in error.casefold()
                for marker in self.config.empty_error_markers
            )
            if not matched_empty:
                return AdapterRun(
                    source_run=SourceRunDraft(
                        source=self.name,
                        status=SourceRunStatus.failed,
                        error_reason=error,
                        ran_at=ran_at,
                    ),
                    raw=raw,
                )
            return AdapterRun(
                source_run=SourceRunDraft(
                    source=self.name,
                    status=SourceRunStatus.empty,
                    result_count=0,
                    ran_at=ran_at,
                ),
                raw=raw,
            )

        blocks = self._result_blocks(raw.payload)
        if not blocks:
            return AdapterRun(
                source_run=SourceRunDraft(
                    source=self.name,
                    status=SourceRunStatus.empty,
                    result_count=0,
                    ran_at=ran_at,
                ),
                raw=raw,
            )

        return AdapterRun(
            source_run=SourceRunDraft(
                source=self.name,
                status=SourceRunStatus.found,
                result_count=len(blocks),
                ran_at=ran_at,
            ),
            drafts=self.normalize(raw),
            raw=raw,
            record_context=self.record_context(raw),
        )


class GoogleSearchAdapter(_SerpApiAdapter):
    """Google web search by name and context."""

    engine = "google"

    def _params(self, query: Query) -> dict:
        terms = [t for t in (query.name, query.address, query.context) if t]
        if not terms:
            # A photo-only subject gives a text search nothing to search on.
            raise AdapterNotApplicable("needs a name or context")
        return {
            "engine": self.engine,
            "q": " ".join(terms),
            "num": self.config.num_results,
        }


class _ReverseImageAdapter(_SerpApiAdapter):
    """Shared behaviour for the two engines that search by photo.

    These engines fetch the image from their own servers, so the URL has to be
    reachable from the public internet. A local path, or a localhost URL, is
    invisible to them however valid it looks from here.
    """

    def _params(self, query: Query) -> dict:
        if not query.photo_url:
            # No photo at all is a different thing from a photo nobody can
            # fetch. The first was never asked; the second was asked and could
            # not be answered.
            raise AdapterNotApplicable("needs a photo")

        reason = unreachable_reason(query.photo_url)
        if reason is not None:
            # Raised so the run reports *failed* with this reason. Reporting
            # empty would say the engine looked and found nothing, when in
            # fact it never saw the photograph.
            raise ValueError(reason)
        return {"engine": self.engine, "url": query.photo_url}

    def _result_blocks(self, payload: Mapping) -> list[tuple[Mapping, tuple[str, ...]]]:
        blocks = list(super()._result_blocks(payload))
        # Visual matches are the substance of a reverse image search: each is a
        # page showing an image, which is exactly what face comparison needs.
        for key in ("visual_matches", "image_results"):
            for result in payload.get(key) or []:
                if isinstance(result, Mapping):
                    blocks.append((result, ("original", "thumbnail")))
        return blocks


class GoogleLensAdapter(_ReverseImageAdapter):
    """Google Lens reverse image search."""

    engine = "google_lens"


class YandexImagesAdapter(_ReverseImageAdapter):
    """Yandex reverse image search. Different recall from Lens, same shape."""

    engine = "yandex_images"
