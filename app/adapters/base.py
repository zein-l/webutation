"""The contract every source adapter implements.

An adapter has exactly two jobs, kept separate on purpose:

* ``collect`` talks to the outside world and returns the response untouched.
* ``normalize`` turns that response into drafts and touches no network.

Splitting them is what lets a fixture and a live API share one code path. A
fixture adapter reads a file where a network adapter makes a request, and
everything downstream of ``normalize`` cannot tell the difference. It is also
what makes normalization testable without a network, and re-normalization
possible against a stored artifact after the parsing rules change.

These are drafts, not rows. A draft carries what a source claimed; deciding
which candidate it attaches to, and with what confidence, happens later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.models import AccessCategory, EvidenceKind, SourceOrigin, SourceRunStatus


@dataclass(frozen=True)
class Query:
    """A search request. Every field is optional, but an adapter needs one.

    Mirrors Subject: a photo, or a name with whatever context narrows it.
    """

    photo_url: str | None = None
    name: str | None = None
    address: str | None = None
    context: str | None = None

    def is_empty(self) -> bool:
        """True when there is nothing to search on."""
        return not any((self.photo_url, self.name, self.address, self.context))


@dataclass(frozen=True)
class RawResponse:
    """One source response, exactly as received.

    ``payload`` is not modified on the way in. ``artifact_ref`` names the
    stored copy, so a draft can be traced back to the bytes it came from.
    """

    payload: dict
    fetched_at: datetime
    artifact_ref: str
    source: str


@dataclass(frozen=True)
class AssertionDraft:
    """One claim a source made, before it is attached or scored.

    ``source_origin`` is the lineage state and ``origin_key`` names the origin
    itself. They travel together because corroboration counts distinct origins,
    not publishers, so a claim with a known lineage state but no named origin
    would be uncountable.

    ``evidence_kind`` is a separate axis and must not be inferred from lineage.
    Lineage answers where a claim came from and feeds corroboration; evidence
    kind answers how the source came to know it and feeds reliability. A
    first-party origin can relay a claim it never observed.

    ``observed_at`` is when the source recorded the fact. It is deliberately
    not paired with a fetch time here: that belongs to the RawResponse, and
    collapsing the two is how a mirror fetched today comes to look like fresh
    evidence for a five-year-old claim.

    ``record_ref`` says which source record this claim came from, and is what
    lets candidate formation know that a name and a locality describe the same
    person. Adapters must populate it. Left null, the draft is treated as its
    own single-claim record, which fragments but never invents a pairing.
    """

    predicate: str
    raw_value: str | None = None
    normalized_value: str | None = None
    #: Stable identity of the originating record: one search result, one
    #: profile, one row. Unique within a response; adapters should prefix it so
    #: it stays unique across responses.
    record_ref: str | None = None
    publisher: str | None = None
    origin_key: str | None = None
    source_origin: SourceOrigin = SourceOrigin.unknown
    #: How the source knows this, set per claim by the adapter. Defaults to
    #: unknown so an adapter that has not made the judgement does not get
    #: credited with having made it.
    evidence_kind: EvidenceKind = EvidenceKind.unknown
    observed_at: datetime | None = None
    #: The period the fact applies to, usually unknown. Distinct from
    #: observed_at: when a source recorded something says nothing about the
    #: span it was true for. Conflict detection needs both bounds to tell a
    #: historical change from a genuine disagreement.
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    access_category: AccessCategory = AccessCategory.PUBLIC_WEB


class AdapterNotApplicable(Exception):
    """This source cannot answer this question, and nothing went wrong.

    A text search with no text, or an image search with no image, has not
    failed: it was never applicable. Reporting it as a failure invites someone
    to go looking for a fault that does not exist, and reporting it as empty
    claims a search happened. The message says what the source would need.
    """


@dataclass(frozen=True)
class SourceRunDraft:
    """What one source did for one subject.

    ``empty`` and ``failed`` are separate outcomes and must never collapse
    into one. A source that returned nothing answered the question; a source
    that errored did not, and the difference is what the report has to show.
    """

    source: str
    status: SourceRunStatus
    result_count: int | None = None
    error_reason: str | None = None
    ran_at: datetime | None = None


@dataclass
class AdapterRun:
    """One adapter call, including the outcome when it did not succeed.

    ``run`` never raises, because a failed source is a result to report rather
    than an exception to propagate. ``raw`` is None exactly when the call
    failed.
    """

    source_run: SourceRunDraft
    drafts: list[AssertionDraft] = field(default_factory=list)
    raw: RawResponse | None = None
    #: Free text a result carried, kept against the record rather than turned
    #: into claims. Snippets belong here; parsing them would invent facts.
    record_context: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Adapter(Protocol):
    """What a source must provide to take part in a run.

    ``default_origin`` is the adapter-level fallback used when a record does
    not name its own origin. An adapter that returns content from many origins,
    such as a general web search, should leave it ``None`` so unnamed lineage
    stays unknown rather than being silently credited to the adapter.
    """

    name: str
    default_origin: str | None

    def collect(self, query: Query) -> RawResponse:
        """Fetch from the source and return the response unmodified."""
        ...

    def normalize(self, raw: RawResponse) -> list[AssertionDraft]:
        """Parse a response into drafts. Pure: no network, no database."""
        ...
