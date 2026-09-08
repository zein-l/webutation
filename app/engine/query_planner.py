"""Query expansion. One query is not a search strategy.

A single "name plus everything" query is the narrowest search possible: it asks
the engine to match every term at once, so any result missing one of them is
lost. Recall is the product here, and recall comes from asking several narrower
questions rather than one wide one.

The variants, in the order they are worth spending on:

1. **Name alone.** The broadest reach, and the only query that finds a person
   described in words the caller did not supply.
2. **Name as an exact phrase.** The highest precision, and the one that keeps
   "Michael Petrie" from matching a page about Michael and a page about Petrie.
3. **Name plus the organisation** from the context, if one is recognisable.
4. **Name plus each remaining context term**, one term at a time.
5. **The organisation alone**, which finds the person through the organisation
   rather than the other way round.

Every variant costs a search, so the count is capped in config. Deduplication
of what comes back is not this module's job: two queries returning the same
page produce the same ``record_ref``, and candidate formation groups on it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.adapters.base import AssertionDraft, Query

_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)

#: Words that carry no search value on their own.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "was", "were",
        "are", "his", "her", "its", "their", "has", "have", "had", "who",
        "not", "but", "all", "any", "our", "into", "over", "under",
    }
)

#: Segments that look like a proper noun but name a field rather than a body.
_GENERIC_SEGMENTS: frozenset[str] = frozenset(
    {
        "private investigator", "insurance fraud", "osint", "open source",
        "background check", "due diligence", "risk", "security", "compliance",
        "investigations", "consultant", "consulting",
    }
)


@dataclass(frozen=True)
class QueryPlanConfig:
    """How wide to cast, and how much that is allowed to cost."""

    #: Hard cap on planned queries. Each one is a paid search, so recall is
    #: bought deliberately rather than by accident.
    max_queries: int = 6

    include_name_alone: bool = True
    include_exact_phrase: bool = True
    include_context_terms: bool = True
    include_organisation_alone: bool = True

    #: Context terms shorter than this are ignored.
    min_term_length: int = 3
    stopwords: frozenset[str] = _STOPWORDS
    #: Comma-separated context segments this long or shorter may be read as an
    #: organisation name.
    max_organisation_tokens: int = 3
    generic_segments: frozenset[str] = _GENERIC_SEGMENTS


def query_text(query: Query) -> str:
    """The search string an adapter will build from this query.

    Mirrors how the SerpAPI adapters join their parameters, so a plan can be
    displayed and deduplicated using exactly what will be sent.
    """
    return " ".join(
        part for part in (query.name, query.address, query.context) if part
    )


def context_segments(context: str | None) -> list[str]:
    """Comma-separated pieces of the supplied context, in order."""
    if not context:
        return []
    return [segment.strip() for segment in context.split(",") if segment.strip()]


def context_terms(context: str | None, config: QueryPlanConfig) -> list[str]:
    """Distinct, meaningful terms from the context, order preserved."""
    if not context:
        return []
    seen: list[str] = []
    for segment in context_segments(context):
        cleaned = " ".join(_PUNCTUATION.sub(" ", segment).split())
        if not cleaned:
            continue
        tokens = [
            t
            for t in cleaned.split()
            if len(t) >= config.min_term_length
            and t.casefold() not in config.stopwords
        ]
        if not tokens:
            continue
        term = " ".join(tokens)
        if term.casefold() not in {s.casefold() for s in seen}:
            seen.append(term)
    return seen


def organisation_name(context: str | None, config: QueryPlanConfig) -> str | None:
    """The organisation in the context, if one is recognisable.

    A heuristic, and labelled as one: the first comma-separated segment that
    reads as a proper noun and is not a field of work. "Webutation, private
    investigator, insurance fraud" yields "Webutation". Wrong guesses cost one
    search, which is why this is capped and configurable rather than clever.
    """
    generic = {g.casefold() for g in config.generic_segments}
    for segment in context_segments(context):
        if segment.casefold() in generic:
            continue
        tokens = segment.split()
        if not tokens or len(tokens) > config.max_organisation_tokens:
            continue
        # Proper nouns are capitalised; a field of work usually is not.
        if all(token[0].isupper() for token in tokens if token[:1].isalpha()):
            return segment
    return None


def plan_queries(
    subject: Query, config: QueryPlanConfig | None = None
) -> list[Query]:
    """Expand one subject into a bounded list of searches.

    Returns queries in descending order of expected value, truncated to
    ``max_queries``, with duplicates removed by the text each would send.
    A subject with no name and no context yields nothing to expand, so the
    subject is returned unchanged when it has any content at all.
    """
    config = config or QueryPlanConfig()
    planned: list[Query] = []
    seen: set[str] = set()

    def add(query: Query) -> None:
        text = query_text(query).strip()
        if not text or text.casefold() in seen:
            return
        seen.add(text.casefold())
        planned.append(query)

    name = (subject.name or "").strip()
    organisation = organisation_name(subject.context, config)
    terms = context_terms(subject.context, config) if config.include_context_terms else []

    if name:
        if config.include_name_alone:
            add(Query(name=name, photo_url=subject.photo_url))
        if config.include_exact_phrase:
            add(Query(name=f'"{name}"', photo_url=subject.photo_url))
        if organisation:
            add(Query(name=f'"{name}"', context=organisation))
        for term in terms:
            if organisation and term.casefold() == organisation.casefold():
                continue
            add(Query(name=name, context=term))
    else:
        # No name: the context is all there is to search on.
        for term in terms:
            add(Query(context=term))

    if config.include_organisation_alone and organisation:
        add(Query(context=organisation))

    if not planned and query_text(subject).strip():
        # Nothing expandable, but the subject itself is still searchable.
        planned.append(subject)

    return planned[: config.max_queries]


def deduplicate_drafts(drafts: Iterable[AssertionDraft]) -> list[AssertionDraft]:
    """Drop drafts that repeat a claim already seen from the same record.

    Formation already collapses repeat records by ``record_ref``, so this
    changes no grouping and no score. It exists so that a page found by three
    queries is listed once rather than three times, which is a reporting
    concern, not a correctness one.
    """
    seen: set[tuple] = set()
    unique: list[AssertionDraft] = []
    for draft in drafts:
        key = (
            draft.record_ref,
            draft.predicate,
            draft.normalized_value,
            draft.raw_value,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(draft)
    return unique


def describe_plan(queries: Sequence[Query]) -> list[str]:
    """One display string per planned query, in plan order."""
    return [query_text(query) for query in queries]
