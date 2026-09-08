"""Run the full pipeline on one subject and print a report.

    python scripts/run_subject.py --name "Marcus Webb" --context "Austin, TX"

Every search goes through the cache, so the first run costs one API call per
adapter and every run after that costs nothing. The live-call count is printed
at the end, because search spend that is invisible is search spend that gets
forgotten.

Nothing is written to the database. This prints and exits.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

# Allow "python scripts/run_subject.py" from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Results come from the whole web, so names arrive in every script there is.
# A Windows console defaults to a codepage that cannot encode most of them,
# and the report must not die partway through because one result was Japanese.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.adapters.base import AdapterRun, AssertionDraft, Query  # noqa: E402
from app.adapters.serpapi import (  # noqa: E402
    GoogleLensAdapter,
    GoogleSearchAdapter,
    YandexImagesAdapter,
)
from app.cache import CACHE_DIR, cache_key, cached_get  # noqa: E402
from app.engine.candidates import CandidateDraft, form_candidates  # noqa: E402
from app.engine.conflicts import ConflictDraft, detect_conflicts  # noqa: E402
from app.engine.faces import FaceMatcher  # noqa: E402
from app.engine.query_planner import (  # noqa: E402
    QueryPlanConfig,
    deduplicate_drafts,
    plan_queries,
    query_text,
)
from app.engine.scoring import (  # noqa: E402
    derive_attribution,
    score_identity,
    score_item,
)
from app.models import SourceRunStatus  # noqa: E402

ADAPTERS = {
    "google": GoogleSearchAdapter,
    "google_lens": GoogleLensAdapter,
    "yandex_images": YandexImagesAdapter,
}

WIDTH = 100


class CountingFetch:
    """cached_get, plus a tally of what actually went over the wire.

    A cache hit and a live call are indistinguishable from the outside, so the
    cache file is checked before delegating. That is the same check
    ``cached_get`` makes, one moment earlier.
    """

    def __init__(self) -> None:
        self.live = 0
        self.cached = 0

    def __call__(self, params: dict) -> dict:
        if (CACHE_DIR / f"{cache_key(params)}.json").exists():
            self.cached += 1
        else:
            self.live += 1
        return cached_get(params)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def heading(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(f" {text}")
    print("=" * WIDTH)


def clip(value: object, width: int) -> str:
    """Truncate for the table. ASCII only: consoles here are not always UTF-8."""
    text = "-" if value is None else str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def origin_summary(drafts: Iterable[AssertionDraft]) -> dict[str, int]:
    """The counts behind the honest corroboration line.

    Mirrors ``Subject.origin_summary`` but reads drafts, where the publisher
    sits on the claim rather than on a Provenance row. Unnamed publishers are
    counted separately rather than dropped: dropping them would understate the
    source count, and understating errs in the flattering direction.
    """
    drafts = list(drafts)
    return {
        "named_publishers": len({d.publisher for d in drafts if d.publisher}),
        "unattributed": sum(1 for d in drafts if not d.publisher),
        "distinct_origins": len({d.origin_key for d in drafts if d.origin_key}),
    }


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------


def print_subject(query: Query, names: Sequence[str]) -> None:
    heading("SUBJECT")
    print(f"  name      : {query.name or '-'}")
    print(f"  context   : {query.context or '-'}")
    print(f"  photo_url : {query.photo_url or '-'}")
    print(f"  adapters  : {', '.join(names)}")


def print_source_runs(runs: Sequence[tuple[str, AdapterRun]]) -> None:
    """Per source and per query, because each query is a separate search."""
    heading("1. PER-SOURCE OUTCOME")
    print(f"  {'source':<22} {'query':<44} outcome")
    print("  " + "-" * 90)
    for label, run in runs:
        source_run = run.source_run
        if source_run.status is SourceRunStatus.found:
            outcome = f"found {source_run.result_count}"
        elif source_run.status is SourceRunStatus.empty:
            outcome = "empty (returned nothing)"
        else:
            outcome = f"failed  {source_run.error_reason}"
        print(f"  {source_run.source:<22} {clip(label, 44):<44} {outcome}")

    statuses = [r.source_run.status for _, r in runs]
    print()
    print(
        f"  {len(runs)} search(es): "
        f"{statuses.count(SourceRunStatus.found)} found, "
        f"{statuses.count(SourceRunStatus.empty)} empty, "
        f"{statuses.count(SourceRunStatus.failed)} failed. "
        "Empty and failed are different outcomes."
    )


def print_candidate(
    index: int, candidate: CandidateDraft, query: Query, total: int
) -> None:
    p = score_identity(candidate, query)

    localities = ", ".join(sorted(k or "(none)" for k in candidate.locality_keys))
    label = "ANCHOR - the person searched for" if candidate.is_anchor else "other"
    print()
    print(f"  Candidate {index} of {total}   p = {float(p):.4f}   [{label}]")
    print(f"    name key   : {candidate.name_key or '(none)'}")
    print(f"    localities : {localities}")
    print(
        f"    held by    : {candidate.grouping_basis.value} "
        f"(x{p.components['grouping_basis_factor']:.2f} on p)"
    )

    if candidate.is_merged or candidate.merge_evidence:
        print(f"    merged     : {len(candidate.blocking_keys)} blocks")
        for evidence in candidate.merge_evidence:
            print(f"                 - {evidence}")
    if p.penalties:
        for name, value in p.penalties.items():
            print(f"    penalty    : {name} (-{value})")

    signals = ", ".join(
        f"{k}={'-' if v is None else f'{v:.2f}'}" for k, v in p.components.items()
    )
    print(f"    signals    : {signals}")

    if candidate.anchor_matches:
        print("    anchored by:")
        for ref, match in sorted(
            candidate.anchor_matches.items(), key=lambda kv: -kv[1].strength
        ):
            link = "link" if match.has_link else "no link"
            face = (
                f"face {match.face_signal:.2f}"
                + ("!" if match.face_ambiguous else "")
                if match.face_signal is not None
                else "-"
            )
            print(f"                 {match.strength:.3f}  {match.basis:<34} "
                  f"{link:<8} {face:<11} {clip(ref, 34)}")

    print()
    header = (
        f"    {'predicate':<12} {'value':<30} {'q':>6} {'r':>6}  "
        f"{'origin':<18} {'evidence':<19} {'freshness':<10}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))

    scored = []
    for assertion in candidate.assertions:
        q = score_item(
            assertion, candidate, consumed_record_refs=p.consumed_record_refs
        )
        scored.append((assertion, q, derive_attribution(p, q)))
    scored.sort(key=lambda row: row[2] or 0.0, reverse=True)

    for assertion, q, r in scored:
        freshness = (
            "unknown" if q.freshness_unknown else f"{q.components['freshness']:.2f}"
        )
        print(
            f"    {assertion.predicate:<12} "
            f"{clip(assertion.normalized_value, 30):<30} "
            f"{float(q):>6.4f} {r:>6.4f}  "
            f"{clip(assertion.origin_key, 18):<18} "
            f"{assertion.evidence_kind.value:<19} "
            f"{freshness:<10}"
        )

    summary = origin_summary(candidate.assertions)
    print()
    print(
        f"    corroboration: {summary['named_publishers']} named publishers, "
        f"{summary['unattributed']} unattributed, "
        f"{summary['distinct_origins']} distinct origins"
    )
    print(
        "    (publishers are not confirmations; corroboration counts origins)"
    )


def print_candidates(
    candidates: Sequence[CandidateDraft], query: Query
) -> list[CandidateDraft]:
    heading("2. CANDIDATES (ranked by p)")
    if not candidates:
        print("  0 candidates.")
        print("  Nothing met threshold. This is a completed run, not a failure.")
        return []

    # The anchor leads regardless of p. It is the answer to the question that
    # was asked; the rest are other people who happen to share a name.
    ranked = sorted(
        candidates,
        key=lambda c: (c.is_anchor, float(score_identity(c, query))),
        reverse=True,
    )
    for index, candidate in enumerate(ranked, start=1):
        print_candidate(index, candidate, query, len(ranked))
    return ranked


def print_conflicts(candidates: Sequence[CandidateDraft], query: Query) -> None:
    heading("3. CONFLICTS")
    found: list[tuple[int, ConflictDraft]] = []
    for index, candidate in enumerate(candidates, start=1):
        for conflict in detect_conflicts(candidate):
            found.append((index, conflict))

    if not found:
        print("  None. Disagreement over a multivalued predicate is not a conflict.")
        return

    for index, conflict in found:
        kind = "potential conflict" if conflict.potential else "conflict"
        print()
        print(f"  Candidate {index}: {kind} on {conflict.predicate}")
        print(f"    reason  : {conflict.reason.value}")
        print(f"    status  : {conflict.status.value}")
        print(f"    values  : {', '.join(str(v) for v in conflict.values)}")
        print(f"    basis   : {conflict.basis}")
        print(f"    preferred_assertion: {conflict.preferred_assertion}")
    print()
    print("  Nothing was overwritten. Every conflicting assertion is still above.")


def print_rejected_by_anchor(formation) -> None:
    """Which records did not match the subject, and by how much they missed."""
    if not formation.anchor_available:
        return
    anchored = set()
    for candidate in formation.candidates:
        anchored.update(candidate.anchor_matches)

    rejected = {
        ref: match
        for ref, match in formation.anchor_matches.items()
        if ref not in anchored
    }
    heading("5. RECORDS REJECTED BY THE ANCHOR")
    print(f"  {len(rejected)} record(s) scored below the anchor threshold.")
    if not rejected:
        return
    print("  Kept and clustered separately. A rejected same-name result is a")
    print("  finding in its own right, not noise to hide.")
    print()
    for ref, match in sorted(rejected.items(), key=lambda kv: -kv[1].strength):
        why = match.rejected_because or "below threshold"
        print(
            f"    {match.strength:.3f}  {match.basis:<28} {why:<30} "
            f"{clip(ref, 40)}"
        )


def print_unattached(unattached: Sequence[AssertionDraft]) -> None:
    heading("4. UNATTACHED ASSERTIONS")
    print(f"  {len(unattached)} assertion(s) matched no candidate above threshold.")
    if not unattached:
        return
    print("  Retained, not discarded, and never forced onto the nearest candidate.")
    print()
    print(
        f"    {'predicate':<12} {'value':<38} {'origin':<20} {'publisher':<20}"
    )
    print("    " + "-" * 92)
    for assertion in unattached:
        print(
            f"    {assertion.predicate:<12} "
            f"{clip(assertion.normalized_value, 38):<38} "
            f"{clip(assertion.origin_key, 20):<20} "
            f"{clip(assertion.publisher, 20):<20}"
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_subject",
        description="Run the pipeline on one subject and print a report.",
    )
    parser.add_argument("--name", help="Subject name to search for")
    parser.add_argument("--photo-url", help="Public image URL for reverse search")
    parser.add_argument("--context", help="Anything narrowing the search, e.g. a city")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=QueryPlanConfig().max_queries,
        help="Cap on planned queries; each one is a paid search",
    )
    parser.add_argument(
        "--adapters",
        default=",".join(ADAPTERS),
        help=f"Comma-separated: {', '.join(ADAPTERS)} (default: all three)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not (args.name or args.photo_url or args.context):
        print("Nothing to search on. Give --name, --photo-url or --context.")
        return 2

    names = [n.strip() for n in args.adapters.split(",") if n.strip()]
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        print(f"Unknown adapter(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(ADAPTERS)}")
        return 2

    query = Query(name=args.name, photo_url=args.photo_url, context=args.context)
    fetch = CountingFetch()

    print_subject(query, names)

    # One query is not a search strategy. Text engines get the expanded plan;
    # image engines search by photo, which has nothing to expand.
    plan_config = QueryPlanConfig(max_queries=args.max_queries)
    planned = plan_queries(query, plan_config) or [query]
    print()
    print(f"  query plan ({len(planned)} of max {plan_config.max_queries}):")
    for position, planned_query in enumerate(planned, start=1):
        print(f"    {position}. {query_text(planned_query)}")

    runs: list[tuple[str, AdapterRun]] = []
    for name in names:
        adapter = ADAPTERS[name](fetch=fetch)
        if name == "google":
            for planned_query in planned:
                runs.append((query_text(planned_query), adapter.run(planned_query)))
        else:
            runs.append((query_text(query), adapter.run(query)))

    print_source_runs(runs)

    drafts: list[AssertionDraft] = []
    context: dict[str, str] = {}
    for _, run in runs:
        drafts.extend(run.drafts)
        context.update(run.record_context)

    # The same page found by several queries is already one record, because
    # record_ref is the same. This only stops it being listed several times.
    before_dedup = len(drafts)
    drafts = deduplicate_drafts(drafts)

    # Face verification, when a photo was supplied. Built after collection so
    # the images to compare against are the ones the search actually found.
    face_matcher = None
    if query.photo_url:
        face_matcher = FaceMatcher(subject_photo_url=query.photo_url)
        heading("FACE VERIFICATION")
        if face_matcher.available:
            print(f"  subject photo   : {query.photo_url}")
            print(f"  faces detected  : {len(face_matcher.subject_faces)}")
            if face_matcher.subject_is_ambiguous:
                print("  WARNING: more than one face in the subject photo.")
        else:
            print(f"  unavailable: {face_matcher.subject_error}")
            print("  Records will be judged on name and context alone.")

    formation = form_candidates(
        drafts,
        subject=query,
        record_context=context,
        face_match=face_matcher.match_record if face_matcher else None,
        face_distance=face_matcher.candidate_distance if face_matcher else None,
    )
    if formation.anchor_available:
        print()
        print(
            f"  anchoring: on ({len(formation.anchor_matches)} records scored "
            f"against the subject)"
        )
    else:
        print()
        print("  anchoring: off (no name or context supplied)")
    ranked = print_candidates(formation.candidates, query)
    print_conflicts(ranked, query)
    print_unattached(formation.unattached)
    print_rejected_by_anchor(formation)

    heading("SEARCH SPEND")
    print(f"  searches issued         : {len(runs)}")
    print(f"  live API calls this run : {fetch.live}")
    print(f"  served from cache       : {fetch.cached}")
    print(f"  drafts collected        : {len(drafts)}")
    print(f"  duplicate drafts merged : {before_dedup - len(drafts)}")
    if fetch.live == 0:
        print("  This run cost nothing. Every response came from cache/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
