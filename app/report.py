"""Assembling a run into a report, and running one in the background.

The report is the product. Everything upstream exists to fill it in, and the
one rule it enforces is that a reader must never be able to mistake an
estimate for a measurement. So freshness is the word "unknown" rather than a
stand-in number, rejected candidates travel with the score they missed by, and
an empty source and a failed source are different fields, not one status.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.adapters.base import AssertionDraft, Query
from app.adapters.fixture import FIXTURES_DIR, FixtureAdapter
from app.adapters.serpapi import (
    GoogleLensAdapter,
    GoogleSearchAdapter,
    YandexImagesAdapter,
)
from app.cache import CACHE_DIR, cache_key, cached_get
from app.uploads import path_for_digest
from app.engine.candidates import (
    CandidateDraft,
    CandidateFormation,
    FormationConfig,
    form_candidates,
)
from app.engine.conflicts import detect_conflicts
from app.engine.faces import FaceConfig, FaceMatcher
from app.engine.query_planner import (
    QueryPlanConfig,
    deduplicate_drafts,
    plan_queries,
    query_text,
)
from app.engine.scoring import derive_attribution, score_identity, score_item

ADAPTERS = {
    "google": GoogleSearchAdapter,
    "google_lens": GoogleLensAdapter,
    "yandex_images": YandexImagesAdapter,
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

#: What each fixture is for, so a reviewer can pick a case rather than a file.
FIXTURE_NOTES: dict[str, dict[str, str]] = {
    "01_same_name_two_cities": {
        "title": "Same name, two cities",
        "expects": "Two candidates. A shared name never merges them.",
        "name": "Michael Petrie",
        "context": "logistics operations, Austin",
    },
    "04_conflicting_birth_dates": {
        "title": "Incompatible birth dates",
        "expects": "Both claims kept. One conflict, unresolved.",
        "name": "Michael Petrie",
        "context": "Austin, records",
    },
    "05_source_failure": {
        "title": "A source errors",
        "expects": "Reported failed with a reason, never empty.",
        "name": "Michael Petrie",
        "context": "Austin",
    },
    "06_clean_match": {
        "title": "Clean match",
        "expects": "One candidate, nothing unattached, no conflicts.",
        "name": "Marcus Webb",
        "context": "Austin, logistics operations",
    },
    "07_no_suitable_candidate": {
        "title": "Nobody matches",
        "expects": "No anchor. The run completes and says why.",
        "name": "Michael Petrie",
        "context": "insurance fraud, Austin",
    },
}


def available_fixtures() -> list[dict[str, str]]:
    """Fixture cases a reviewer can trigger, in file order."""
    cases = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        note = FIXTURE_NOTES.get(path.stem, {})
        cases.append(
            {
                "id": path.stem,
                "title": note.get("title", path.stem.replace("_", " ")),
                "expects": note.get("expects", ""),
                # The picker fills the form with these, so a reviewer can see
                # exactly what subject the case is being searched against.
                "name": note.get("name", ""),
                "context": note.get("context", ""),
            }
        )
    return cases


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def _freshness(item) -> str | float:
    """The word, not a number, when a source never said when it looked.

    Returning 0.5 here would be a lie a reader could not see through.
    """
    if item.freshness_unknown:
        return "unknown"
    return round(float(item.components["freshness"]), 4)


def _others_summary(candidates: list[dict]) -> dict[str, Any]:
    """What to call the collapsed panel of candidates that are not the anchor.

    It read "116 other people sharing this name" while only 37 of them shared
    it; the other 79 scored 0.00 against the name and were rejected on
    something else. The same defect as the per-candidate labels, one level up
    and from the same cause: a run-level flag says a name was supplied, and the
    heading took that as licence to describe every candidate beneath it.

    Counted from the stamps the server has already decided, so the heading and
    the cards cannot disagree — deriving them separately is what let them drift
    the first time.
    """
    others = [c for c in candidates if not c["is_anchor"]]
    counts: dict[str, int] = {}
    for candidate in others:
        counts[candidate["stamp"]] = counts.get(candidate["stamp"], 0) + 1

    total = len(others)
    same_name = counts.get("Other person, same name", 0)
    no_face = counts.get("Other record, no face match", 0)
    unmatched = counts.get("Other record, not matched", 0)

    def plural(n: int, one: str, many: str) -> str:
        return f"{n} {one if n == 1 else many}"

    # One population: name it, rather than counting it out into parts of one.
    if same_name == total and total:
        label = plural(total, "other person sharing this name",
                       "other people sharing this name")
    elif no_face == total and total:
        label = plural(total, "record whose face did not match",
                       "records whose faces did not match")
    elif unmatched == total and total:
        label = plural(total, "other record that did not match the subject",
                       "other records that did not match the subject")
    else:
        parts = []
        if same_name:
            parts.append(f"{same_name} sharing this name")
        if no_face:
            parts.append(f"{no_face} whose face did not match")
        if unmatched:
            parts.append(f"{unmatched} that matched the subject on nothing")
        label = plural(total, "other candidate", "other candidates")
        if parts:
            label += " — " + ", ".join(parts)

    return {
        "total": total,
        "same_name": same_name,
        "no_face_match": no_face,
        "unmatched": unmatched,
        "label": label,
    }


def _candidate_stamp(candidate: CandidateDraft, components: dict) -> str:
    """What to call this candidate, from what was compared for this candidate.

    The run-level ``compared`` block says a name was supplied. It does not say
    the name was compared *to this candidate*, and on a mixed search most of
    them are not: a reverse image search returns pages about whoever the
    picture resembles, and 79 of 116 non-anchor candidates in one live run had
    a name similarity of exactly 0.00 while being labelled "Other person, same
    name". "Prof Mark Walterfang", "Meet the PPG Team" and "My Bio" among them.

    The candidate's own name score is the right discriminator and is already
    computed. Zero means the name was compared and did not match, or that there
    was no name to compare — either way the label may not claim one.
    """
    if candidate.is_anchor:
        return "The person searched for"
    if components.get("name"):
        return "Other person, same name"
    # A face was compared and did not match: that is what decided this one.
    if components.get("face") is not None:
        return "Other record, no face match"
    return "Other record, not matched"


def _first_name_for(candidate: CandidateDraft, ref: str) -> str | None:
    """The name one record of this candidate carried, for display."""
    for assertion in candidate.assertions:
        if assertion.record_ref == ref and assertion.predicate == "name":
            value = assertion.raw_value or assertion.normalized_value
            if value:
                return value
    return None


def _plain_basis(
    candidate: CandidateDraft,
    subject: Query | None = None,
    name_signal: float | None = None,
) -> str:
    """What holds this candidate together, in words a reviewer would use.

    For an anchor this is a union across its records, and it now says so. It
    used to read "face match 0.99, name, context", which is the grammar of a
    conjunction: one record that matched on all three. No record did. The face
    came from a single page with no context agreement at all, and every record
    that agreed on context had no face to compare. A reader was being shown the
    best of each column and left to assume they described one thing.

    The wording separates the two cases. When one record does carry every
    signal, that is worth saying plainly. When the evidence is spread, the
    count says how thinly.
    """
    if candidate.is_anchor:
        matches = list(candidate.anchor_matches.values())
        if not matches:
            return "matched the subject"

        best_face = max((m.face_signal for m in matches if m.face_signal), default=None)
        present: list[tuple[str, str]] = []
        if best_face is not None:
            present.append(("face", f"a face match at {best_face:.2f}"))
        if any(m.name_signal for m in matches):
            present.append(("name", "the name"))
        if any(m.context_signal for m in matches):
            present.append(("context", "context"))
        # Announced explicitly: without it a candidate admitted on its locality
        # reported "name" as its basis, crediting the one signal this system
        # refuses to treat as evidence of identity.
        if any(m.locality_signal for m in matches):
            present.append(("locality", "locality"))

        if not present:
            return "matched the subject"

        def carries(match, kind: str) -> bool:
            return bool(getattr(match, f"{kind}_signal", None))

        kinds = [kind for kind, _ in present]
        together = sum(
            1 for m in matches if all(carries(m, kind) for kind in kinds)
        )
        phrases = [phrase for _, phrase in present]
        if len(phrases) == 1:
            joined = phrases[0]
        else:
            joined = ", ".join(phrases[:-1]) + " and " + phrases[-1]

        if len(matches) == 1:
            return f"one record, on {joined}"
        if together:
            return (
                f"{joined} — all of it on {together} of {len(matches)} records"
            )
        return f"{joined}, spread across {len(matches)} records — no one record has all of it"

    basis = candidate.grouping_basis.value

    # "name_only" describes how records were clustered — by their own name keys,
    # against each other — and not what the subject was matched on. Rendered as
    # "a shared name only" it claims a comparison against the subject that may
    # never have happened.
    #
    # The test is this candidate's own name score, not whether the run had a
    # name. A photo-only search compares no name at all; a mixed search
    # compares one and most candidates still score zero, because a reverse
    # image search returns pages about whoever the picture resembles. Both
    # cases were claiming a shared name, and the rejection list beneath
    # correctly reported them as decided on something else.
    unmatched_name = not name_signal
    if unmatched_name and basis in {"name_only", "name_and_locality"}:
        records = len({a.record_ref for a in candidate.assertions if a.record_ref})
        if records <= 1:
            return "a single record, matched against nothing"
        return f"{records} records that share a name with each other"

    spoken = {
        "distinctive_attribute": "a shared email, phone or identifier",
        "face": "a face match",
        "name_and_locality": "name and locality",
        "name_only": "a shared name only",
        "unspecified": "not recorded",
    }
    return spoken.get(basis, basis)


#: Separators a site joins a person's name to the rest of a page title with.
#: Splitting on these is a display convention, not parsing: the whole title is
#: still carried on the row underneath.
_TITLE_SEPARATORS = (" - ", " – ", " — ", " | ", " · ", " :: ")


#: Words a page appends to a person's name to describe what the page sells.
#: Trailing runs of these are dropped, so "Michael Petrie Email & Phone Number"
#: becomes "Michael Petrie". They are only ever removed from the end, and never
#: far enough to leave less than a first and last name.
_NOT_NAME_WORDS: frozenset[str] = frozenset(
    {
        "email", "emails", "phone", "phones", "number", "numbers", "address",
        "addresses", "contact", "contacts", "details", "detail", "info",
        "information", "profile", "profiles", "resume", "cv", "bio",
        "biography", "overview", "records", "record", "background", "report",
        "reports", "lookup", "search", "results", "page", "post", "posts",
        "and", "or", "the", "amp",
    }
)

_MIN_NAME_WORDS = 2


def _plausible_person_name(text: str) -> str:
    """Trim a page title down to the part that could be somebody's name.

    Cuts at a title separator first, then peels trailing words that describe a
    page rather than a person. Never cuts below two words: "Michael Page" is a
    person as well as a recruiter, and losing a surname is worse than keeping a
    stray noun.
    """
    for separator in _TITLE_SEPARATORS:
        if separator in text:
            text = text.split(separator, 1)[0]

    words = text.split()
    while len(words) > _MIN_NAME_WORDS:
        tail = words[-1].strip(".,;:&-").casefold()
        if tail and tail not in _NOT_NAME_WORDS:
            break
        words.pop()

    return " ".join(words).strip(" .,;:&-")


def _display_name(rows: list[dict[str, Any]]) -> str | None:
    """The name to head a candidate with: a person, not a page title.

    Ordered by what the evidence says, strongest first: the claim's own score,
    then the face similarity behind the record it came from, then the shortest
    name once page-title furniture is removed.

    The face fallback matters because r can be zero for every row — a
    photo-only subject scored zero across the board before this — and a
    tie-break on a column that is uniformly zero silently degrades to whatever
    order the rows happened to arrive in.
    """
    names = [r for r in rows if r["predicate"] == "name" and r.get("raw_value")]
    if not names:
        return None

    def rank(row: dict[str, Any]) -> tuple[float, float, int]:
        cleaned = _plausible_person_name(row["raw_value"]) or row["raw_value"]
        return (-row.get("r", 0.0), -(row.get("face") or 0.0), len(cleaned))

    best = min(names, key=rank)
    return _plausible_person_name(best["raw_value"]) or best["raw_value"]


def _origin_summary(drafts: list[AssertionDraft]) -> dict[str, int]:
    return {
        "named_publishers": len({d.publisher for d in drafts if d.publisher}),
        "unattributed": sum(1 for d in drafts if not d.publisher),
        "distinct_origins": len({d.origin_key for d in drafts if d.origin_key}),
    }


def _serialise_candidate(
    candidate: CandidateDraft, subject: Query, index: int
) -> dict[str, Any]:
    p = score_identity(candidate, subject)
    face_by_record = {
        ref: match.face_signal
        for ref, match in (candidate.anchor_matches or {}).items()
        if match.face_signal is not None
    }
    rows = []
    for assertion in candidate.assertions:
        q = score_item(
            assertion, candidate, consumed_record_refs=p.consumed_record_refs
        )
        rows.append(
            {
                "predicate": assertion.predicate,
                "value": assertion.normalized_value or assertion.raw_value,
                "raw_value": assertion.raw_value,
                "q": round(float(q), 4),
                "r": round(derive_attribution(p, q) or 0.0, 4),
                "origin": assertion.origin_key,
                "publisher": assertion.publisher,
                "evidence_kind": assertion.evidence_kind.value,
                "lineage": assertion.source_origin.value,
                "freshness": _freshness(q),
                "attachment": q.attachment_basis,
                "record_ref": assertion.record_ref,
                "face": face_by_record.get(assertion.record_ref),
            }
        )
    rows.sort(key=lambda row: -row["r"])

    # The blocking key is lowercased and stripped for matching. Showing it to
    # an investigator would look like a data error, so the heading is what a
    # source actually printed, chosen by score rather than by length.
    display_name = _display_name(rows)

    return {
        "index": index,
        "is_anchor": candidate.is_anchor,
        "name": candidate.name_key,
        "display_name": display_name,
        "localities": sorted(k for k in candidate.locality_keys if k),
        "p": round(float(p), 4),
        "grouping_basis": candidate.grouping_basis.value,
        "held_by": _plain_basis(candidate, subject, p.components.get("name")),
        "stamp": _candidate_stamp(candidate, p.components),
        "signals": {
            key: (None if value is None else round(float(value), 4))
            for key, value in p.components.items()
        },
        "penalties": {k: round(v, 4) for k, v in p.penalties.items()},
        "merge_evidence": list(candidate.merge_evidence),
        "anchored_by": [
            {
                "record_ref": ref,
                "source": record_source(ref, _first_name_for(candidate, ref)),
                "strength": round(match.strength, 4),
                "basis": match.basis,
                "face": None if match.face_signal is None else round(match.face_signal, 4),
                "face_ambiguous": match.face_ambiguous,
                "has_link": match.has_link,
            }
            for ref, match in sorted(
                candidate.anchor_matches.items(), key=lambda kv: -kv[1].strength
            )
        ],
        "corroboration": _origin_summary(candidate.assertions),
        "assertions": rows,
    }


def record_source(ref: str, title: str | None = None) -> dict[str, Any]:
    """What to show for a record: where it is, or failing that, what it was.

    A record reference is an internal identity, and for a result the source
    returned without a URL it is a hash of the result's own bytes —
    ``serpapi:google:sha256:0f085f7e010adeb4:10``. That is a sound identity and
    an appalling thing to put in front of a reader, who cannot tell it from a
    bug. The reference is still carried for tracing; this is what gets shown.

    ``url`` is None exactly when the source supplied no link, which the caller
    should say rather than leave as a blank.
    """
    parts = ref.split(":", 2)
    rest = parts[2] if len(parts) == 3 else ref
    url = None if rest.startswith("sha256:") else rest
    return {
        "url": url,
        "title": (title or "").strip() or None,
        # Said plainly, because "no link" is a fact about the source rather
        # than something that went wrong here.
        "note": None if url else "no link supplied by the source",
    }


def _titles_by_ref(formation: CandidateFormation) -> dict[str, str]:
    """The first name each record carried, for records with nothing else to show."""
    titles: dict[str, str] = {}
    for candidate in formation.candidates:
        for assertion in candidate.assertions:
            if assertion.predicate != "name" or not assertion.record_ref:
                continue
            value = assertion.raw_value or assertion.normalized_value
            if value:
                titles.setdefault(assertion.record_ref, value)
    return titles


def _rejection(
    ref: str, match, config: FormationConfig, title: str | None = None
) -> dict[str, Any]:
    """Why one record is not the subject, phrased so the score cannot mislead.

    Two different decisions land here and they must not read alike. Most
    records simply scored below the threshold. A handful score *high* and are
    still refused, because a name was the only signal they carried, and the
    strength of a lone signal is just that signal: an exact match on a common
    name reads 1.00 and settles nothing. For those the number decided nothing,
    and the wording has to say so, or a reader sees 1.00 beside a rejection and
    concludes the report contradicts itself.
    """
    parts = []
    if match.name_signal is not None:
        parts.append(f"name {match.name_signal:.2f}")
    if match.context_signal is not None:
        parts.append(f"context {match.context_signal:.2f}")
    if match.locality_signal is not None:
        parts.append(f"locality {match.locality_signal:.2f}")
    if match.face_signal is not None:
        parts.append(f"face {match.face_signal:.2f}")
    components = ", ".join(parts) or "nothing comparable to match on"

    #: What each refusal means, in the reviewer's terms. Keyed on the reason the
    #: engine recorded, so a new rule cannot quietly inherit an old sentence.
    substance = {
        "name only — no other signal": (
            f"{components} — and nothing else. No context term, no locality and "
            "no face to check it against, so the name was matching a label "
            "rather than a person."
        ),
        "locality contradicts the subject": (
            f"{components} — and the locality disagrees. Whatever else matched, "
            "this record places the person somewhere the subject is not."
        ),
        "does not carry the subject's name": (
            f"{components} — but not the subject's name. What agreed was the "
            "surrounding detail, which many people would share."
        ),
    }

    if match.rejected_because:
        decided_by = "substance"
        explanation = substance.get(
            match.rejected_because,
            f"{components} — refused: {match.rejected_because}.",
        )
    else:
        decided_by = "threshold"
        explanation = (
            f"{components} — below the {config.anchor_threshold:.2f} threshold."
        )

    return {
        "record_ref": ref,
        "source": record_source(ref, title),
        "strength": round(match.strength, 4),
        "threshold": config.anchor_threshold,
        "decided_by": decided_by,
        "components": components,
        "explanation": explanation,
        "basis": match.basis,
        "reason": match.rejected_because or "scored below the anchor threshold",
        "face": None if match.face_signal is None else round(match.face_signal, 4),
        "has_link": match.has_link,
        "locality": match.locality_signal,
    }


def _serialise_conflicts(candidates: list[CandidateDraft]) -> list[dict[str, Any]]:
    out = []
    for position, candidate in enumerate(candidates, start=1):
        for conflict in detect_conflicts(candidate):
            out.append(
                {
                    "candidate_index": position,
                    "predicate": conflict.predicate,
                    "reason": conflict.reason.value,
                    "status": conflict.status.value,
                    "potential": conflict.potential,
                    "basis": conflict.basis,
                    "values": [str(v) for v in conflict.values],
                    "preferred": None,
                }
            )
    return out


#: Most images shown for one candidate. A verified strip is evidence, not a
#: gallery, and past a handful it stops being read.
MAX_VERIFIED_IMAGES = 6


def verified_images(
    candidate: CandidateDraft,
    matcher: FaceMatcher | None,
    config: FaceConfig | None = None,
) -> list[dict[str, Any]]:
    """Images that actually matched the subject's face, strongest first.

    Only images that were compared *and* cleared the threshold appear. An
    image that could not be fetched, held no detectable face, or scored below
    the line is left out entirely rather than shown with a caveat. Putting a
    stranger's face under a person's name is the false attribution this whole
    system exists to prevent, and a caption does not undo a photograph.
    """
    if matcher is None or not matcher.available:
        return []

    config = config or matcher.config
    by_record: dict[str | None, list[AssertionDraft]] = {}
    for assertion in candidate.assertions:
        by_record.setdefault(assertion.record_ref, []).append(assertion)

    found: list[dict[str, Any]] = []
    for record_ref, drafts in by_record.items():
        page = next(
            (
                d.raw_value
                for d in drafts
                if d.predicate == "profile_url" and d.raw_value
            ),
            None,
        )
        publisher = next((d.publisher for d in drafts if d.publisher), None)
        origin = next((d.origin_key for d in drafts if d.origin_key), None)
        for url, similarity in matcher.image_matches(drafts):
            if similarity < config.same_person_similarity:
                continue
            found.append(
                {
                    "image_url": url,
                    "source_page": page or url,
                    "publisher": publisher or origin,
                    "similarity": round(float(similarity), 4),
                    "record_ref": record_ref,
                }
            )

    found.sort(key=lambda item: -item["similarity"])
    return found[:MAX_VERIFIED_IMAGES]


def build_report(
    subject: Query,
    runs: list[tuple[str, Any]],
    formation: CandidateFormation,
    spend: dict[str, int],
    matcher: FaceMatcher | None = None,
) -> dict[str, Any]:
    """Everything the results view renders, already ordered findings-first."""
    ranked = sorted(
        formation.candidates,
        key=lambda c: (c.is_anchor, float(score_identity(c, subject))),
        reverse=True,
    )
    candidates = [
        _serialise_candidate(c, subject, i) for i, c in enumerate(ranked, start=1)
    ]
    # Only the anchor carries a verified strip. The other candidates are people
    # who share a name; showing faces beside them would suggest they had been
    # checked against the subject and passed.
    for shape, draft in zip(candidates, ranked):
        shape["verified_images"] = (
            verified_images(draft, matcher) if draft.is_anchor else []
        )

    anchored_refs: set[str] = set()
    for candidate in formation.candidates:
        anchored_refs.update(candidate.anchor_matches)
    config = FormationConfig()
    titles = _titles_by_ref(formation)
    rejected = [
        _rejection(ref, match, config, titles.get(ref))
        for ref, match in formation.anchor_matches.items()
        if ref not in anchored_refs
    ]
    # Two populations, and they are not comparable on the number. A record
    # refused by the threshold is ranked by how close it came; a record refused
    # on substance scored high and the score decided nothing, so 1.00 there
    # means less than 0.40 does above. Sorting them into one list and calling
    # it "closest first" was a contradiction a reader could see: entries at
    # 0.00 sat above entries at 1.00.
    #
    # They stay grouped, ordered by score within each group, and the view
    # labels the groups rather than implying one ranking runs through both.
    rejected.sort(key=lambda r: (r["decided_by"] == "substance", -r["strength"]))

    return {
        "subject": {
            "name": subject.name,
            "address": subject.address,
            "context": subject.context,
            "photo_url": subject.photo_url,
        },
        "sources": [
            {
                "source": run.source_run.source,
                "query": label,
                "status": run.source_run.status.value,
                "result_count": run.source_run.result_count,
                "error_reason": run.source_run.error_reason,
            }
            for label, run in runs
        ],
        "anchor_available": formation.anchor_available,
        # The collapsed panel's heading, counted from the stamps rather than
        # derived a second time. See _others_summary.
        "other_candidates": _others_summary(candidates),
        # Which comparisons the caller made possible. A view that labels a
        # candidate has to know this: on a photo-only search there is no name
        # to compare, so calling the others "same name" describes something
        # that never happened.
        "compared": {
            "name": bool(subject.name),
            "context": bool(subject.context),
            "locality": bool(subject.address),
            "face": bool(subject.photo_url),
        },
        "candidates": candidates,
        "rejected": rejected,
        "conflicts": _serialise_conflicts(ranked),
        "unattached": [
            {
                "predicate": d.predicate,
                "value": d.normalized_value or d.raw_value,
                "origin": d.origin_key,
                "publisher": d.publisher,
            }
            for d in formation.unattached
        ],
        "spend": spend,
    }


# --------------------------------------------------------------------------
# Running one subject, in the background, reporting progress
# --------------------------------------------------------------------------


class _CountingFetch:
    """cached_get, plus a tally of what actually went over the wire."""

    def __init__(self) -> None:
        self.live = 0
        self.cached = 0

    def __call__(self, params: dict) -> dict:
        if (CACHE_DIR / f"{cache_key(params)}.json").exists():
            self.cached += 1
        else:
            self.live += 1
        return cached_get(params)


@dataclass
class Run:
    """One search, from submitted to finished, observable while it happens."""

    id: str
    subject: dict[str, Any]
    status: str = "running"
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    planned: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    face: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "subject": self.subject,
            "started_at": self.started_at,
            "planned": self.planned,
            "sources": list(self.sources),
            "face": self.face,
            "report": self.report,
            "error": self.error,
        }


_RUNS: dict[str, Run] = {}
_LOCK = threading.Lock()


def get_run(run_id: str) -> Run | None:
    with _LOCK:
        return _RUNS.get(run_id)


def start_run(
    name: str | None,
    address: str | None,
    context: str | None,
    photo_url: str | None,
    adapters: list[str],
    fixture: str | None,
    max_queries: int,
    photo_digest: str | None = None,
) -> Run:
    """Register a run and execute it on a worker thread."""
    subject = Query(
        name=name or None,
        address=address or None,
        context=context or None,
        photo_url=photo_url or None,
    )
    run = Run(
        id=uuid.uuid4().hex[:12],
        subject={
            "name": subject.name,
            "address": subject.address,
            "context": subject.context,
            "photo_url": subject.photo_url,
            "fixture": fixture,
        },
    )
    with _LOCK:
        _RUNS[run.id] = run

    local_photo = path_for_digest(photo_digest) if photo_digest else None
    thread = threading.Thread(
        target=_execute,
        args=(run, subject, adapters, fixture, max_queries, local_photo),
        daemon=True,
    )
    thread.start()
    return run


def _record(run: Run, label: str, source: str, status: str, **extra) -> None:
    with _LOCK:
        run.sources.append(
            {"query": label, "source": source, "status": status, **extra}
        )


def _execute(
    run: Run,
    subject: Query,
    adapters: list[str],
    fixture: str | None,
    max_queries: int,
    local_photo: Path | None = None,
) -> None:
    try:
        fetch = _CountingFetch()
        collected: list[tuple[str, Any]] = []
        drafts: list[AssertionDraft] = []
        context_map: dict[str, str] = {}

        if fixture:
            # A fixture stands in for the network entirely, so a reviewer can
            # trigger an edge case without spending a search.
            adapter = FixtureAdapter(fixture)
            with _LOCK:
                run.planned = [f"fixture {fixture}"]
            result = adapter.run_fixture()
            collected.append((f"fixture {fixture}", result))
            _record(
                run,
                f"fixture {fixture}",
                adapter.name,
                result.source_run.status.value,
                result_count=result.source_run.result_count,
                error_reason=result.source_run.error_reason,
            )
            drafts.extend(result.drafts)
            context_map.update(result.record_context)
        else:
            planned = plan_queries(subject, QueryPlanConfig(max_queries=max_queries))
            planned = planned or [subject]
            with _LOCK:
                run.planned = [query_text(q) for q in planned]

            for adapter_name in adapters:
                adapter = ADAPTERS[adapter_name](fetch=fetch)
                queries = planned if adapter_name == "google" else [subject]
                for planned_query in queries:
                    label = query_text(planned_query) or "photo"
                    result = adapter.run(planned_query)
                    collected.append((label, result))
                    _record(
                        run,
                        label,
                        result.source_run.source,
                        result.source_run.status.value,
                        result_count=result.source_run.result_count,
                        error_reason=result.source_run.error_reason,
                    )
                    drafts.extend(result.drafts)
                    context_map.update(result.record_context)

        drafts = deduplicate_drafts(drafts)

        # Read the photo off disk when we stored it ourselves. Fetching our
        # own upload back over a public URL adds a way to fail that has nothing
        # to do with faces: a tunnel that has gone down takes face matching
        # with it, and the report then says "photo could not be fetched" about
        # a file sitting right here.
        face_source = str(local_photo) if local_photo else subject.photo_url
        matcher = None
        if face_source:
            matcher = FaceMatcher(subject_photo_url=face_source)
            with _LOCK:
                run.face = {
                    "available": matcher.available,
                    "read_from": "local file" if local_photo else "url",
                    "faces_detected": len(matcher.subject_faces),
                    "ambiguous": matcher.subject_is_ambiguous,
                    "error": matcher.subject_error,
                }

        formation = form_candidates(
            drafts,
            subject=subject,
            record_context=context_map,
            face_match=matcher.match_record if matcher else None,
            face_distance=matcher.candidate_distance if matcher else None,
        )

        report = build_report(
            subject,
            collected,
            formation,
            matcher=matcher,
            spend={
                "searches": len(collected),
                "live_calls": fetch.live,
                "from_cache": fetch.cached,
                "drafts": len(drafts),
            },
        )
        with _LOCK:
            run.report = report
            run.status = "complete"
    except Exception as exc:  # noqa: BLE001 - a failed run is a result to show
        with _LOCK:
            run.error = f"{type(exc).__name__}: {exc}"
            run.status = "failed"
