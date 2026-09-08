"""An adapter that reads synthetic source records from ``fixtures/*.json``.

The fixtures are raw source responses, not prepared reports, and they enter
through the same ``collect`` then ``normalize`` path a live adapter uses. Only
``collect`` differs: it opens a file where a network adapter makes a request.
Everything downstream sees an identical RawResponse, so a fixture exercises
normalization, candidate formation, attachment and scoring for real. A canned
report would exercise none of it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.adapters.base import (
    AdapterRun,
    AssertionDraft,
    Query,
    RawResponse,
    SourceRunDraft,
)
from app.models import (
    AccessCategory,
    EvidenceKind,
    SourceOrigin,
    SourceRunStatus,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"

# Source field name to predicate. Sources name things their own way; this is
# the only place that vocabulary is translated.
_PREDICATE_BY_FIELD = {
    # Name, as three different sources happen to label it.
    "full_name": "name",
    "profile_name": "name",
    "display_name": "name",
    "date_of_birth": "birth_date",
    "current_employer": "employer",
    "job_title": "job_title",
    # Locality, likewise.
    "city": "city",
    "location": "city",
    "locality": "city",
}


def _normalize_value(predicate: str, raw: str) -> str | None:
    """Canonical form of a raw value, or None when it cannot be parsed.

    Kept deliberately small. Real normalization per predicate is a larger
    problem; what matters here is that comparison happens on a canonical form
    rather than on whatever text a publisher happened to render.
    """
    text = " ".join(raw.split())
    if not text:
        return None

    if predicate == "birth_date":
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue
        # An unparseable date is reported as unnormalized rather than guessed,
        # so it can never be compared against a real date and read as a match.
        return None

    if predicate == "name":
        return text.replace(".", "").lower()

    return text.lower()


def _evidence_kind(
    value: object, default: EvidenceKind = EvidenceKind.unknown
) -> EvidenceKind:
    """Read a declared evidence kind, falling back rather than guessing.

    An unrecognised or absent value becomes the supplied default, and the
    default at record level is ``unknown``. Nothing is inferred from lineage:
    that would rebuild the very coupling this field exists to break.
    """
    if value is None:
        return default
    try:
        return EvidenceKind(value)
    except ValueError:
        return default


def _parse_observed_at(value: str | None) -> datetime | None:
    """Read a record date as an aware UTC datetime, or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class FixtureAdapter:
    """Loads one fixture file and normalizes it like any other source."""

    def __init__(self, fixture: str, default_origin: str | None = None) -> None:
        """``fixture`` is a stem such as "06_clean_match", or a path."""
        candidate = Path(fixture)
        if candidate.suffix == ".json" and candidate.exists():
            self.path = candidate.resolve()
        else:
            self.path = (FIXTURES_DIR / f"{Path(fixture).stem}.json").resolve()

        self.name = f"fixture:{self.path.stem}"
        # Each fixture record declares its own lineage, so there is no
        # adapter-wide origin to fall back on. Left None so a record without
        # lineage stays unknown instead of being credited to this adapter.
        self.default_origin = default_origin

    def collect(self, query: Query) -> RawResponse:
        """Read the fixture and return it as an unmodified payload.

        ``fetched_at`` is now, not the fixture's own dates. Collection time and
        observation time are different facts, and a fixture read today is
        genuinely collected today.
        """
        body = self.path.read_bytes()
        payload = json.loads(body.decode("utf-8"))

        # A synthetic reference, distinguishable at a glance from a real cache
        # key while still pinning the exact bytes that produced the drafts.
        digest = hashlib.sha256(body).hexdigest()
        artifact_ref = f"fixture:{self.path.stem}:{digest[:16]}"

        return RawResponse(
            payload=payload,
            fetched_at=datetime.now(timezone.utc),
            artifact_ref=artifact_ref,
            source=self.name,
        )

    def _record_ref(self, result: dict) -> str | None:
        """Stable identity of one result within this fixture.

        Prefers whatever the source itself uses to identify a record: an
        explicit id, else the result URL, else its position. Prefixed with the
        fixture stem so references stay unique across fixtures.
        """
        identity = (
            result.get("result_id")
            or result.get("link")
            or (f"position:{result['position']}" if "position" in result else None)
        )
        if identity is None:
            return None
        return f"{self.path.stem}:{identity}"

    def normalize(self, raw: RawResponse) -> list[AssertionDraft]:
        """Turn search results into one draft per recognized field."""
        drafts: list[AssertionDraft] = []

        for result in raw.payload.get("results", []):
            if not isinstance(result, dict):
                continue

            publisher = result.get("publisher")
            observed_at = _parse_observed_at(result.get("record_date"))
            record_ref = self._record_ref(result)

            lineage = result.get("lineage") or {}
            origin_key = lineage.get("origin") or self.default_origin

            # A lineage state is only meaningful alongside a named origin.
            # Without one, corroboration has nothing to count, so the claim is
            # recorded as unknown rather than as an uncountable known origin.
            if origin_key is None:
                source_origin = SourceOrigin.unknown
            else:
                try:
                    source_origin = SourceOrigin(lineage.get("state", "unknown"))
                except ValueError:
                    source_origin = SourceOrigin.unknown

            try:
                access_category = AccessCategory(result.get("access", "PUBLIC_WEB"))
            except ValueError:
                access_category = AccessCategory.PUBLIC_WEB

            # Reliability is read from its own block, never from lineage. A
            # record states one kind for all its claims and may override it per
            # field, mirroring how origin works: record default, claim override.
            evidence = result.get("evidence") or {}
            record_kind = _evidence_kind(evidence.get("kind"))
            by_field = evidence.get("by_field") or {}

            for field_name, raw_value in (result.get("fields") or {}).items():
                predicate = _PREDICATE_BY_FIELD.get(field_name)
                if predicate is None or raw_value is None:
                    continue

                drafts.append(
                    AssertionDraft(
                        predicate=predicate,
                        raw_value=str(raw_value),
                        normalized_value=_normalize_value(predicate, str(raw_value)),
                        record_ref=record_ref,
                        publisher=publisher,
                        origin_key=origin_key,
                        source_origin=source_origin,
                        evidence_kind=_evidence_kind(
                            by_field.get(field_name), default=record_kind
                        ),
                        observed_at=observed_at,
                        access_category=access_category,
                    )
                )

        return drafts

    def record_context(self, raw: RawResponse) -> dict[str, str]:
        """Snippets, kept against their record and never turned into claims."""
        context: dict[str, str] = {}
        for result in raw.payload.get("results", []):
            if not isinstance(result, dict):
                continue
            snippet = result.get("snippet")
            ref = self._record_ref(result)
            if ref and isinstance(snippet, str) and snippet.strip():
                context[ref] = snippet.strip()
        return context

    def run_fixture(self) -> AdapterRun:
        """Collect and normalize, reporting the outcome instead of raising.

        A fixture can declare a failure, because "this source errored" is a
        case a reviewer needs to see rendered, and it must not be reachable
        only by unplugging the network.
        """
        ran_at = datetime.now(timezone.utc)
        try:
            raw = self.collect(Query())
        except Exception as exc:  # noqa: BLE001 - a failed source is a result
            return AdapterRun(
                source_run=SourceRunDraft(
                    source=self.name,
                    status=SourceRunStatus.failed,
                    error_reason=f"{type(exc).__name__}: {exc}",
                    ran_at=ran_at,
                )
            )

        declared = raw.payload.get("source_run") or {}
        if declared.get("status") == "failed":
            return AdapterRun(
                source_run=SourceRunDraft(
                    source=self.name,
                    status=SourceRunStatus.failed,
                    error_reason=declared.get("error_reason", "the source errored"),
                    ran_at=ran_at,
                ),
                raw=raw,
            )

        results = raw.payload.get("results") or []
        if not results:
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
                result_count=len(results),
                ran_at=ran_at,
            ),
            drafts=self.normalize(raw),
            raw=raw,
            record_context=self.record_context(raw),
        )
