"""Conflict detection. Nothing here resolves, prefers, or overwrites.

The hard part is not finding disagreements, it is refusing to call every
disagreement a conflict. Four cases the plan separates:

* Different addresses over different periods are a **historical change**. A
  person moved. Both claims were true.
* Two employers are **compatible**. Employment is multivalued, and a person can
  hold two jobs at once.
* Incompatible values for an exclusive predicate over overlapping validity are
  a **conflict**.
* One source silent where another speaks is **missing evidence**. Silence is
  not disagreement, and nothing here treats it as such.

Exclusivity is declared per predicate for the handful where it matters, and
everything else is multivalued by default. This is a lookup table, not a
general exclusivity system: getting one wrong should cost an entry in a dict.

A conflict may also mean the grouping was wrong rather than the fact. When
conflicting claims arrive from records that were brought together on weak
evidence, the reason names the grouping decision to inspect rather than
preserving it and marking all of its facts uncertain.

Two groupings can be at fault and they are reported separately.
``possible_bad_merge`` means two blocks were merged because they resembled each
other. ``possible_bad_anchor`` means records were each matched against the
subject and never compared with one another, so one of them is probably a
different person who also matched. Sending a reviewer to inspect a merge that
never happened wastes the one thing the reason exists to buy.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

from app.adapters.base import AssertionDraft
from app.engine.candidates import (
    DISTINCTIVE_PREDICATES,
    LOCALITY_PREDICATES,
    CandidateDraft,
    normalize_locality_key,
    normalize_name_key,
)
from app.models import ConflictReason, ConflictStatus


class Exclusivity(str, enum.Enum):
    """How many values of a predicate a person may hold."""

    #: One value, full stop, and time does not enter into it. A birth date is
    #: not something a person has a series of.
    exclusive = "exclusive"
    #: One value at a time. Two values over separate periods are a history;
    #: two values over the same period are a contradiction.
    temporally_exclusive = "temporally_exclusive"
    #: Any number of simultaneous values. The default, and the reason employer
    #: never produces a conflict.
    multivalued = "multivalued"


@dataclass(frozen=True)
class ConflictConfig:
    """Which predicates are exclusive, and what counts as a strong merge."""

    #: Anything absent from this map is multivalued. Deliberately short.
    exclusivity: Mapping[str, Exclusivity] = field(
        default_factory=lambda: {
            # One value ever.
            "birth_date": Exclusivity.exclusive,
            "ssn": Exclusivity.exclusive,
            "national_id": Exclusivity.exclusive,
            "government_id": Exclusivity.exclusive,
            "drivers_license": Exclusivity.exclusive,
            "passport_number": Exclusivity.exclusive,
            # One value at a time. A person moves; both addresses were true.
            "address": Exclusivity.temporally_exclusive,
            "city": Exclusivity.temporally_exclusive,
        }
    )
    default_exclusivity: Exclusivity = Exclusivity.multivalued

    #: Sharing one of these across two records is strong enough that a
    #: disagreement is about the facts rather than about the merge.
    strong_join_predicates: frozenset[str] = DISTINCTIVE_PREDICATES

    #: Merge justifications strong enough to survive a conflict. Face distance
    #: is deliberately absent: it is good enough to propose a merge, but when
    #: an exclusive predicate then contradicts itself, the merge is exactly
    #: what should be questioned. Add "face distance" here to change that.
    strong_merge_markers: tuple[str, ...] = ("shared distinctive attribute",)


@dataclass
class ConflictDraft:
    """A detected conflict, before it is stored.

    Mirrors the plan's conflict JSON. ``preferred_assertion`` is always None:
    choosing a preferred value is a separate, auditable decision, and nothing
    in detection makes it. Every conflicting assertion stays in ``assertions``
    and none is modified.
    """

    predicate: str
    assertions: list[AssertionDraft]
    reason: ConflictReason
    status: ConflictStatus = ConflictStatus.unresolved
    preferred_assertion: AssertionDraft | None = None

    #: True when validity periods are unknown, so overlap could not be shown.
    #: The disagreement is flagged for review, not asserted as a contradiction.
    potential: bool = False
    #: Why this was emitted, in words, for the audit trail.
    basis: str = ""

    @property
    def values(self) -> list[str | None]:
        """The distinct normalized values in dispute."""
        seen: list[str | None] = []
        for assertion in self.assertions:
            if assertion.normalized_value not in seen:
                seen.append(assertion.normalized_value)
        return seen

    @property
    def record_refs(self) -> set[str | None]:
        return {a.record_ref for a in self.assertions}

    def __repr__(self) -> str:
        kind = "potential " if self.potential else ""
        return (
            f"<{kind}ConflictDraft {self.predicate!r} "
            f"reason={self.reason.value} values={self.values}>"
        )


# --------------------------------------------------------------------------
# Records and blocks
# --------------------------------------------------------------------------

_RecordKey = tuple[str, str | int]


def _record_key(assertion: AssertionDraft, position: int) -> _RecordKey:
    """Same convention as candidate formation: null refs stand alone."""
    if assertion.record_ref:
        return ("ref", assertion.record_ref)
    return ("solo", position)


def _record_label(key: _RecordKey) -> str:
    """String form of a record key, matching how formation labels records."""
    kind, value = key
    return str(value) if kind == "ref" else f"solo:{value}"


def _group_by_record(
    assertions: Iterable[AssertionDraft],
) -> dict[_RecordKey, list[AssertionDraft]]:
    records: dict[_RecordKey, list[AssertionDraft]] = {}
    for position, assertion in enumerate(assertions):
        records.setdefault(_record_key(assertion, position), []).append(assertion)
    return records


def _blocking_key(drafts: Sequence[AssertionDraft]) -> tuple[str | None, str | None]:
    """Recomputed from the record, matching how formation blocked it."""
    name = None
    for draft in drafts:
        if draft.predicate == "name":
            name = normalize_name_key(draft.normalized_value or draft.raw_value)
            if name:
                break

    by_predicate = {d.predicate: d for d in drafts}
    locality = None
    for predicate in LOCALITY_PREDICATES:
        draft = by_predicate.get(predicate)
        if draft is not None:
            locality = normalize_locality_key(draft.normalized_value or draft.raw_value)
            if locality:
                break

    return (name, locality)


def _distinctive_signals(
    drafts: Iterable[AssertionDraft], config: ConflictConfig
) -> set[tuple[str, str]]:
    signals: set[tuple[str, str]] = set()
    for draft in drafts:
        value = draft.normalized_value or draft.raw_value
        if value and draft.predicate in config.strong_join_predicates:
            signals.add((draft.predicate, value.casefold()))
    return signals


# --------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------


def _has_validity(assertion: AssertionDraft) -> bool:
    return assertion.valid_from is not None or assertion.valid_to is not None


def _periods_overlap(left: AssertionDraft, right: AssertionDraft) -> bool | None:
    """True, False, or None when the periods are not stated.

    Unknown is a distinct answer, not a default to either side. An open bound
    means "still true as far as this source knows", so two open-ended claims do
    overlap.
    """
    if not _has_validity(left) or not _has_validity(right):
        return None

    def starts_before_other_ends(a: AssertionDraft, b: AssertionDraft) -> bool:
        if a.valid_from is None or b.valid_to is None:
            return True
        return a.valid_from <= b.valid_to

    return starts_before_other_ends(left, right) and starts_before_other_ends(
        right, left
    )


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _anchor_grouped(
    candidate: CandidateDraft, left_label: str, right_label: str
) -> bool:
    """Did anchoring bring these two records together, rather than a merge?

    Anchored records were never compared with each other: each matched the
    subject on its own. Two of them disagreeing about an exclusive predicate
    therefore impeaches the anchor, not a merge.

    The fallback covers records that reached the anchor candidate by
    attachment rather than by anchoring: if the candidate recorded no merge at
    all, no merge can be what grouped them.
    """
    if not candidate.is_anchor:
        return False
    if left_label in candidate.anchor_matches and right_label in candidate.anchor_matches:
        return True
    return not candidate.merge_evidence


def _grouping_fault(
    left: Sequence[AssertionDraft],
    right: Sequence[AssertionDraft],
    candidate: CandidateDraft,
    left_label: str,
    right_label: str,
    config: ConflictConfig,
) -> ConflictReason | None:
    """Which grouping decision this conflict calls into question, if any.

    None means the grouping stands and the facts are what disagree.

    Asked only of records in different blocking keys. Records inside one block
    agreed on name and locality and were never actively brought together, so
    there is no grouping decision to impeach. That check matters more than it
    looks: locality predicates are part of the blocking key, so two records
    disagreeing about a city necessarily fall in different blocks, and the
    disagreement would otherwise be the very thing used to convict a grouping
    that never happened.
    """
    # Strong positive evidence beats any doubt about how they were grouped.
    if _distinctive_signals(left, config) & _distinctive_signals(right, config):
        return None

    if _anchor_grouped(candidate, left_label, right_label):
        return ConflictReason.possible_bad_anchor

    if not (candidate.is_merged or candidate.merge_evidence):
        return None

    if any(
        marker in evidence
        for evidence in candidate.merge_evidence
        for marker in config.strong_merge_markers
    ):
        return None

    return ConflictReason.possible_bad_merge


def detect_conflicts(
    candidate: CandidateDraft, config: ConflictConfig | None = None
) -> list[ConflictDraft]:
    """Find genuine conflicts among a candidate's assertions.

    Compares normalized values, so two spellings of one date are one value and
    never a conflict. A predicate with a single distinct value produces nothing,
    which is what makes silence read as missing evidence rather than dissent.

    Returns an empty list when nothing conflicts, and never raises. Nothing is
    modified, nothing is preferred, and every assertion remains exactly as it
    was.
    """
    config = config or ConflictConfig()
    conflicts: list[ConflictDraft] = []

    records = _group_by_record(candidate.assertions)
    record_of: dict[int, _RecordKey] = {}
    for key, drafts in records.items():
        for draft in drafts:
            record_of[id(draft)] = key
    blocks = {key: _blocking_key(drafts) for key, drafts in records.items()}

    by_predicate: dict[str, list[AssertionDraft]] = {}
    for assertion in candidate.assertions:
        if assertion.normalized_value is None:
            # Nothing comparable was extracted, so nothing can be contradicted.
            continue
        by_predicate.setdefault(assertion.predicate, []).append(assertion)

    for predicate, assertions in by_predicate.items():
        mode = config.exclusivity.get(predicate, config.default_exclusivity)
        if mode is Exclusivity.multivalued:
            # Employment and its kind. Two values are two facts.
            continue

        distinct = {a.normalized_value for a in assertions}
        if len(distinct) < 2:
            # One value, however many sources said it. Agreement, or silence.
            continue

        for left, right in combinations(assertions, 2):
            if left.normalized_value == right.normalized_value:
                continue

            if mode is Exclusivity.exclusive:
                overlap: bool | None = True
                basis = "exclusive predicate with incompatible values"
            else:
                overlap = _periods_overlap(left, right)
                if overlap is False:
                    # A person moved. Both claims were true, at different times.
                    continue
                basis = (
                    "overlapping validity periods"
                    if overlap
                    else "validity periods unknown, overlap not established"
                )

            left_record, right_record = record_of[id(left)], record_of[id(right)]
            grouped_across_blocks = (
                left_record != right_record
                and blocks[left_record] != blocks[right_record]
            )
            fault = (
                _grouping_fault(
                    records[left_record],
                    records[right_record],
                    candidate,
                    _record_label(left_record),
                    _record_label(right_record),
                    config,
                )
                if grouped_across_blocks
                else None
            )

            if fault is ConflictReason.possible_bad_anchor:
                reason = fault
                basis = (
                    f"{basis}; both records matched the subject independently "
                    "and were never compared with each other, so the anchor "
                    "may be wrong rather than the facts"
                )
            elif fault is ConflictReason.possible_bad_merge:
                reason = fault
                basis = (
                    f"{basis}; the two records were merged without a shared "
                    "distinctive attribute, so the merge may be wrong rather "
                    "than the facts"
                )
            else:
                reason = ConflictReason.incompatible_normalized_values

            conflicts.append(
                ConflictDraft(
                    predicate=predicate,
                    assertions=[left, right],
                    reason=reason,
                    potential=overlap is None,
                    basis=basis,
                )
            )

    return conflicts
