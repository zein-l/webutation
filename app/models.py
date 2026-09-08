"""SQLAlchemy 2.0 declarative models for the Webutation data model.

Schema only. The scoring inputs described in the plan (embedding distance,
recency decay, corroboration counting) are not implemented here; the columns
exist to record their outputs.

Two structural commitments from the plan are worth stating up front:

* An assertion always belongs to a subject but only optionally to a
  candidate. An unattached assertion matched no candidate above threshold and
  is retained rather than discarded.
* Conflicts are stored, never resolved away. Nothing in this schema
  overwrites a competing value, and preferred_assertion_id records a derived
  choice without destroying the alternatives.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# --------------------------------------------------------------------------
# Enums. Values are the plan's strings verbatim, including the upper-case
# access categories, so stored data reads the same as the specification.
# --------------------------------------------------------------------------


class ReviewStatus(str, enum.Enum):
    """Human-set review state. Never derived from a confidence score."""

    unreviewed = "unreviewed"
    needs_review = "needs_review"
    reviewed = "reviewed"
    rejected = "rejected"


class SourceOrigin(str, enum.Enum):
    """Lineage of the assertion, not of the adapter that fetched it."""

    known_origin = "known_origin"
    suspected_copy = "suspected_copy"
    unknown = "unknown"


class EvidenceKind(str, enum.Enum):
    """How a source came to know a claim: the reliability axis.

    Distinct from :class:`SourceOrigin`, which answers where the claim came
    from and drives corroboration counting. The two are independent: a
    first-party origin can relay something it never observed, and a republished
    row can faithfully reproduce a direct observation. Keeping them in one
    field would make a lineage judgement double as a reliability judgement.
    """

    #: The source recorded the fact itself. A county filing office.
    direct_observation = "direct_observation"
    #: The subject asserted it about themselves. A profile field.
    self_reported = "self_reported"
    #: The source relays someone else's observation.
    secondhand = "secondhand"
    #: Reproduced from another publisher's record. An aggregator row.
    republished = "republished"
    unknown = "unknown"


class GroupingBasis(str, enum.Enum):
    """What holds a candidate together.

    A candidate is a claim that several records describe one person. That claim
    rests on something, and what it rests on varies enormously in strength: a
    shared government identifier is near-proof, a shared name is barely
    evidence at all. Recording the basis lets scoring price the difference
    instead of treating every grouping as equally earned.
    """

    #: Every record independently matched the subject that was searched for.
    anchor = "anchor"
    #: Records share an email, phone, or government identifier.
    distinctive_attribute = "distinctive_attribute"
    #: Records were joined by face distance below threshold.
    face = "face"
    #: Records agree on both name and locality.
    name_and_locality = "name_and_locality"
    #: Records agree on a name and nothing else. The weakest basis there is,
    #: and the one that silently fuses strangers if left unpriced.
    name_only = "name_only"
    #: Not stated. Scoring applies no adjustment rather than guessing.
    unspecified = "unspecified"


class AccessCategory(str, enum.Enum):
    """Descriptive access metadata. Records a characteristic, enforces nothing."""

    PUBLIC_WEB = "PUBLIC_WEB"
    LICENSED = "LICENSED"
    RESTRICTED = "RESTRICTED"


class ConflictReason(str, enum.Enum):
    """Why two assertions are held to be in conflict.

    The last two are the load-bearing members. Without them the system
    preserves a mistaken grouping and marks all of its facts uncertain,
    instead of questioning how those facts were brought together.

    They name two different mistakes. A merge joined two blocks that looked
    like each other; an anchor gathered records that each looked like the
    subject and were never compared with one another. Reporting an anchoring
    mistake as a merge would send a reviewer to inspect a merge that never
    happened.
    """

    incompatible_normalized_values = "incompatible_normalized_values"
    #: Two blocks were merged, and the merge may be what is wrong.
    possible_bad_merge = "possible_bad_merge"
    #: Records were gathered against the subject, and the anchor may be wrong:
    #: one of them is probably not the person searched for.
    possible_bad_anchor = "possible_bad_anchor"


class ConflictStatus(str, enum.Enum):
    unresolved = "unresolved"
    resolved = "resolved"


class SourceRunStatus(str, enum.Enum):
    """Outcome of one source for one subject.

    Four outcomes, and collapsing any two of them loses something a reader
    needs. A source that returned nothing answered the question. One that
    errored did not. One that was never applicable was not asked.
    """

    searched = "searched"
    found = "found"
    empty = "empty"
    failed = "failed"
    #: Not applicable to this subject. A text search with no text, an image
    #: search with no image. Nothing is wrong and nothing was searched.
    skipped = "skipped"


# --------------------------------------------------------------------------
# One SAEnum instance per enum, shared by every column that uses it.
#
# Two separate instances carrying the same name are what would risk a second
# CREATE TYPE for the same Postgres type. Binding each type once removes the
# possibility rather than relying on the DDL layer to collapse the duplicate.
# --------------------------------------------------------------------------

review_status_enum = SAEnum(ReviewStatus, name="review_status")
source_origin_enum = SAEnum(SourceOrigin, name="source_origin")
grouping_basis_enum = SAEnum(GroupingBasis, name="grouping_basis")
evidence_kind_enum = SAEnum(EvidenceKind, name="evidence_kind")
access_category_enum = SAEnum(AccessCategory, name="access_category")
conflict_reason_enum = SAEnum(ConflictReason, name="conflict_reason")
conflict_status_enum = SAEnum(ConflictStatus, name="conflict_status")
source_run_status_enum = SAEnum(SourceRunStatus, name="source_run_status")


# --------------------------------------------------------------------------
# Association table
# --------------------------------------------------------------------------

# The plan's JSON carries claim_ids on a conflict and conflict_ids on an
# assertion. Those are the two directions of one many-to-many.
assertion_conflict = Table(
    "assertion_conflict",
    Base.metadata,
    Column(
        "assertion_id",
        ForeignKey("assertion.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "conflict_id",
        ForeignKey("conflict.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class Subject(Base):
    """The search request: a photo hash, or a name with address and context."""

    __tablename__ = "subject"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    photo_hash: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str | None] = mapped_column(String(255))
    address: Mapped[str | None] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    candidates: Mapped[list[Candidate]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )
    # Every assertion for this subject, attached or not.
    assertions: Mapped[list[Assertion]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )
    source_runs: Mapped[list[SourceRun]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )

    @property
    def unattached_assertions(self) -> list[Assertion]:
        """Assertions that matched no candidate above threshold."""
        return [a for a in self.assertions if a.candidate_id is None]

    def origin_summary(self) -> dict[str, int]:
        """Counts backing the "5 publishers, 1 known origin, 2 unresolved" line.

        The counts answer different questions and are deliberately not
        collapsed into a single confirmation count: publishers say how many
        sites carried a claim, known origins say how many independent origins
        stand behind them, and unresolved says how much lineage is unknown.

        A provenance row with no recorded publisher cannot be deduplicated by
        name, so it is reported separately as unnamed_publishers rather than
        folded into the distinct count or dropped. Dropping it would understate
        the source count, and understating errs in the flattering direction.
        """
        publishers: set[str] = set()
        known_origins: set[str] = set()
        unnamed_publishers = 0
        unresolved = 0

        for assertion in self.assertions:
            for prov in assertion.provenances:
                if prov.publisher is None:
                    unnamed_publishers += 1
                else:
                    publishers.add(prov.publisher)
            if (
                assertion.source_origin is SourceOrigin.known_origin
                and assertion.origin_key is not None
            ):
                known_origins.add(assertion.origin_key)
            if assertion.source_origin is SourceOrigin.unknown:
                unresolved += 1

        return {
            "publishers": len(publishers),
            "known_origins": len(known_origins),
            "unnamed_publishers": unnamed_publishers,
            "unresolved": unresolved,
        }

    def __repr__(self) -> str:
        return f"<Subject id={self.id} name={self.name!r}>"


class Candidate(Base):
    """A hypothesized person, zero or more per subject."""

    __tablename__ = "candidate"
    __table_args__ = (
        CheckConstraint(
            "identity_confidence IS NULL "
            "OR (identity_confidence >= 0 AND identity_confidence <= 1)",
            name="ck_candidate_identity_confidence_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subject.id", ondelete="CASCADE"), nullable=False, index=True
    )

    display_name: Mapped[str | None] = mapped_column(String(255))

    # p: is this candidate the same human as the input? An uncalibrated
    # confidence score, not a probability.
    identity_confidence: Mapped[float | None] = mapped_column(Float)

    # What holds this candidate together. Scoring multiplies p by a factor
    # keyed on it, so a candidate reloaded without it re-scores differently
    # and the stored p stops matching the evidence.
    grouping_basis: Mapped[GroupingBasis] = mapped_column(
        grouping_basis_enum,
        default=GroupingBasis.unspecified,
        nullable=False,
    )

    review_status: Mapped[ReviewStatus] = mapped_column(
        review_status_enum,
        default=ReviewStatus.unreviewed,
        nullable=False,
    )
    # Every status change records who, when and why.
    review_changed_by: Mapped[str | None] = mapped_column(String(255))
    review_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_reason: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    subject: Mapped[Subject] = relationship(back_populates="candidates")
    assertions: Mapped[list[Assertion]] = relationship(back_populates="candidate")
    conflicts: Mapped[list[Conflict]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Candidate id={self.id} p={self.identity_confidence} "
            f"status={self.review_status.value}>"
        )


class Assertion(Base):
    """One fact claimed about a candidate, or unattached to any candidate."""

    __tablename__ = "assertion"
    __table_args__ = (
        CheckConstraint(
            "item_confidence IS NULL "
            "OR (item_confidence >= 0 AND item_confidence <= 1)",
            name="ck_assertion_item_confidence_range",
        ),
        # Re-collection must not insert a second copy of the same claim from
        # the same origin, which would inflate corroboration counts.
        #
        # NULLS NOT DISTINCT is what makes this cover the case that matters.
        # Postgres treats NULLs as distinct by default, and origin_key is NULL
        # exactly when lineage is unknown, which the plan expects to be the
        # common state. Without it the constraint would police only the rows
        # least likely to duplicate. Requires Postgres 15 or newer.
        UniqueConstraint(
            "subject_id",
            "predicate",
            "normalized_value",
            "origin_key",
            name="uq_assertion_dedup",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subject.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Nullable by design: an unattached assertion matched no candidate.
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidate.id", ondelete="SET NULL"), index=True
    )

    predicate: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    raw_value: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[str | None] = mapped_column(Text)

    # q: given the match, is this claim true and current? Conditional, not
    # marginal, so it carries the chance that a name-matched record concerns
    # a different person.
    item_confidence: Mapped[float | None] = mapped_column(Float)

    # The lineage state: is this a known origin, a suspected copy, or unknown?
    source_origin: Mapped[SourceOrigin] = mapped_column(
        source_origin_enum,
        default=SourceOrigin.unknown,
        nullable=False,
    )

    # How the source came to know this claim: the reliability axis. Read by
    # scoring for reliability and never for corroboration, which is what
    # source_origin above is for. The two are independent judgements.
    evidence_kind: Mapped[EvidenceKind] = mapped_column(
        evidence_kind_enum,
        default=EvidenceKind.unknown,
        nullable=False,
    )

    # Which origin, as opposed to what kind of lineage. A stable identifier
    # for the underlying data origin, such as "pdl", "linkedin" or
    # "serpapi:google_lens". Null when the lineage is unknown, since there is
    # then no origin to name. Corroboration counts distinct origin_key
    # values, so five publishers echoing one origin count once.
    origin_key: Mapped[str | None] = mapped_column(String(128), index=True)

    # Which source record this claim came from: one search result, one profile,
    # one row. Scoring reads it to judge attachment certainty, since a record
    # carrying only a name is weaker evidence than one carrying an email.
    # Without it a reloaded assertion cannot be re-scored to the same q.
    record_ref: Mapped[str | None] = mapped_column(String(255), index=True)

    # Which adapter produced this row. Records the collection path, not the
    # data's origin, and the two differ whenever an adapter returns
    # republished content.
    collector: Mapped[str] = mapped_column(String(128), nullable=False)

    # Four independent timestamps. Observation time does not establish
    # validity, and fetch time must never stand in for either.
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Acceptance is per-assertion: a reviewed candidate does not imply that
    # every claim attached to it was accepted as true.
    review_status: Mapped[ReviewStatus] = mapped_column(
        review_status_enum,
        default=ReviewStatus.unreviewed,
        nullable=False,
    )
    review_changed_by: Mapped[str | None] = mapped_column(String(255))
    review_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_reason: Mapped[str | None] = mapped_column(Text)

    subject: Mapped[Subject] = relationship(back_populates="assertions")
    candidate: Mapped[Candidate | None] = relationship(back_populates="assertions")
    provenances: Mapped[list[Provenance]] = relationship(
        back_populates="assertion", cascade="all, delete-orphan"
    )
    conflicts: Mapped[list[Conflict]] = relationship(
        secondary=assertion_conflict, back_populates="assertions"
    )

    @property
    def attribution(self) -> float | None:
        """r = p times q. Derived, never estimated or stored directly.

        Returns None when either component is missing, which includes every
        unattached assertion: with no candidate there is no p, so no
        attribution exists. A missing score is reported as unknown rather
        than defaulted, matching the plan's treatment of unknown freshness.

        Deliberately not min(p, q), which would destroy signal: at p=0.40 it
        returns 0.40 whether q is 0.95 or 0.55, where the product separates
        those cases as 0.38 and 0.22.
        """
        if self.item_confidence is None:
            return None
        if self.candidate is None or self.candidate.identity_confidence is None:
            return None
        return self.candidate.identity_confidence * self.item_confidence

    @property
    def conflict_ids(self) -> list[int]:
        """The plan's conflict_ids view of the many-to-many."""
        return [c.id for c in self.conflicts]

    def __repr__(self) -> str:
        return (
            f"<Assertion id={self.id} predicate={self.predicate!r} "
            f"q={self.item_confidence} candidate_id={self.candidate_id}>"
        )


class Provenance(Base):
    """Where one assertion came from, and which stored artifact records it."""

    __tablename__ = "provenance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assertion_id: Mapped[int] = mapped_column(
        ForeignKey("assertion.id", ondelete="CASCADE"), nullable=False, index=True
    )

    publisher: Mapped[str | None] = mapped_column(String(255), index=True)

    access_category: Mapped[AccessCategory] = mapped_column(
        access_category_enum,
        default=AccessCategory.PUBLIC_WEB,
        nullable=False,
    )

    # The cache key from app.cache, naming the stored raw response. That
    # artifact preserves the response, not the record it references.
    artifact_ref: Mapped[str | None] = mapped_column(String(64), index=True)

    assertion: Mapped[Assertion] = relationship(back_populates="provenances")

    def __repr__(self) -> str:
        return f"<Provenance id={self.id} publisher={self.publisher!r}>"


class Conflict(Base):
    """Two or more assertions held to be incompatible. Stored, not resolved."""

    __tablename__ = "conflict"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Nullable: a conflict among unattached assertions has no candidate.
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidate.id", ondelete="CASCADE"), index=True
    )

    predicate: Mapped[str | None] = mapped_column(String(128), index=True)

    reason: Mapped[ConflictReason] = mapped_column(
        conflict_reason_enum, nullable=False
    )
    status: Mapped[ConflictStatus] = mapped_column(
        conflict_status_enum,
        default=ConflictStatus.unresolved,
        nullable=False,
    )

    # A derived preference, recorded without discarding the alternatives.
    preferred_assertion_id: Mapped[int | None] = mapped_column(
        ForeignKey("assertion.id", ondelete="SET NULL")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    candidate: Mapped[Candidate | None] = relationship(back_populates="conflicts")
    assertions: Mapped[list[Assertion]] = relationship(
        secondary=assertion_conflict, back_populates="conflicts"
    )
    preferred_assertion: Mapped[Assertion | None] = relationship(
        foreign_keys=[preferred_assertion_id]
    )

    @property
    def claim_ids(self) -> list[int]:
        """The plan's claim_ids view of the conflicting assertions."""
        return [a.id for a in self.assertions]

    def __repr__(self) -> str:
        return f"<Conflict id={self.id} reason={self.reason.value}>"


class SourceRun(Base):
    """Per-source outcome for one subject.

    Records what was searched and what remains unknown, because "find all
    footprints" has no completion criterion.
    """

    __tablename__ = "source_run"
    __table_args__ = (
        CheckConstraint(
            "result_count IS NULL OR result_count >= 0",
            name="ck_source_run_result_count_nonneg",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subject.id", ondelete="CASCADE"), nullable=False, index=True
    )

    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    status: Mapped[SourceRunStatus] = mapped_column(
        source_run_status_enum, nullable=False
    )

    # The N in "found N". Null for any other outcome.
    result_count: Mapped[int | None] = mapped_column(Integer)

    # Set only when status is failed. Empty is not failed.
    error_reason: Mapped[str | None] = mapped_column(Text)

    ran_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    subject: Mapped[Subject] = relationship(back_populates="source_runs")

    def __repr__(self) -> str:
        return (
            f"<SourceRun id={self.id} source={self.source!r} "
            f"status={self.status.value} n={self.result_count}>"
        )
