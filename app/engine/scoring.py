"""Two estimated scores and one derived from them.

``p`` asks whether this candidate is the same human as the subject.
``q`` asks, *given that the match is right*, whether a claim is true and current.
``r = p * q`` is derived from them and never estimated on its own.

These are uncalibrated heuristic scores, not probabilities. There is no labeled
outcome set to calibrate against, so the combination rule mirrors the chain rule
without being one. It is also lossy: 0.4 x 0.9 and 0.9 x 0.4 both give 0.36,
which is why p and q must be displayed alongside r and never replaced by it.

Three rules are structural rather than stylistic:

* **q is never capped at p.** ``min(p, q)`` destroys signal. At p=0.40 it
  returns 0.40 whether q is 0.95 or 0.55, where the product separates those as
  0.38 and 0.22. Nothing here reads p while computing q.
* **Corroboration counts distinct origins**, never publishers and never drafts.
  Five sites republishing one aggregator is one origin. A suspected copy earns
  no independence bonus and unknown lineage is never counted as independent.
* **Evidence is not counted twice.** Records consumed establishing the identity
  match are discounted, not excluded, when they also corroborate a claim.

Reliability and lineage are separate axes and are read from separate fields.
``evidence_kind`` says how a source came to know a claim and feeds reliability;
``source_origin`` says where the claim came from and feeds corroboration. One
field serving both would make every lineage judgement double as a reliability
judgement.

Access category is deliberately absent from both calculations. Licensing is an
access right, not a reliability signal; it is recorded as metadata elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.adapters.base import AssertionDraft, Query
from app.engine.candidates import (
    DISTINCTIVE_PREDICATES,
    LOCALITY_PREDICATES,
    CandidateDraft,
    normalize_locality_key,
    normalize_name_key,
)
from app.engine.faces import similarity_to_distance
from app.models import EvidenceKind, GroupingBasis, SourceOrigin


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringConfig:
    """Every weight and threshold the scorers use.

    Nothing in the logic below carries a numeric literal. Tuning happens here,
    where a reviewer can see the whole set of judgements at once.
    """

    # --- identity (p) signal weights ---------------------------------------
    # Relative importance in a weighted mean. A signal that is unavailable is
    # dropped from both numerator and denominator rather than scored zero, so
    # absent evidence is not treated as evidence against.
    #: A face is direct evidence about the human. Name, locality and
    #: corroboration are proxies for it. Weighted so that when a face is
    #: available it drives the result, which is the whole reason a photo is
    #: worth asking for.
    weight_face: float = 6.0
    weight_name: float = 2.0
    weight_locality: float = 1.5
    weight_corroboration: float = 1.5

    # --- face distance -----------------------------------------------------
    #: At or below this cosine distance the face signal is full strength.
    face_match_distance: float = 0.20
    #: At or above this it contributes nothing. Between the two it interpolates.
    face_reject_distance: float = 0.60

    # --- name match strength -----------------------------------------------
    name_exact: float = 1.00
    #: One name's tokens are a subset of the other: "marcus webb" in
    #: "marcus a webb". Common between a profile and a public record.
    name_subset: float = 0.70
    name_partial: float = 0.40
    name_none: float = 0.00
    #: Jaccard overlap at or above which a non-subset name pair counts partial.
    name_partial_overlap: float = 0.50

    # --- locality ----------------------------------------------------------
    locality_match: float = 1.00
    locality_mismatch: float = 0.00

    # --- corroboration -----------------------------------------------------
    #: Distinct corroborating origins beyond the first at which the bonus is
    #: full. Three means one origin gives nothing and four give the maximum.
    corroboration_saturation: float = 3.0
    #: Independence credit for a suspected copy. Zero by policy: a copy is
    #: retained as evidence but earns no independence. Exposed as a knob so the
    #: policy can be revisited without editing logic.
    suspected_copy_independence: float = 0.00
    #: Independence credit for unknown lineage. Zero: unknown is surfaced as
    #: unknown, never silently counted as independent.
    unknown_origin_independence: float = 0.00

    # --- what holds the candidate together ---------------------------------
    #: Multiplier on p for the grouping basis. A candidate is a claim that
    #: several records describe one person, and that claim is only as good as
    #: what joined them. A shared name is barely evidence, so a name-only
    #: grouping cannot reach the same p as one anchored to the subject or held
    #: together by a shared identifier, however well its other signals score.
    #: ``unspecified`` is neutral so a hand-built candidate is not penalised
    #: for a judgement nobody made.
    grouping_basis_factor: Mapping[GroupingBasis, float] = field(
        default_factory=lambda: {
            GroupingBasis.anchor: 1.00,
            GroupingBasis.distinctive_attribute: 1.00,
            GroupingBasis.face: 0.95,
            GroupingBasis.name_and_locality: 0.85,
            GroupingBasis.name_only: 0.40,
            GroupingBasis.unspecified: 1.00,
        }
    )

    # --- negative signals --------------------------------------------------
    # Subtracted after the weighted mean. These are affirmative contradictions,
    # which is a different thing from missing evidence: a missing locality is
    # dropped from the mean, while a contradicting one both scores zero in the
    # mean and incurs the penalty below.
    penalty_contradictory_birth_date: float = 0.25
    penalty_incompatible_locality: float = 0.30

    # --- item (q) signal weights -------------------------------------------
    weight_reliability: float = 2.0
    weight_freshness: float = 1.5
    weight_item_corroboration: float = 1.5

    #: Source reliability for this claim, keyed on how the source came to know
    #: it. Read from evidence_kind and never from lineage: the two are separate
    #: axes, and a first-party origin can still relay something it never
    #: observed. Lineage is used for corroboration and nothing else.
    reliability_by_evidence_kind: Mapping[EvidenceKind, float] = field(
        default_factory=lambda: {
            EvidenceKind.direct_observation: 0.95,
            EvidenceKind.self_reported: 0.75,
            EvidenceKind.secondhand: 0.60,
            EvidenceKind.unknown: 0.50,
            EvidenceKind.republished: 0.45,
        }
    )

    #: Corroborating origins at which the item bonus saturates.
    item_corroboration_saturation: float = 2.0
    #: Credit for an origin whose records were already consumed establishing
    #: the identity match. Discounted rather than excluded, per the no
    #: double-counting rule: the evidence is still real, it is just not new.
    reused_evidence_weight: float = 0.35

    # --- recency -----------------------------------------------------------
    #: Half-life in days, per predicate. Contact details go stale quickly.
    half_life_days_by_predicate: Mapping[str, float] = field(
        default_factory=lambda: {
            "phone": 180.0,
            "address": 365.0,
            "email": 540.0,
            "city": 540.0,
            "employer": 540.0,
            "job_title": 540.0,
            "name": 3650.0,
        }
    )
    default_half_life_days: float = 730.0
    #: Facts that do not go stale. A birth date is as true today as when it
    #: was recorded, so age carries no information about it.
    non_decaying_predicates: frozenset[str] = frozenset({"birth_date"})

    #: Stand-in freshness when observed_at is null. Neither fresh nor stale.
    #: The accompanying flag is what callers must surface; this value exists
    #: only so the weighted mean has something to consume. Never derived from
    #: fetched_at: a mirror fetched today must not refresh an old claim.
    freshness_when_unknown: float = 0.50

    # --- record-attachment uncertainty -------------------------------------
    # A multiplier on q. Even granting that the candidate is the right person,
    # this particular record may describe someone else who shares the name.
    # This is what makes q conditional rather than marginal.
    attachment_distinctive: float = 1.00
    attachment_name_and_locality: float = 0.85
    attachment_name_only: float = 0.65
    attachment_unknown: float = 0.55

    # --- bounds ------------------------------------------------------------
    score_floor: float = 0.0
    #: The value of a signal that is fully present. Not a cap on p: it is what
    #: an exact name match, a certain face match or saturated corroboration is
    #: worth, and it is the default grouping_basis_factor. Those are a different
    #: quantity from "the most confident this system may claim to be", and
    #: putting both in one field would make either impossible to change alone.
    score_ceiling: float = 1.0
    #: The most p may ever read. Every input to p is an uncalibrated heuristic,
    #: so 1.00 would assert a certainty none of them support, and it would
    #: contradict the disclaimer printed beside it on every page.
    p_ceiling: float = 0.95
    #: The most q may ever read, for exactly the reason p has a ceiling.
    #: Reliability is a table of guesses about publishers, freshness is a decay
    #: curve nobody calibrated, and corroboration counts websites. None of them
    #: can support "certain", so q must not print it either. Without this, p was
    #: honest about its limits and q was not, in a product whose whole claim is
    #: that it shows you how much to trust what it found.
    q_ceiling: float = 0.95


# --------------------------------------------------------------------------
# Score types
# --------------------------------------------------------------------------


class _Score(float):
    """A float that carries how it was arrived at.

    Subclassing float keeps the declared return type honest, so callers can do
    arithmetic directly, while leaving room for the component breakdown the UI
    needs in order to show p and q beside r instead of only r.
    """

    components: Mapping[str, float | None]

    def __new__(cls, value: float, **extra: object) -> "_Score":
        obj = super().__new__(cls, value)
        for key, item in extra.items():
            object.__setattr__(obj, key, item)
        return obj


class IdentityScore(_Score):
    """p, with the signals that produced it.

    ``consumed_record_refs`` names the source records this match was built
    from. Passing it to :func:`score_item` is what prevents the same evidence
    from being counted a second time as independent corroboration.
    """

    consumed_record_refs: frozenset[str]
    penalties: Mapping[str, float]
    #: What held the candidate together, and therefore what p is a claim about.
    grouping_basis: GroupingBasis


class ItemScore(_Score):
    """q, with an explicit statement about freshness.

    ``freshness_unknown`` is true when the claim carried no ``observed_at``.
    Callers must surface it rather than presenting the score as though its
    recency were established.
    """

    freshness_unknown: bool
    attachment_basis: str


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def _clamp(value: float, config: ScoringConfig) -> float:
    return max(config.score_floor, min(config.score_ceiling, value))


def _weighted_mean(signals: Iterable[tuple[float, float]]) -> float:
    """Mean of (value, weight) pairs. Unavailable signals are simply absent."""
    pairs = list(signals)
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total_weight


def _subject_locality(subject: Query) -> str | None:
    """The locality the caller supplied, from address or context."""
    for raw in (getattr(subject, "address", None), getattr(subject, "context", None)):
        key = normalize_locality_key(raw)
        if key:
            return key
    return None


def _tokens(key: str | None) -> set[str]:
    return set(key.split()) if key else set()


def _localities_agree(candidate_locality: str | None, subject_locality: str) -> bool:
    """Token overlap, so "austin tx" and "austin texas" agree."""
    return bool(_tokens(candidate_locality) & _tokens(subject_locality))


def _independence_credit(origin: SourceOrigin, config: ScoringConfig) -> float:
    """How much independence one origin in this lineage state is worth."""
    if origin is SourceOrigin.known_origin:
        return 1.0
    if origin is SourceOrigin.suspected_copy:
        return config.suspected_copy_independence
    return config.unknown_origin_independence


def _origins_with_credit(
    assertions: Iterable[AssertionDraft], config: ScoringConfig
) -> dict[str, float]:
    """Best independence credit available per distinct origin_key.

    Keyed by origin, so repeated drafts and repeated publishers from one origin
    collapse to a single entry. This is the only place corroboration is counted
    and it never sees a publisher or a draft count.
    """
    credits: dict[str, float] = {}
    for assertion in assertions:
        if not assertion.origin_key:
            continue
        credit = _independence_credit(assertion.source_origin, config)
        previous = credits.get(assertion.origin_key)
        if previous is None or credit > previous:
            credits[assertion.origin_key] = credit
    return credits


# --------------------------------------------------------------------------
# p: identity confidence
# --------------------------------------------------------------------------


def _anchor_name_similarity(candidate: CandidateDraft) -> float | None:
    """The name agreement anchoring actually measured, per record.

    An anchor candidate is built with ``name_key`` set to the *subject's* name,
    because that is the hypothesis it represents. Comparing that key back to the
    subject therefore compares the subject's name with itself and returns a
    perfect 1.00 for every anchor, whatever its records are really called: a
    record titled "Michael Petrie Email & Phone Number" was measured at 0.85
    during anchoring and then scored as 1.00 here.

    That is the same mistake this module is otherwise careful about — treating a
    label as though it were evidence — so the measured figures are used instead.
    The weakest is taken, not the strongest: the score answers whether all these
    records are one person, and the least convincing member is what that claim
    rests on.
    """
    matches = getattr(candidate, "anchor_matches", None) or {}
    measured = [
        m.name_signal for m in matches.values() if getattr(m, "name_signal", None)
    ]
    return min(measured) if measured else None


def _anchor_face_similarity(candidate: CandidateDraft) -> float | None:
    """The strongest face match recorded while this candidate was anchored.

    Anchoring already compared every record's images against the subject photo.
    Recomputing that here would be waste; ignoring it, as this once did, throws
    away the best evidence on the page. A photo-only subject has no name and no
    context, so without this p has nothing to stand on and collapses to zero
    while the report says "held by face match 0.99" directly above it.
    """
    matches = getattr(candidate, "anchor_matches", None) or {}
    scores = [
        match.face_signal
        for match in matches.values()
        if getattr(match, "face_signal", None) is not None
    ]
    return max(scores) if scores else None


def _face_signal(distance: float | None, config: ScoringConfig) -> float | None:
    """Full strength below the match distance, nothing above the reject one."""
    if distance is None:
        return None
    if distance <= config.face_match_distance:
        return config.score_ceiling
    if distance >= config.face_reject_distance:
        return config.score_floor
    span = config.face_reject_distance - config.face_match_distance
    if span <= 0:
        return config.score_floor
    return (config.face_reject_distance - distance) / span


def _name_signal(
    candidate_name: str | None, subject_name: str | None, config: ScoringConfig
) -> float | None:
    """Exact, subset, partial or none. None when there is nothing to compare."""
    if not candidate_name or not subject_name:
        return None
    if candidate_name == subject_name:
        return config.name_exact

    candidate_tokens, subject_tokens = _tokens(candidate_name), _tokens(subject_name)
    if not candidate_tokens or not subject_tokens:
        return config.name_none
    if candidate_tokens <= subject_tokens or subject_tokens <= candidate_tokens:
        return config.name_subset

    union = candidate_tokens | subject_tokens
    overlap = len(candidate_tokens & subject_tokens) / len(union)
    return config.name_partial if overlap >= config.name_partial_overlap else config.name_none


def _locality_signal(
    candidate: CandidateDraft, subject_locality: str | None, config: ScoringConfig
) -> tuple[float | None, bool]:
    """Locality agreement, and whether it is an affirmative contradiction."""
    known = {key for key in candidate.locality_keys if key}
    if subject_locality is None or not known:
        return None, False
    if any(_localities_agree(key, subject_locality) for key in known):
        return config.locality_match, False
    return config.locality_mismatch, True


#: What two origins may not corroborate each other with. A name is the thing
#: this system refuses to treat as evidence of identity, and a locality is left
#: out of :data:`DISTINCTIVE_PREDICATES` on the same reasoning: large numbers of
#: unrelated people share both. Agreeing on one is not independent testimony,
#: it is the coincidence that put the records in the same pile to begin with.
CORROBORATION_BLIND_PREDICATES: frozenset[str] = frozenset({"name"}) | frozenset(
    LOCALITY_PREDICATES
)


def _corroborating_origins(
    candidate: CandidateDraft, config: ScoringConfig
) -> dict[str, float]:
    """Origins that agree with some other origin about something identifying.

    Corroboration is meant to say that independent sources tell the same story.
    It was counting the distinct origins *present* on the candidate and never
    asking whether any two of them agreed — so for a group of same-name
    strangers, every additional stranger raised confidence that the group was
    one person. That is the inversion this whole system exists to prevent,
    arriving through the signal that is supposed to guard against it.

    Two origins agree when they assert the same value for the same predicate,
    excluding the name and the locality. ``score_identity`` already described
    corroboration as "distinct origins that agree"; this makes that true.
    """
    credits = _origins_with_credit(candidate.assertions, config)
    if len(credits) < 2:
        return {}

    claims: dict[tuple[str, str], set[str]] = {}
    for assertion in candidate.assertions:
        if not assertion.origin_key:
            continue
        if assertion.predicate in CORROBORATION_BLIND_PREDICATES:
            continue
        value = assertion.normalized_value or assertion.raw_value
        if not value:
            continue
        claims.setdefault((assertion.predicate, value), set()).add(
            assertion.origin_key
        )

    agreeing: set[str] = set()
    for origins in claims.values():
        if len(origins) > 1:
            agreeing |= origins

    return {key: credit for key, credit in credits.items() if key in agreeing}


def _corroboration_signal(candidate: CandidateDraft, config: ScoringConfig) -> float:
    """Independence beyond a single agreeing origin, saturating."""
    credits = _corroborating_origins(candidate, config)
    effective = max(0.0, sum(credits.values()) - 1.0)
    if config.corroboration_saturation <= 0:
        return config.score_ceiling if effective > 0 else config.score_floor
    return min(config.score_ceiling, effective / config.corroboration_saturation)


def _has_contradictory_birth_date(candidate: CandidateDraft) -> bool:
    values = {
        a.normalized_value
        for a in candidate.assertions
        if a.predicate == "birth_date" and a.normalized_value
    }
    return len(values) > 1


def score_identity(
    candidate: CandidateDraft,
    subject: Query,
    config: ScoringConfig | None = None,
    face_distance: float | None = None,
) -> IdentityScore:
    """p: is this candidate the same human as the subject?

    Combines the signals that are actually available as a weighted mean, then
    subtracts penalties for affirmative contradictions. A signal that cannot be
    computed, such as face distance with no embedding, is left out of the mean
    entirely rather than scored zero, so silence is not mistaken for dissent.

    ``face_distance`` is injected because this module owns no face model. When
    it is None the face signal is simply unavailable.

    The candidate's ``grouping_basis`` scales the result. A candidate whose
    records share only a name cannot reach the score of one anchored to the
    subject, because p answers "is this the same human", and a shared name is
    not an answer to that question.

    The result is capped at ``config.p_ceiling`` (0.95). Every input here is an
    uncalibrated heuristic — a name similarity, a cosine distance, a count of
    origins — and no combination of them earns 1.00, which a reader takes as
    certainty and which would contradict the "not a probability" disclaimer
    printed beside the figure. The cap is applied after the basis factor so
    nothing downstream can raise it again.

    Note that the cap only bounds the number; it does not make it right. p rose
    with corroboration, and corroboration counts distinct origins that agree, so
    several *different* people sharing a name once corroborated one another
    upward. That is fixed where it is caused, in candidate formation, not here.
    """
    config = config or ScoringConfig()

    subject_name = normalize_name_key(getattr(subject, "name", None))
    subject_locality = _subject_locality(subject)

    face = _face_signal(face_distance, config)
    if face is None:
        # Fall back to what anchoring measured. similarity_to_distance is the
        # one place the sign flips, so it is used here rather than repeated.
        similarity = _anchor_face_similarity(candidate)
        if similarity is not None:
            face = _face_signal(similarity_to_distance(similarity), config)

    name = _anchor_name_similarity(candidate)
    if name is None:
        name = _name_signal(candidate.name_key, subject_name, config)
    locality, locality_contradicts = _locality_signal(
        candidate, subject_locality, config
    )
    corroboration = _corroboration_signal(candidate, config)

    signals: list[tuple[float, float]] = [(corroboration, config.weight_corroboration)]
    if face is not None:
        signals.append((face, config.weight_face))
    if name is not None:
        signals.append((name, config.weight_name))
    if locality is not None:
        signals.append((locality, config.weight_locality))

    base = _weighted_mean(signals)

    penalties: dict[str, float] = {}
    if _has_contradictory_birth_date(candidate):
        penalties["contradictory_birth_date"] = config.penalty_contradictory_birth_date
    if locality_contradicts:
        penalties["incompatible_locality"] = config.penalty_incompatible_locality

    # The grouping basis prices the claim that these records are one person.
    # Applied after penalties and before clamping, so it scales the whole
    # score rather than one signal: no amount of corroboration rescues a
    # candidate whose members share only a name.
    basis = getattr(candidate, "grouping_basis", GroupingBasis.unspecified)
    basis_factor = config.grouping_basis_factor.get(basis, config.score_ceiling)
    value = _clamp((base - sum(penalties.values())) * basis_factor, config)
    # Last, so nothing downstream can lift it back. See ``p_ceiling``.
    value = min(value, config.p_ceiling)

    # Every record that contributed to this match, so that item scoring can
    # recognise the same evidence arriving a second time.
    consumed = frozenset(
        a.record_ref for a in candidate.assertions if a.record_ref is not None
    )

    return IdentityScore(
        value,
        components={
            "face": face,
            "name": name,
            "locality": locality,
            "corroboration": corroboration,
            "base": base,
            "grouping_basis_factor": basis_factor,
        },
        grouping_basis=basis,
        penalties=penalties,
        consumed_record_refs=consumed,
    )


# --------------------------------------------------------------------------
# q: item confidence
# --------------------------------------------------------------------------


def _freshness(
    assertion: AssertionDraft, config: ScoringConfig, now: datetime | None = None
) -> tuple[float, bool]:
    """Recency from observed_at, and whether freshness is unknown.

    Decay is exponential on a per-predicate half-life. ``fetched_at`` is never
    consulted: when a source did not say when it observed something, that fact
    is unknown, and a mirror collected today must not refresh a five-year-old
    claim.
    """
    if assertion.predicate in config.non_decaying_predicates:
        return config.score_ceiling, False

    if assertion.observed_at is None:
        return config.freshness_when_unknown, True

    now = now or datetime.now(timezone.utc)
    observed = assertion.observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)

    age_days = (now - observed).total_seconds() / 86400.0
    if age_days <= 0:
        return config.score_ceiling, False

    half_life = config.half_life_days_by_predicate.get(
        assertion.predicate, config.default_half_life_days
    )
    if half_life <= 0:
        return config.score_floor, False

    return _clamp(0.5 ** (age_days / half_life), config), False


def _corroborating_assertions(
    assertion: AssertionDraft, candidate: CandidateDraft
) -> list[AssertionDraft]:
    """Other assertions on the candidate stating the very same claim."""
    return [
        other
        for other in candidate.assertions
        if other.predicate == assertion.predicate
        and other.normalized_value == assertion.normalized_value
    ]


def _item_corroboration(
    assertion: AssertionDraft,
    candidate: CandidateDraft,
    config: ScoringConfig,
    consumed_record_refs: frozenset[str],
) -> float:
    """Independent origins agreeing with this exact claim.

    Origins already consumed establishing the identity match are discounted
    rather than dropped: the agreement is real, it is simply not new evidence.
    """
    same_claim = _corroborating_assertions(assertion, candidate)
    credits = _origins_with_credit(same_claim, config)
    credits.pop(assertion.origin_key, None)
    if not credits:
        return config.score_floor

    # An origin counts as reused when every record supporting it here was
    # already spent on the identity match.
    refs_by_origin: dict[str, set[str | None]] = {}
    for other in same_claim:
        if other.origin_key in credits:
            refs_by_origin.setdefault(other.origin_key, set()).add(other.record_ref)

    effective = 0.0
    for origin, credit in credits.items():
        refs = {ref for ref in refs_by_origin.get(origin, set()) if ref is not None}
        reused = bool(refs) and refs <= consumed_record_refs
        effective += credit * (config.reused_evidence_weight if reused else 1.0)

    if config.item_corroboration_saturation <= 0:
        return config.score_ceiling if effective > 0 else config.score_floor
    return min(config.score_ceiling, effective / config.item_corroboration_saturation)


def _attachment_certainty(
    assertion: AssertionDraft, candidate: CandidateDraft, config: ScoringConfig
) -> tuple[float, str]:
    """How sure we are this record is about the candidate at all.

    This is what keeps q conditional rather than marginal. Even granting the
    candidate is the right person, a record attached on name alone may describe
    a different person who happens to share it.
    """
    if assertion.record_ref is None:
        return config.attachment_unknown, "no_record_ref"

    siblings = [
        other
        for other in candidate.assertions
        if other.record_ref == assertion.record_ref
    ]
    predicates = {other.predicate for other in siblings}

    if predicates & DISTINCTIVE_PREDICATES:
        return config.attachment_distinctive, "distinctive_attribute"
    has_name = "name" in predicates
    has_locality = bool(predicates & {"city", "locality", "region", "state", "address"})
    if has_name and has_locality:
        return config.attachment_name_and_locality, "name_and_locality"
    if has_name:
        return config.attachment_name_only, "name_only"
    return config.attachment_unknown, "no_identifying_fields"


def score_item(
    assertion: AssertionDraft,
    candidate: CandidateDraft,
    config: ScoringConfig | None = None,
    consumed_record_refs: frozenset[str] | None = None,
    now: datetime | None = None,
) -> ItemScore:
    """q: given the match is right, is this claim true and current?

    Conditional on the identity match, and computed without ever reading p.
    Nothing here caps q at p, and nothing here compares the two.

    ``consumed_record_refs`` comes from :attr:`IdentityScore.consumed_record_refs`.
    Passing it lets corroboration discount evidence already spent on the match.
    Omitting it is safe but scores generously, since every corroborating origin
    then looks new.
    """
    config = config or ScoringConfig()
    consumed = consumed_record_refs or frozenset()

    reliability = config.reliability_by_evidence_kind.get(
        assertion.evidence_kind, config.score_floor
    )
    freshness, freshness_unknown = _freshness(assertion, config, now=now)
    corroboration = _item_corroboration(assertion, candidate, config, consumed)

    base = _weighted_mean(
        [
            (reliability, config.weight_reliability),
            (freshness, config.weight_freshness),
            (corroboration, config.weight_item_corroboration),
        ]
    )

    certainty, basis = _attachment_certainty(assertion, candidate, config)
    value = _clamp(base * certainty, config)
    # Last, so nothing downstream can lift it back. See ``q_ceiling``.
    value = min(value, config.q_ceiling)

    return ItemScore(
        value,
        components={
            "reliability": reliability,
            "freshness": freshness,
            "corroboration": corroboration,
            "attachment_certainty": certainty,
            "base": base,
        },
        freshness_unknown=freshness_unknown,
        attachment_basis=basis,
    )


# --------------------------------------------------------------------------
# r: derived, never estimated
# --------------------------------------------------------------------------


def derive_attribution(p: float | None, q: float | None) -> float | None:
    """r = p x q, computed here and nowhere else.

    Returns None when either component is missing, which includes every
    unattached assertion: with no candidate there is no p, so no attribution
    exists. Unknown is reported as unknown rather than defaulted.

    Never ``min(p, q)``. The product keeps p and q distinguishable in the
    result; the minimum discards whichever is larger.
    """
    if p is None or q is None:
        return None
    return float(p) * float(q)
