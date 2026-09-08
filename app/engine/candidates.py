"""Candidate formation: grouping and attachment only.

No scoring and no conflict detection happen here. This module decides how many
people the evidence describes and which claims belong to each, which is a
prerequisite for scoring and a different question from how confident we are.

The pipeline, and why each step exists:

1. **Records.** Drafts that arrived together in one source record are grouped,
   because a name and a locality only form a blocking key if they describe the
   same person. See ``_record_key`` for the proxy used and its limits.
2. **Blocking.** Records are bucketed by normalized name plus locality. This is
   a cheap prefilter that proposes groups to evaluate. It decides nothing.
3. **Merging.** Two blocks combine only on positive evidence: a face distance
   below threshold, or a shared distinctive attribute. A shared name is never
   positive evidence, so two people called Michael Petrie in different cities
   remain two candidates.
4. **Attachment.** Each draft joins the best-matching candidate only if the
   match clears a threshold. Otherwise it is reported unattached rather than
   forced onto the nearest candidate.

Over-merging is treated as worse than fragmenting. A fragmented result shows a
reviewer two candidates that may be one person, which is visible and fixable.
An over-merged result invents a person who does not exist and lends every claim
attached to them false support.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from app.adapters.base import AssertionDraft, Query
from app.models import GroupingBasis

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

#: Predicates that can supply a locality, most specific first. ``address`` is
#: last and deliberately so: a full street address normalizes to a narrow key
#: that will not match a bare city, which fragments rather than over-merges.
LOCALITY_PREDICATES: tuple[str, ...] = (
    "city",
    "locality",
    "region",
    "state",
    "address",
)

#: Attributes distinctive enough that sharing one is positive evidence of the
#: same person. Deliberately excludes name, employer and locality, all of which
#: large numbers of unrelated people share.
DISTINCTIVE_PREDICATES: frozenset[str] = frozenset(
    {
        "email",
        "phone",
        "government_id",
        "ssn",
        "national_id",
        "drivers_license",
        "passport_number",
    }
)

#: Employer alone is common; employer together with job title is not. Both must
#: match for the pair to count as one distinctive signal.
COMPOSITE_DISTINCTIVE: tuple[tuple[str, ...], ...] = (("employer", "job_title"),)

#: Generational suffixes dropped from a name key. Bare "v" is excluded on
#: purpose: it is indistinguishable from a middle initial.
_NAME_SUFFIXES: frozenset[str] = frozenset({"jr", "jnr", "sr", "snr", "ii", "iii", "iv"})

_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)

#: Possessives, removed before punctuation so "Michael Petrie's Post" does not
#: become the three-token name "michael petrie s".
_POSSESSIVE = re.compile("['‘’ʼ]s\\b", re.IGNORECASE)

#: Trailing tokens that belong to a page, not to a person. Search titles carry
#: the platform and the content type: "Michael Petrie's Post", "Michael Petrie
#: on LinkedIn", "Michael Petrie - Wikipedia". Left in, each spawns a separate
#: person from a single record.
_NAME_TAIL_TOKENS: frozenset[str] = frozenset(
    {
        # content type
        "post", "posts", "profile", "profiles", "page", "pages", "photo",
        "photos", "video", "videos", "biography", "obituary", "resume", "cv",
        "overview", "about", "bio",
        # platform and site names
        "linkedin", "facebook", "twitter", "instagram", "github", "youtube",
        "tiktok", "reddit", "medium", "substack", "wikipedia", "wikidata",
        "britannica", "imdb", "crunchbase", "zoominfo", "spokeo", "whitepages",
        "pinterest", "threads", "quora", "flickr",
        # connectives a stripped tail leaves behind
        "on", "at", "in", "the", "and", "of", "from",
    }
)

#: Never strip a name below this many tokens. "Michael Page" is a person as
#: well as a company, and losing the surname is worse than keeping a tail.
_MIN_NAME_TOKENS = 2

#: Dropped when comparing a subject's context against a record's snippet.
#: Small on purpose: over-filtering silently weakens the anchor.
_CONTEXT_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "was", "were",
        "are", "his", "her", "its", "their", "has", "have", "had", "who",
        "she", "him", "they", "them", "not", "but", "all", "any", "our",
    }
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FormationConfig:
    """Every threshold and weight used by formation.

    Named rather than inlined so a reviewer can see what the system's idea of
    "close enough" actually is, and change it without editing logic.
    """

    #: A draft attaches only when its match strength exceeds this, strictly.
    attachment_threshold: float = 0.60

    #: Faces merge when distance is strictly below this. Cosine distance on
    #: whatever embedding the injected callable uses.
    face_distance_threshold: float = 0.38

    # Match strengths, all in 0..1.
    #: The draft's own record is part of the candidate.
    direct_membership: float = 1.00
    #: Name and locality both agree.
    name_and_locality_match: float = 0.90
    #: Name agrees, this record has no locality to compare.
    name_match_locality_absent: float = 0.45
    #: Name agrees, localities disagree.
    name_match_locality_conflict: float = 0.15
    #: No name in common.
    no_name_match: float = 0.00

    #: Added when a distinctive attribute is shared. Large enough that shared
    #: positive evidence carries a draft over the threshold on its own, which
    #: is what lets a shared email merge across differing localities.
    distinctive_attribute_bonus: float = 0.65

    # --- subject anchoring -------------------------------------------------
    # The subject is a hypothesis, not just another cluster. A record that
    # independently matches the supplied name and context joins the anchor
    # candidate. Records that do not fall through to blind clustering, so
    # other people with the same name still form their own candidates and stay
    # visible: showing which Michael Petries were rejected, and why, is the
    # point of the screen rather than noise to be hidden.
    #: A record joins the anchor when its match strength exceeds this.
    anchor_threshold: float = 0.65
    weight_anchor_name: float = 2.0
    weight_anchor_context: float = 1.5
    weight_anchor_locality: float = 1.5
    #: A face match is direct evidence about the person, where a name is only a
    #: label many people share. It is weighted accordingly.
    weight_anchor_face: float = 4.0

    anchor_name_exact: float = 1.00
    #: The subject's name sits inside the record's name or title, as in
    #: "About Ada Lovelace" for the subject "Ada Lovelace".
    anchor_name_subject_contained: float = 0.85
    #: The record carries only part of the subject's name.
    anchor_name_record_contained: float = 0.70
    anchor_name_partial: float = 0.40
    anchor_name_none: float = 0.00
    #: Jaccard overlap at or above which a non-containing name pair counts
    #: as a partial match.
    anchor_name_partial_overlap: float = 0.50

    #: How many matching context terms constitute full context agreement.
    #:
    #: Context used to score as a share of the terms the caller typed, which is
    #: a scale no record can reach: a search snippet is ~150 characters and
    #: cannot contain six context terms. Across every run of the Michael Petrie
    #: corpus the highest context signal ever observed was 0.50, while the
    #: figure was being compared against a 0.65 threshold as though 1.00 were
    #: attainable. Worse, the share punished precision — the same record, on
    #: identical evidence, scored 1.00 when the caller typed "Webutation" and
    #: 0.17 when they typed "Webutation, private investigator, insurance fraud,
    #: OSINT". Counting to a saturation point removes both problems.
    #:
    #: 2.0 follows ``item_corroboration_saturation`` in the scoring module,
    #: which uses the same form and the same value. It is NOT a fitted or
    #: validated number, and must not be read as one:
    #:
    #:   * At 2.0 the subject's own RocketReach page (matching one term,
    #:     "webutation") reaches strength 0.70 and is admitted. At 3.0 it
    #:     reaches 0.63 and is not. The outcome for that record is sensitive to
    #:     a value chosen on precedent, not measured.
    #:   * The supporting evidence is one subject and one corpus of 133
    #:     records, where the change took admissions from 4 to 10 with all 10
    #:     verified as the subject. Precision 1.00 on ten admissions is
    #:     encouraging, not proven.
    #:
    #: Re-measure before trusting it on a second subject.
    anchor_context_saturation: float = 2.0

    #: Terms shorter than this are ignored when comparing context.
    anchor_context_min_term_length: int = 3
    anchor_context_stopwords: frozenset[str] = _CONTEXT_STOPWORDS

    #: Cosine similarity at or above which a face match admits a record on its
    #: own, with no name and no context. Higher is stricter. A face is direct
    #: evidence, so it needs no corroboration from a name; the reverse does not
    #: hold, which is why there is no name-alone equivalent of this.
    anchor_face_similarity: float = 0.55

    #: A name is a label, and labels are shared. On its own it can propose that
    #: a record be compared to the subject; it can never be the evidence that
    #: settles it. Admission therefore needs at least one signal that is not the
    #: name: an overlapping context term, an agreeing locality, or a face at
    #: ``anchor_face_similarity``.
    #:
    #: A link used to satisfy this and no longer does. Practically every real
    #: web record has a link, so the rule admitted nearly everything; but the
    #: deeper problem is that a page existing says nothing about which human it
    #: describes. This is what let a college baseball roster, a real estate
    #: agent, an obituary, a professor and an insider-trading filing sit inside
    #: one candidate as though they were one man.
    anchor_requires_independent_signal: bool = True
    #: Predicates whose presence means the record points at a real page. Kept
    #: for reporting only: a link is no longer evidence of identity.
    link_predicates: frozenset[str] = frozenset(
        {"profile_url", "url", "page_url"}
    )

    #: Locality agreement between the record and the address the caller gave.
    #: Graded, because "shares a token with" is not "is in the same place as":
    #: a state is shared by millions, which is why locality is excluded from
    #: :data:`DISTINCTIVE_PREDICATES` on exactly that reasoning.
    anchor_locality_match: float = 1.00
    #: Only a state or region in common. Weakly confirmatory, so it counts
    #: toward strength, and far too coarse to admit a record on its own.
    anchor_locality_region_only: float = 0.25
    anchor_locality_mismatch: float = 0.00
    #: A locality must reach this to serve as the signal that admits a record.
    #: Being merely non-contradictory is not evidence.
    anchor_locality_admits: float = 0.50

    #: Records in a name-only block still need positive evidence to combine.
    #: Blocking proposes; only evidence disposes. Without this a shared name
    #: quietly becomes a merge, which is the one thing the merge rule forbids.
    name_only_blocks_require_evidence: bool = True


#: Alias for FormationConfig. Both names refer to the same class; the
#: shorter one predates subject anchoring.
CandidateFormationConfig = FormationConfig

#: Given two candidates, return a face distance, or None when no comparable
#: embedding exists for the pair. Injected so this module needs no face model.
FaceDistanceFn = Callable[["CandidateDraft", "CandidateDraft"], float | None]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorMatch:
    """How well one record matched the subject that was searched for.

    Kept per record, including for records that fell below threshold, so a
    report can say why a record joined the anchor candidate and why another
    one did not.
    """

    strength: float
    name_signal: float | None
    context_signal: float | None
    basis: str
    #: Agreement with the address the caller supplied. None means no comparison
    #: was possible, never that the localities differ.
    locality_signal: float | None = None
    #: Cosine similarity to the subject photo. Higher is more alike. None means
    #: no comparison was possible, never that the faces differ.
    face_signal: float | None = None
    #: More than one face in the record's image, so the match may be to
    #: somebody standing next to the subject.
    face_ambiguous: bool = False
    #: Whether the record points at a real page rather than an unlinked block.
    has_link: bool = False
    #: Cleared the threshold *and* the substance requirement.
    admitted: bool = False
    #: Why a record that cleared the threshold was still not admitted.
    rejected_because: str | None = None

    def __repr__(self) -> str:
        state = "admitted" if self.admitted else "rejected"
        return f"<AnchorMatch {self.strength:.3f} {state} {self.basis}>"


@dataclass
class CandidateDraft:
    """A hypothesized person, before any confidence score is assigned."""

    name_key: str | None
    locality_keys: set[str | None] = field(default_factory=set)
    assertions: list[AssertionDraft] = field(default_factory=list)

    #: What actually supports this grouping. Scoring reads it, because a
    #: candidate held together by a shared name is not the same claim as one
    #: held together by a shared email.
    grouping_basis: GroupingBasis = GroupingBasis.unspecified

    #: True for the candidate assembled from records that matched the subject.
    #: Its members were not merged with each other; each matched the anchor
    #: independently, which is a different and stronger claim than agreeing on
    #: a name.
    is_anchor: bool = False
    #: Anchor strength per record, keyed by record reference.
    anchor_matches: dict[str, AnchorMatch] = field(default_factory=dict)

    #: Blocking keys folded into this candidate. More than one means a merge
    #: happened, and ``merge_evidence`` records what justified it.
    blocking_keys: set[tuple[str | None, str | None]] = field(default_factory=set)
    #: Human-readable justification for each merge, for audit. Empty means the
    #: candidate came from a single block and no merge was performed.
    merge_evidence: list[str] = field(default_factory=list)

    @property
    def is_merged(self) -> bool:
        return len(self.blocking_keys) > 1

    def __repr__(self) -> str:
        return (
            f"<CandidateDraft name={self.name_key!r} "
            f"localities={sorted(k or '-' for k in self.locality_keys)} "
            f"assertions={len(self.assertions)} merged={self.is_merged}>"
        )


@dataclass
class CandidateFormation:
    """The outcome of formation.

    An empty ``candidates`` list with a populated ``unattached`` list is a
    successful run in which nothing met threshold. It is not an error, and
    formation never raises to signal it. Callers distinguish success from
    failure by whether the call returned at all.
    """

    candidates: list[CandidateDraft] = field(default_factory=list)
    unattached: list[AssertionDraft] = field(default_factory=list)

    #: Anchor strength for every record evaluated, matched or not. A rejected
    #: record's score is the explanation a reviewer needs.
    anchor_matches: dict[str, AnchorMatch] = field(default_factory=dict)
    #: False when the subject supplied no name and no context, as with a
    #: photo-only query. Anchoring is then skipped entirely rather than
    #: applied with nothing to match against.
    anchor_available: bool = False

    @property
    def anchor_candidate(self) -> CandidateDraft | None:
        """The candidate the search was for, if anchoring produced one."""
        return next((c for c in self.candidates if c.is_anchor), None)

    @property
    def formed_nothing(self) -> bool:
        """True for the valid outcome of zero candidates."""
        return not self.candidates


# --------------------------------------------------------------------------
# Key normalization
# --------------------------------------------------------------------------


def normalize_name_key(value: str | None) -> str | None:
    """Blocking key form of a name: casefolded, depunctuated, suffix-free.

    Possessives and platform tails are removed as well, because a search title
    is a page title: "Michael Petrie's Post" and "Michael Petrie on LinkedIn"
    describe one person, and leaving the tail in invents a second.

    For blocking only. The stored ``normalized_value`` on a draft is left
    untouched, because this transformation is lossy in ways that are fine for
    bucketing and not fine for display or comparison.
    """
    if not value:
        return None
    text = _PUNCTUATION.sub(" ", _POSSESSIVE.sub("", value)).casefold()
    tokens = [t for t in text.split() if t and t not in _NAME_SUFFIXES]

    # Peel platform and content-type tails, one token at a time, stopping
    # before the name itself is eaten.
    while len(tokens) > _MIN_NAME_TOKENS and tokens[-1] in _NAME_TAIL_TOKENS:
        tokens.pop()

    return " ".join(tokens) or None


def normalize_locality_key(value: str | None) -> str | None:
    """Blocking key form of a locality: casefolded, depunctuated, collapsed."""
    if not value:
        return None
    text = _PUNCTUATION.sub(" ", value).casefold()
    return " ".join(text.split()) or None


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

#: Either a real record reference, or a draft's position standing in for one.
_RecordKey = tuple[str, str | int]


def _record_key(draft: AssertionDraft, position: int) -> _RecordKey:
    """Which source record a draft belongs to.

    ``record_ref`` is authoritative: drafts sharing one came from the same
    search result, profile or row, and only those may combine into a blocking
    key. Nothing is inferred from publisher, origin or observation date.

    A null ``record_ref`` makes the draft its own single-claim record rather
    than pooling all nulls together. Pooling would be the dangerous reading:
    unrelated claims from different people would share a blocking key and could
    pair one person's name with another's locality. Standing alone, a draft with
    no name or locality of its own simply fails to block and is reported
    unattached, which is the safe failure.

    No fallback inference is offered on purpose. An adapter that cannot
    delimit records should synthesize a reference from whatever does identify
    them, such as a result URL or a hash of the result block. It has the raw
    response and can do that correctly; this module would only be guessing.
    """
    if draft.record_ref:
        return ("ref", draft.record_ref)
    return ("solo", position)


@dataclass
class _Record:
    """Drafts believed to describe one person, from one source record."""

    key: _RecordKey
    drafts: list[AssertionDraft] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Stable string identifying this record, for anchor reporting."""
        kind, value = self.key
        return str(value) if kind == "ref" else f"solo:{value}"

    @property
    def name_texts(self) -> list[str]:
        """Every rendering of a name this record carries.

        Both the extracted name and the page title it came from, because a
        title often contains the subject's name where the extracted head does
        not, as in "Ada Lovelace | Biography and Facts".
        """
        texts: list[str] = []
        for d in self.drafts:
            if d.predicate != "name":
                continue
            for value in (d.normalized_value, d.raw_value):
                if value and value not in texts:
                    texts.append(value)
        return texts

    @property
    def name_key(self) -> str | None:
        for d in self.drafts:
            if d.predicate == "name":
                key = normalize_name_key(d.normalized_value or d.raw_value)
                if key:
                    return key
        return None

    @property
    def locality_key(self) -> str | None:
        """Most specific locality available, or None."""
        by_predicate = {d.predicate: d for d in self.drafts}
        for predicate in LOCALITY_PREDICATES:
            draft = by_predicate.get(predicate)
            if draft is not None:
                key = normalize_locality_key(draft.normalized_value or draft.raw_value)
                if key:
                    return key
        return None

    @property
    def blocking_key(self) -> tuple[str | None, str | None]:
        return (self.name_key, self.locality_key)

    def distinctive_signals(self) -> set[tuple[str, str]]:
        """Distinctive (label, value) pairs this record carries."""
        signals: set[tuple[str, str]] = set()
        values: dict[str, str] = {}

        for d in self.drafts:
            value = d.normalized_value or d.raw_value
            if not value:
                continue
            if d.predicate in DISTINCTIVE_PREDICATES:
                signals.add((d.predicate, value.casefold()))
            values[d.predicate] = value.casefold()

        for group in COMPOSITE_DISTINCTIVE:
            if all(p in values for p in group):
                combined = "|".join(values[p] for p in group)
                signals.add(("+".join(group), combined))

        return signals


def _group_records(drafts: Iterable[AssertionDraft]) -> list[_Record]:
    """Bucket drafts by their source record, preserving input order."""
    records: dict[_RecordKey, _Record] = {}
    for position, draft in enumerate(drafts):
        key = _record_key(draft, position)
        records.setdefault(key, _Record(key=key)).drafts.append(draft)
    return list(records.values())


# --------------------------------------------------------------------------
# Subject anchoring
# --------------------------------------------------------------------------


def _context_terms(text: str | None, config: FormationConfig) -> set[str]:
    """Comparable terms from free text, with noise words removed."""
    if not text:
        return set()
    tokens = _PUNCTUATION.sub(" ", text).casefold().split()
    return {
        token
        for token in tokens
        if len(token) >= config.anchor_context_min_term_length
        and token not in config.anchor_context_stopwords
    }


def _anchor_name_signal(
    record: _Record, subject_name_key: str | None, config: FormationConfig
) -> float | None:
    """Best name agreement between the subject and any of the record's names.

    None when there is nothing to compare, so an absent name is dropped from
    the mean rather than counted as disagreement.
    """
    if not subject_name_key:
        return None

    subject_tokens = _tokens_of(subject_name_key)
    if not subject_tokens:
        return None

    best: float | None = None
    for text in record.name_texts:
        key = normalize_name_key(text)
        if not key:
            continue
        record_tokens = _tokens_of(key)
        if not record_tokens:
            continue

        if record_tokens == subject_tokens:
            score = config.anchor_name_exact
        elif subject_tokens <= record_tokens:
            score = config.anchor_name_subject_contained
        elif record_tokens <= subject_tokens:
            score = config.anchor_name_record_contained
        else:
            overlap = len(record_tokens & subject_tokens) / len(
                record_tokens | subject_tokens
            )
            score = (
                config.anchor_name_partial
                if overlap >= config.anchor_name_partial_overlap
                else config.anchor_name_none
            )

        if best is None or score > best:
            best = score

    return best


def _anchor_context_signal(
    record_text: str | None, subject_terms: set[str], config: FormationConfig
) -> float | None:
    """How much of the subject's context the record carries, counted not shared.

    Matching terms are counted toward ``anchor_context_saturation`` rather than
    divided by however many terms the caller supplied. Dividing made the scale
    unreachable — no snippet holds six context terms — and made a caller who
    described the subject precisely score lower than one who typed a single
    word, on the same evidence.

    None when the subject gave no context or the record carried no text. A
    record that has text and shares none of it scores zero, which is evidence
    against rather than absence of evidence.

    Any overlap above zero admits the record, which is a real weakness: one
    coincidentally shared common word satisfies the "name plus one other
    signal" rule. Weighting terms by how rare they are across the result set
    was tried and reverted. Document frequency cannot tell a term that is
    everywhere because the search biased the results from a term that is
    everywhere because the records really are all one person, and those need
    opposite treatment: eight pages about Ada Lovelace share their whole
    vocabulary, and dropping them is a worse error than admitting a stranger.
    Fixing this properly needs a rarity estimate from outside the result set.
    """
    if not subject_terms:
        return None
    record_terms = _context_terms(record_text, config)
    if not record_terms:
        return None
    matched = len(subject_terms & record_terms)
    if config.anchor_context_saturation <= 0:
        return 1.0 if matched else 0.0
    return min(1.0, matched / config.anchor_context_saturation)


#: Region names that are two words. Matched as phrases before anything is
#: tokenised, because their halves are ordinary place words: treating a bare
#: "york" as a region turns "New York" into the settlement "New", which then
#: reads as a prefix of "New Orleans".
_REGION_PHRASES: dict[str, str] = {
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "west virginia": "wv",
    "district of columbia": "dc",
    "puerto rico": "pr",
}

#: One-word regions, mapped to the postal code so "tx" and "texas" are the same
#: region rather than two. Some collide with ordinary words — "in" is Indiana,
#: "or" is Oregon — and the collision is deliberately resolved in favour of
#: reading them as regions, because that can only weaken a locality signal.
#: Weakening one costs a rejection; strengthening one costs a false
#: identification, and only one of those is recoverable.
_REGION_CANON: dict[str, str] = {
    **{
        code: code
        for code in """
        al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms
        mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv
        wi wy dc pr
        """.split()
    },
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}

#: Words that say where a building stands rather than which settlement it is
#: in. A street address is a locality predicate, so without this "100 Main St"
#: and "St Louis" agree on the token "st".
_ADDRESS_NOISE_TOKENS: frozenset[str] = frozenset(
    """
    st street ave avenue rd road dr drive blvd boulevard ln lane ct court
    pl place way pkwy parkway hwy highway cir circle ter terrace sq square
    suite ste apt unit floor rm room bldg building po box
    n s e w ne nw se sw north south east west
    """.split()
)


def _split_locality(key: str) -> tuple[set[str], set[str]]:
    """Separate a locality into the regions it names and the settlement.

    Returns ``(regions, place)``. Regions are canonical postal codes. The place
    is what is left once regions, street furniture and bare numbers are gone,
    which is the only part specific enough to identify anywhere.
    """
    regions: set[str] = set()
    place: set[str] = set()

    # A two-word region is also, often, a city: New York, Kansas City's
    # neighbour Kansas, Washington. The phrase is recorded as a region *and*
    # kept as a settlement, so "New York, NY" still matches itself while the
    # region check below keeps it away from "New Orleans, LA".
    for phrase, code in _REGION_PHRASES.items():
        if phrase in key:
            key = key.replace(phrase, " ")
            regions.add(code)
            place.update(phrase.split())

    for token in key.split():
        canonical = _REGION_CANON.get(token)
        if canonical is not None:
            regions.add(canonical)
        elif token not in _ADDRESS_NOISE_TOKENS and not token.isdigit():
            place.add(token)

    return regions, place


def _anchor_locality_signal(
    record: _Record, subject_locality_key: str | None, config: FormationConfig
) -> float | None:
    """Whether the record sits where the caller said the subject does.

    None when either side is silent, because an absent locality is not a
    disagreement.

    Region and settlement are compared separately. Regions that disagree settle
    it: nobody is in Texas and Massachusetts. Settlements are compared by subset
    rather than by overlap, so "Austin" still agrees with "Austin, Texas" and
    "Springfield" with "100 Main St, Springfield", while "New York" no longer
    agrees with "New Orleans" on the word they share.

    Any shared token used to count. That made every same-name person in one
    state a locality match, and under the admission rule that was enough to put
    six unrelated men into one anchor — the original bug, one notch up.

    When one side names no settlement, the most that can be honestly said is
    that the regions agree, which is reported as such and is deliberately below
    ``anchor_locality_admits``. Millions of people share a state.
    """
    if not subject_locality_key:
        return None
    record_key = record.locality_key
    if not record_key:
        return None

    subject_regions, subject_place = _split_locality(subject_locality_key)
    record_regions, record_place = _split_locality(record_key)

    if subject_regions and record_regions and not (subject_regions & record_regions):
        return config.anchor_locality_mismatch

    if not subject_place or not record_place:
        return (
            config.anchor_locality_region_only
            if subject_regions & record_regions
            else config.anchor_locality_mismatch
        )

    if subject_place <= record_place or record_place <= subject_place:
        return config.anchor_locality_match
    return config.anchor_locality_mismatch


def _anchor_match(
    record: _Record,
    subject_name_key: str | None,
    subject_terms: set[str],
    record_context: Mapping[str, str],
    config: FormationConfig,
    face: object | None = None,
    subject_locality_key: str | None = None,
) -> AnchorMatch:
    """How strongly one record matches the subject that was searched for.

    ``face`` is a FaceComparison or None. None means no comparison was
    possible, so the face signal is dropped from the mean rather than scored
    zero: a record with no photo has said nothing about whether it is the
    subject.

    Note what dropping an unavailable signal does to the mean. With nothing but
    a name to compare, the mean of one signal *is* the name, so a common name
    scores a perfect 1.00 and looks like certainty. The strength alone therefore
    cannot decide admission; see the independent-signal rule in the anchor pass.
    """
    name = _anchor_name_signal(record, subject_name_key, config)
    context = _anchor_context_signal(
        record_context.get(record.label), subject_terms, config
    )
    locality = _anchor_locality_signal(record, subject_locality_key, config)
    face_signal = getattr(face, "best_similarity", None) if face is not None else None
    face_ambiguous = bool(getattr(face, "ambiguous", False)) if face else False

    signals: list[tuple[float, float]] = []
    parts: list[str] = []
    if face_signal is not None:
        signals.append((max(0.0, face_signal), config.weight_anchor_face))
        parts.append(
            f"face {face_signal:.2f}" + (" (ambiguous)" if face_ambiguous else "")
        )
    if name is not None:
        signals.append((name, config.weight_anchor_name))
        parts.append(f"name {name:.2f}")
    if context is not None:
        signals.append((context, config.weight_anchor_context))
        parts.append(f"context {context:.2f}")
    if locality is not None:
        signals.append((locality, config.weight_anchor_locality))
        parts.append(f"locality {locality:.2f}")

    strength = _weighted_mean(signals) if signals else 0.0
    basis = ", ".join(parts) if parts else "no comparable signal"

    has_link = any(
        draft.predicate in config.link_predicates
        and (draft.normalized_value or draft.raw_value)
        for draft in record.drafts
    )
    return AnchorMatch(
        strength=strength,
        name_signal=name,
        context_signal=context,
        locality_signal=locality,
        basis=basis,
        has_link=has_link,
        face_signal=face_signal,
        face_ambiguous=face_ambiguous,
    )


def _weighted_mean(signals: Iterable[tuple[float, float]]) -> float:
    pairs = list(signals)
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total


def _tokens_of(key: str | None) -> set[str]:
    return set(key.split()) if key else set()


# --------------------------------------------------------------------------
# Blocking and merging
# --------------------------------------------------------------------------


@dataclass
class _Block:
    """Records sharing one blocking key. A proposal, not a decision."""

    key: tuple[str | None, str | None]
    records: list[_Record] = field(default_factory=list)

    @property
    def name_key(self) -> str | None:
        return self.key[0]

    @property
    def locality_key(self) -> str | None:
        return self.key[1]

    def distinctive_signals(self) -> set[tuple[str, str]]:
        signals: set[tuple[str, str]] = set()
        for record in self.records:
            signals |= record.distinctive_signals()
        return signals


def _build_blocks(records: Iterable[_Record]) -> list[_Block]:
    blocks: dict[tuple[str | None, str | None], _Block] = {}
    for record in records:
        key = record.blocking_key
        blocks.setdefault(key, _Block(key=key)).records.append(record)
    return list(blocks.values())


def _evidence_components(
    records: Sequence[_Record], config: FormationConfig
) -> list[list[_Record]]:
    """Split records into groups actually joined by positive evidence.

    Two records belong together only if they share a distinctive attribute.
    Everything else stands alone. Face distance is not consulted here because
    it compares candidates, not records, and applies at the merge step.
    """
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    signals = [r.distinctive_signals() for r in records]
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if signals[i] & signals[j]:
                parent[find(i)] = find(j)

    grouped: dict[int, list[_Record]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def _split_unsupported_blocks(
    blocks: Iterable[_Block], config: FormationConfig
) -> list[_Block]:
    """A block is a proposal, not a candidate.

    A block keyed on a name alone says only that several pages used the same
    words. Left whole it becomes one person, which is exactly the merge that
    the "a shared name is never positive evidence" rule forbids, arrived at by
    skipping the merge step rather than passing it.

    Blocks that also agree on locality are left intact: name plus locality is
    more than a name, and splitting them would fragment the clean case.
    """
    result: list[_Block] = []
    for block in blocks:
        needs_evidence = (
            config.name_only_blocks_require_evidence
            and block.name_key is not None
            and block.locality_key is None
            and len(block.records) > 1
        )
        if not needs_evidence:
            result.append(block)
            continue
        for component in _evidence_components(block.records, config):
            result.append(_Block(key=block.key, records=component))
    return result


def _shared_distinctive(
    left: set[tuple[str, str]], right: set[tuple[str, str]]
) -> tuple[str, str] | None:
    """One shared distinctive signal, or None. Name is never in this set."""
    shared = sorted(left & right)
    return shared[0] if shared else None


def _to_candidate(blocks: Iterable[_Block], evidence: list[str]) -> CandidateDraft:
    """Build a candidate from blocks, carrying their drafts along.

    Drafts are included so an injected face callable receives something it can
    actually work with. Real attachment happens later and rebuilds this list.
    """
    blocks = list(blocks)
    names = {b.name_key for b in blocks if b.name_key}
    return CandidateDraft(
        grouping_basis=_basis_of(blocks, evidence),
        name_key=sorted(names)[0] if names else None,
        locality_keys={b.locality_key for b in blocks},
        blocking_keys={b.key for b in blocks},
        merge_evidence=evidence,
        assertions=[d for b in blocks for r in b.records for d in r.drafts],
    )


def _basis_of(blocks: Sequence[_Block], evidence: Sequence[str]) -> GroupingBasis:
    """What this candidate's grouping actually rests on.

    Read in descending order of strength, because a candidate held together by
    a shared email is a different claim from one held together by a name.
    """
    for reason in evidence:
        if "distinctive attribute" in reason:
            return GroupingBasis.distinctive_attribute
    for reason in evidence:
        if "face distance" in reason:
            return GroupingBasis.face

    # Records combined *within* a block leave no merge reason behind, so the
    # evidence that held them together has to be read off the records.
    records = [r for b in blocks for r in b.records]
    if len(records) > 1:
        signals = [r.distinctive_signals() for r in records]
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                if signals[i] & signals[j]:
                    return GroupingBasis.distinctive_attribute

    if any(b.locality_key for b in blocks):
        return GroupingBasis.name_and_locality
    if any(b.name_key for b in blocks):
        return GroupingBasis.name_only
    return GroupingBasis.unspecified


def _distinctive_of(drafts: Iterable[AssertionDraft]) -> set[tuple[str, str]]:
    """Distinctive signals carried by a flat list of drafts."""
    signals: set[tuple[str, str]] = set()
    for draft in drafts:
        value = draft.normalized_value or draft.raw_value
        if value and draft.predicate in DISTINCTIVE_PREDICATES:
            signals.add((draft.predicate, value.casefold()))
    return signals


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def form_candidates(
    drafts: list[AssertionDraft],
    config: FormationConfig | None = None,
    face_distance: FaceDistanceFn | None = None,
    subject: Query | None = None,
    record_context: Mapping[str, str] | None = None,
    face_match: Callable[[Sequence[AssertionDraft]], object | None] | None = None,
) -> CandidateFormation:
    """Group drafts into candidates and attach what clears threshold.

    ``face_distance`` is optional. When it is not supplied the face signal is
    simply unavailable, and merging depends entirely on shared distinctive
    attributes. Nothing degrades to merging on name.

    Subject anchoring
    -----------------
    The subject is a hypothesis: a name and a context were supplied, and
    results should be judged against them rather than only against each other.
    Without that, eight results about one person become eight candidates.

    When ``subject`` carries a name or a context, each record is scored against
    it first. Records above ``anchor_threshold`` *and* carrying a signal that is
    not the name form one anchor candidate. Crucially they are not merged with
    one another: each matched the subject independently, which is a stronger
    claim than agreeing on a name and is why this does not reintroduce
    merge-by-name.

    That second requirement is the whole guard. Strength is a mean over the
    signals that could be computed, so a record whose only comparable signal is
    its name scores exactly its name similarity: an exact match on a common name
    reads 1.00. Admitting on strength alone put five unrelated men — a college
    baseball roster, a real estate agent, an obituary, a professor and an
    insider-trading filing — inside one candidate at p = 1.00. A name may
    propose a comparison. It may never settle one.

    The consequence is deliberate: a search on a name alone, with no address, no
    context and no photo, produces no anchor at all. There is nothing to verify
    it against, and an empty anchor states that honestly where a confident one
    would not.

    Records below threshold fall through to blind clustering unchanged, so a
    different person sharing the name still forms a separate candidate and
    stays visible. ``record_context`` maps a record reference to the free text
    that record carried, normally a search snippet.

    Anchoring is skipped entirely when the subject supplies neither name nor
    context. A photo-only query has no text to anchor against, and inventing
    one would be worse than not having it.


    Missing locality, the rule
    --------------------------
    Most real search results carry no locality, so this case dominates and the
    behaviour is stated rather than left to fall out of the code.

    A record with a name but no locality forms a *provisional* block. It is
    never silently folded into a localized block sharing its name, and it never
    automatically becomes a candidate. It resolves one of three ways:

    * **No localized block shares its name.** It becomes a candidate on its own.
      There is no locality ambiguity to be wrong about.
    * **Exactly one localized block shares its name.** It merges only on
      positive evidence, the same bar as any other merge. Without evidence its
      drafts are reported unattached, because a name in common is not evidence.
    * **Two or more localized blocks share its name.** Its drafts are reported
      unattached. Choosing between equally plausible candidates would be a
      guess, and guessing here is what produces over-merged people.

    A record with neither name nor locality cannot be blocked at all. Its
    drafts attach only if they share a distinctive attribute with a candidate,
    and are otherwise unattached.
    """
    config = config or FormationConfig()
    formation = CandidateFormation()
    record_context = record_context or {}

    if not drafts:
        return formation

    records = _group_records(drafts)

    # --- anchor pass, before any blocking --------------------------------
    subject_name_key = normalize_name_key(getattr(subject, "name", None))
    subject_terms = _context_terms(getattr(subject, "context", None), config)
    # Address only. Context is already compared as context, and counting it
    # twice would let one weak signal admit a record by pretending to be two.
    subject_locality_key = normalize_locality_key(getattr(subject, "address", None))
    # A photo is an anchor too. Without one, a subject with neither name nor
    # context has nothing to match against and anchoring is skipped entirely.
    formation.anchor_available = bool(
        subject_name_key or subject_terms or face_match is not None
    )

    anchored: list[_Record] = []
    if formation.anchor_available:
        for record in records:
            face = face_match(record.drafts) if face_match is not None else None
            match = _anchor_match(
                record,
                subject_name_key,
                subject_terms,
                record_context,
                config,
                face=face,
                subject_locality_key=subject_locality_key,
            )

            # A face above threshold is direct evidence about the person, so it
            # admits the record by itself: no name, no context, no link needed.
            # Nothing else in this system earns that, because nothing else is
            # evidence about the human rather than about a label.
            #
            # An ambiguous face — more than one in the picture — still admits,
            # and is flagged rather than hidden. Refusing it would drop the
            # record from the report altogether, which tells the reader less
            # than showing it with the ambiguity stated. That is a deliberate
            # prior decision; see test_an_ambiguous_face_match_is_flagged.
            face_admits = (
                match.face_signal is not None
                and match.face_signal >= config.anchor_face_similarity
            )

            if not face_admits and match.strength <= config.anchor_threshold:
                formation.anchor_matches[record.label] = match
                continue

            # A locality that disagrees is not a weak signal, it is a positive
            # statement that this is somewhere else. Diluting it in the mean
            # lets a strong context term carry a record that says Boston into an
            # anchor for a subject in Austin.
            contradicts_locality = (
                match.locality_signal is not None
                and match.locality_signal <= config.anchor_locality_mismatch
            )

            # A record has to carry the subject's name to be admitted on text.
            #
            # This used to apply only when the caller supplied a name, which
            # left context free to admit on its own when they did not. The old
            # share-of-terms form hid that: two matched terms out of six scored
            # 0.33 and never cleared the threshold. Counting to a saturation
            # point removed the accidental cover and the hole showed itself
            # immediately — "private investigator" and "insurance fraud" are two
            # terms each, so Wikipedia's article on private investigators, a
            # Reddit thread and a press release about an unrelated man all
            # reached 1.00 and joined the anchor.
            #
            # The rule the rest of this module already asserts settles it: a
            # name is a label many people share and cannot admit alone. A trade
            # is a label far more people share. So text admission needs the
            # name, and a subject who supplied none cannot be identified from
            # their profession — only a face can anchor them.
            missing_name = not match.name_signal

            # Substance check. Strength alone cannot admit, because a record
            # whose only comparable signal is its name scores the name: an exact
            # match on a common name reads 1.00 and means nothing. Admission
            # needs at least one signal that is not the name, and a locality
            # only counts once it names a place rather than a state.
            independent = [
                signal is not None and signal > 0
                for signal in (match.context_signal,)
            ] + [
                match.locality_signal is not None
                and match.locality_signal >= config.anchor_locality_admits
            ]
            thin = config.anchor_requires_independent_signal and not any(independent)

            if face_admits:
                reason = None
            elif contradicts_locality:
                reason = "locality contradicts the subject"
            elif missing_name:
                reason = (
                    "does not carry the subject's name"
                    if subject_name_key
                    else "no name to match, and a trade is not an identity"
                )
            elif thin:
                reason = "name only — no other signal"
            else:
                reason = None

            match = replace(
                match, admitted=reason is None, rejected_because=reason
            )
            formation.anchor_matches[record.label] = match
            if match.admitted:
                anchored.append(record)

    anchor_candidate: CandidateDraft | None = None
    if anchored:
        anchor_candidate = CandidateDraft(
            name_key=subject_name_key or anchored[0].name_key,
            locality_keys={r.locality_key for r in anchored},
            blocking_keys={r.blocking_key for r in anchored},
            grouping_basis=GroupingBasis.anchor,
            is_anchor=True,
            anchor_matches={
                r.label: formation.anchor_matches[r.label] for r in anchored
            },
        )
        # Everything else is clustered blindly, exactly as before.
        anchored_keys = {r.key for r in anchored}
        records = [r for r in records if r.key not in anchored_keys]

    blocks = _split_unsupported_blocks(_build_blocks(records), config)

    # Blocks with no name cannot be blocked at all. They are not collected
    # into a partition here; their records simply reach attachment unassigned.
    localized = [b for b in blocks if b.name_key and b.locality_key]
    provisional = [b for b in blocks if b.name_key and not b.locality_key]

    # --- merge localized blocks on positive evidence only -----------------
    groups: list[tuple[list[_Block], list[str]]] = [([b], []) for b in localized]

    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                left_blocks, left_ev = groups[i]
                right_blocks, right_ev = groups[j]

                reason = _merge_reason(
                    left_blocks, right_blocks, config, face_distance
                )
                if reason is None:
                    continue

                groups[i] = (left_blocks + right_blocks, left_ev + right_ev + [reason])
                del groups[j]
                merged = True
                break
            if merged:
                break

    candidates: list[CandidateDraft] = [_to_candidate(b, ev) for b, ev in groups]

    # Keyed by record, not by blocking key. Splitting a name-only block leaves
    # several blocks carrying the same key, so the key no longer identifies a
    # candidate.
    assigned_records: dict[_RecordKey, CandidateDraft] = {}
    blocks_of: dict[int, list[_Block]] = {}
    for (block_list, _), candidate in zip(groups, candidates):
        blocks_of[id(candidate)] = list(block_list)
        for block in block_list:
            for record in block.records:
                assigned_records[record.key] = candidate

    # --- resolve provisional blocks per the documented rule ---------------
    for block in provisional:
        # Only localized candidates are consulted. The rule exists to stop a
        # locality-free record being absorbed into a *located* person on a
        # shared name; two locality-free blocks are not rivals for that, they
        # are simply two candidates.
        same_name = [
            c
            for c in candidates
            if c.name_key == block.name_key and any(k for k in c.locality_keys)
        ]

        if not same_name:
            candidate = _to_candidate([block], [])
            candidates.append(candidate)
            blocks_of[id(candidate)] = [block]
            for record in block.records:
                assigned_records[record.key] = candidate
            continue

        if len(same_name) == 1:
            target = same_name[0]
            target_blocks = blocks_of.get(id(target), [])
            reason = _merge_reason(
                [block], target_blocks, config, face_distance
            )
            if reason is not None:
                target.blocking_keys.add(block.key)
                target.locality_keys.add(block.locality_key)
                target.merge_evidence.append(reason)
                blocks_of.setdefault(id(target), []).append(block)
                for record in block.records:
                    assigned_records[record.key] = target
            # No evidence: left unassigned, its drafts fall through to
            # attachment scoring and land in unattached.
            continue

        # Two or more same-name candidates: choosing would be a guess.

    # --- attachment, in two passes ---------------------------------------
    # Block-assigned records first, so that a candidate's distinctive signals
    # are known before any leftover record is compared against it. Otherwise
    # the result would depend on the order drafts happened to arrive in.
    if anchor_candidate is not None:
        candidates.insert(0, anchor_candidate)

    for candidate in candidates:
        candidate.assertions = []

    if anchor_candidate is not None:
        for record in anchored:
            anchor_candidate.assertions.extend(record.drafts)

    leftovers: list[_Record] = []
    for record in records:
        candidate = assigned_records.get(record.key)
        if candidate is not None:
            candidate.assertions.extend(record.drafts)
        else:
            leftovers.append(record)

    for record in leftovers:
        best, strength = _best_match(record, candidates, config)
        if best is not None and strength > config.attachment_threshold:
            best.assertions.extend(record.drafts)
        else:
            formation.unattached.extend(record.drafts)

    # A candidate holding nothing is not a candidate.
    formation.candidates = [c for c in candidates if c.assertions]
    return formation


def _merge_reason(
    left: list[_Block],
    right: list[_Block],
    config: FormationConfig,
    face_distance: FaceDistanceFn | None,
) -> str | None:
    """Why these two block groups may merge, or None if they may not.

    Shared name is not consulted. If neither signal is present the answer is
    None, and the blocks stay separate candidates.
    """
    left_signals: set[tuple[str, str]] = set()
    for b in left:
        left_signals |= b.distinctive_signals()
    right_signals: set[tuple[str, str]] = set()
    for b in right:
        right_signals |= b.distinctive_signals()

    shared = _shared_distinctive(left_signals, right_signals)
    if shared is not None:
        label, value = shared
        return f"shared distinctive attribute {label}={value!r}"

    if face_distance is not None:
        distance = face_distance(_to_candidate(left, []), _to_candidate(right, []))
        if distance is not None and distance < config.face_distance_threshold:
            # Both numbers, because the report speaks similarity and the rule
            # is written in distance.
            return (
                f"matched on face, similarity {1.0 - distance:.3f} "
                f"(face distance {distance:.3f} below "
                f"{config.face_distance_threshold})"
            )

    return None


def _best_match(
    record: _Record, candidates: list[CandidateDraft], config: FormationConfig
) -> tuple[CandidateDraft | None, float]:
    """Strongest candidate for a record that no block assigned, and how strong.

    Structural agreement only. This is not the identity confidence score p,
    which scoring assigns later from richer inputs.
    """
    best: CandidateDraft | None = None
    best_strength = 0.0
    record_signals = record.distinctive_signals()

    for candidate in candidates:
        if record.name_key and candidate.name_key == record.name_key:
            if record.locality_key is None:
                strength = config.name_match_locality_absent
            elif record.locality_key in candidate.locality_keys:
                strength = config.name_and_locality_match
            else:
                strength = config.name_match_locality_conflict
        else:
            strength = config.no_name_match

        candidate_signals = _distinctive_of(candidate.assertions)
        if _shared_distinctive(record_signals, candidate_signals) is not None:
            strength = min(1.0, strength + config.distinctive_attribute_bonus)

        if strength > best_strength:
            best, best_strength = candidate, strength

    return best, best_strength
