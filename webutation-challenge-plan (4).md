# Webutation Challenge — Build Plan

**Author:** Zein Khamis
**Context:** Technical evaluation for Full Stack Engineer role at Webutation, Inc.
**Reviewers:** Michael Petrie (CEO), Max Terzi (CTO), Kirill (Lead Dev)

---

## 1. What is being tested

The brief asks for two variants:

1. **Image → identity.** Accept a photo, find the person online, return everything with confidence per item and per identity match.
2. **Context → identity.** Accept name / address / other context, find all digital footprints.

These are not two systems. They are two entry points into one identity resolution engine.

The real evaluation is **not** whether face search works. Face search is a licensed API call. The evaluation is whether the candidate understands that:

- Identity match confidence and item confidence are **different numbers**.
- Every fact needs provenance: which source, when collected, what raw artifact.
- Conflicts between sources must be surfaced, not silently resolved.
- A score does not promote a result to a finding — a human review status does.

That distinction is the entire product. Webutation's reports go to court. A wrong identity match is the failure mode that costs their client.

### But rigor alone loses the room

Mike is a licensed PI of 30 years, not an academic. He asked to be **found**. If the demo returns 2 candidates and 6 facts about him while a reviewer opens a face-search site and gets 30 hits in 4 seconds, the confidence model reads as an excuse for weak recall.

**Recall earns the meeting. The confidence model keeps it.** Both are required, and recall is the one this plan is naturally weak on, so it gets explicit budget.

Target for the demo subject: **20+ distinct assertions across 6+ distinct origins**, including the Social Detection acquisition, the Webutation founding, his licensing, press mentions, and social profiles. He knows exactly what is findable about him. Missing the obvious is the one failure he will notice immediately.

**Demo narrative order:** lead with what was found, then reveal the machinery underneath. Never open with the scoring model.

1. Here is everything we found about you (dense, fast, visibly comprehensive)
2. Here is how sure we are of each piece, and why
3. Here is what we found and *rejected*, and why — this is where the rigor lands hardest
4. Here is what broke, and how the system reported it

Item 3 is the single most persuasive screen in the demo. Showing a plausible wrong-person match that the system caught and demoted proves the whole thesis better than any correct result does.

---

## 2. Architecture

```
                    ┌──────────────┐         ┌─────────────────┐
  photo ──────────► │ Image Adapter│────┐    │                 │
                    └──────────────┘    │    │  Candidate Pool │
                                        ├───►│  (identities +  │
                    ┌──────────────┐    │    │   evidence)     │
  name/address ───► │ Query Planner│────┘    │                 │
                    └──────────────┘         └────────┬────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │ Resolution Engine │
                                            │  - clustering     │
                                            │  - scoring        │
                                            │  - conflicts      │
                                            └─────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  Structured Report│
                                            │  + confidence     │
                                            │  + provenance     │
                                            └───────────────────┘
```

### Layers

**Collectors** — one adapter per source. Each returns raw payload + normalized claims. Never call a source directly from business logic.

**Evidence store** — raw collection artifacts persisted with SHA-256 + `fetched_at` before any parsing. Re-parseable later without re-collection. Integrity, not admissibility: see the preservation note in section 3.

**Candidate formation** — the hardest part, and the one most often left implicit. Explicit expected behaviour:

- Assertions are grouped into candidates by blocking key (normalized name + locality), then merged only on positive evidence (face distance below threshold, shared distinctive attribute).
- Two people sharing a name and differing on locality **stay two candidates**. Never merge on name alone.
- An assertion that matches no candidate above threshold attaches to none and is reported as unattached, not forced onto the nearest.
- **Zero candidates is a valid successful outcome.** The run completes and reports what was searched and why nothing met threshold.
- A conflict may indicate an over-merged candidate, not a wrong fact — allow `possible_bad_merge` as a conflict reason.

**Scoring** — scores each candidate (p) and each attached assertion (q); derives r.

**API + UI** — job submission, polling, results view.

---

## 3. Data model

```
Subject           the search request (photo hash, or name/address/context)
  ├── Candidate   a hypothesized person (0..n per subject)
  │     ├── identity_confidence  p   "is this the same human?"
  │     ├── review_status            unreviewed | needs_review | reviewed | rejected
  │     └── Assertion[]              facts attached to this candidate
  │           ├── predicate, raw_value, normalized_value
  │           ├── item_confidence q  "given the match, is this true and current?"
  │           ├── attribution     r = p × q (derived)
  │           ├── source_origin       known_origin | suspected_copy | unknown
  │           ├── observed_at / valid_from / valid_to / fetched_at
  │           ├── Provenance[]        publisher, access_category, artifact_ref
  │           └── conflict_ids[]
  ├── Assertion[] unattached — matched no candidate above threshold
  └── SourceRun[] per source: searched | found N | empty | failed(reason)
```

### Two estimated scores, one derived

**identity_confidence (p)** — is this candidate the same human as the input?

Inputs:
- face embedding distance (own InsightFace cosine, not just the provider's)
- name match strength (exact / fuzzy / alias)
- location overlap with supplied context
- corroboration across independent **origins** (see below)
- negative signals (contradictory age, location, employer)

**item_confidence (q)** — *given this candidate is the right person*, is this claim true and current?

q is **conditional**, not marginal. It must include the possibility that a source record with a matching name concerns a different person.

Inputs:
- source reliability for **this claim** (direct observation / secondhand / republished). Access category is tracked separately and is not a reliability input.
- recency decay, field-specific (contact data decays fast; birth date does not)
- independent-origin corroboration count
- record-attachment uncertainty

**attribution_confidence (r) = p × q** — derived, never estimated directly.

Do **not** cap q at p. `min(p, q)` destroys signal: with p=0.40, it returns 0.40 whether q is 0.95 or 0.55, whereas p×q gives 0.38 vs 0.22.

Multiplication mirrors the probability chain rule, but with hand-tuned inputs it is a **heuristic combination rule**, not a probability calculation. It is also lossy on its own — 0.4×0.9 and 0.9×0.4 both give 0.36 — which is why p and q are always displayed alongside r, never replaced by it.

**Call these confidence scores, not probabilities.** They are uncalibrated heuristics — there is no labeled outcome set to calibrate against. State this explicitly in the README.

### Corroboration counts origins, not publishers

Five sites repeating one aggregator is **one** confirmation. `source_origin` lives on each **assertion**, not just the adapter — one search call returns many publishers, and one publisher mixes original observations with purchased data. Adapter-level default, assertion-level override, explicit `unknown` otherwise.

Lineage states: `known_origin` / `suspected_copy` / `unknown`.
- Confirmed copy → contributes zero additional corroboration
- Suspected copy → retained, but no full independence bonus
- Unknown → surfaced as unknown, never silently counted as independent

Licensing is an access right, not a reliability signal. Reliability depends on the source's relationship to the specific claim.

**UI requirement:** display `5 publishers · 1 known origin · 2 unresolved`, never `5 confirmations`.

Out of scope for v1: inferring copying from shared rare errors (needs a corpus). Mark lineage unknown and be honest about it.

### No double-counting of evidence

Evidence that helped establish the identity match may also inform confidence in a claim. It may **not** earn a second independent-corroboration bonus for passing through a second calculation. Track evidence IDs through both calculations; discount, don't exclude.

### Three separate dates per claim

- `observed_at` — when the source recorded it (nullable)
- `valid_from` / `valid_to` — the period the fact applies to (nullable, usually unknown)
- `fetched_at` — when we collected it

Observation time does not establish validity. Recency decay uses `observed_at`; when it is unknown, freshness is **unknown**, never derived from `fetched_at`. A mirror fetched today must not refresh a five-year-old claim.

### Empty is not failed

A source that returned nothing and a source that errored are different outcomes. The report lists, per source: searched / found N / returned empty / failed (with reason). "Find all footprints" has no completion criterion, so the report states what was searched and what remains unknown.

### Conflicts are stored, not resolved away

Never overwrite a `person.address` field. Store every source assertion with its raw value, normalized value, provenance, dates and score components. A "preferred" value is a derived view with an auditable decision rule, and the conflicting assertions stay queryable.

First check whether values actually conflict:
- different addresses over different periods → historical change, not conflict
- two employers, simultaneous employment possible → compatible
- incompatible values for an exclusive predicate, overlapping validity → **conflict**
- one source silent where another supplies a value → missing evidence, not conflict

Define exclusivity per predicate for the two or three fields where it matters (birth date, SSN-class identifiers). **Employment is multivalued** — do not mark current employer exclusive. Don't build a general exclusivity system in three days.

### Review status, not a threshold

An uncalibrated score crossing 0.7 does not make something a finding. Use the score as a UI hint for sorting, and a separate human-set status.

Contract: status attaches to a **candidate** and to an **assertion** (not the report as a whole). Each status change records `changed_by`, `changed_at`, `reason`. Values: `unreviewed` / `needs_review` / `reviewed` / `rejected`.

`reviewed` means a person examined it. It does **not** mean every claim was accepted as true — acceptance is per-assertion.

```json
{
  "conflict_id": "conflict_17",
  "candidate_id": "candidate_4",
  "predicate": "birth_date",
  "claim_ids": ["claim_12", "claim_29"],
  "reason": "incompatible_normalized_values",
  "status": "unresolved",
  "preferred_claim_id": null
}
```

A conflict may also indicate **bad record attachment or an over-merged candidate**, not just a wrong fact. Allow that as a conflict reason, or the system will preserve a mistaken merge and mark all its facts uncertain instead of questioning the merge.

### Access category field

Every source carries an `access_category` enum. It is descriptive metadata; recording it does not enforce anything:

```
PUBLIC_WEB | LICENSED | RESTRICTED
```

Do **not** add an eligibility-decision flag. FCRA compliance for employment or insurance eligibility involves permissible purpose, notices and adverse-action procedures that a schema field cannot establish. Leave eligibility use unassessed in the prototype and say so in the README. The point is that access characteristics are recorded as data, not that the system adjudicates permitted use.

### On evidence preservation — be precise in the README

Saving a search API response preserves *that response*, not the original post it references. Hash + timestamp supports integrity checking; it does not establish authorship, truth, or admissibility (FRE 901–902 authentication requirements still apply). Describe the evidence store as "raw collection artifacts with integrity hashes", not as chain of custody.

---

## 4. Sources

### Face variant

**Do not build a face index.** Building one requires scraping billions of faces. That is the Clearview AI model — banned in multiple jurisdictions, fined under GDPR, and exposed under Illinois BIPA and, separately, Texas Business & Commerce Code Ch. 503 (distinct statutes, not one law). Federate instead.

**No-approval strategy (fits a 3-day build):** all reverse-image collection goes through SerpAPI or SearchApi.io — instant signup, free credits, one key, programmatic access to Google Lens, Yandex reverse image, and Google/Bing search. Yandex is unusually strong on faces. PimEyes activates instantly with a card if a dedicated face-search provider is wanted for the demo; treat it as optional, not blocking.

Local pipeline (optional, for demonstrating depth):
- InsightFace / ArcFace embeddings
- pgvector cosine search
- Used to **re-verify** provider results against the input photo, producing your own distance score rather than trusting theirs

That re-verification step is a strong differentiator. You are not just proxying a provider, you are independently scoring their output.

### Context variant

Waterfall, cheapest first:

1. Search engines with structured operators
2. Public records / registries
3. Social platform public endpoints
4. Aggregators (PDL, PIPL-class) — **last**, because paid and stale

Cache aggressively. Never hit a paid source for something already resolved.

---

## 5. Build order — 3 days, fixtures first

**Stack:** FastAPI + Postgres/pgvector + React (Vite). InsightFace is Python, so one language end to end; no .NET↔ML bridge.

**Fixtures come before any real source.** A working end-to-end system on synthetic data by end of day 1 means no external API can block you, and the architectural claims become reviewable.

**Fixtures are synthetic *source records*, not prepared reports.** They enter through the same adapter interface and run the full path: normalization → candidate formation → attachment → scoring → conflict detection. Canned reports would demo the UI and validate nothing.

Each fixture asserts observable outcomes: candidate count, which assertions attach where, which conflicts appear.

| # | Fixture | Expected outcome |
|---|---|---|
| 1 | Same name, two cities | 2 candidates, no merge; assertions split by locality |
| 2 | 3 publishers, 1 origin | 1 candidate; corroboration counts 1, UI shows "3 publishers · 1 origin" |
| 3 | Old `observed_at`, fresh `fetched_at` | q reflects staleness; freshness does not reset |
| 4 | Incompatible birth dates | Both assertions retained; 1 conflict object, status `unresolved` |
| 5 | Adapter raises | Source reported `failed` with reason, distinct from `empty` |
| 6 | Clean match | 1 candidate, high p, no conflicts |
| 7 | **No suitable candidate** | 0 candidates, run completes **successfully**, report explains why |

**Day 1 — One fixture, all the way through, deployed**
- Repo, Docker, Postgres + pgvector, schema migrations
- Assertion + provenance + conflict storage
- Adapter interface: `collect(query) -> RawResponse`, `normalize(RawResponse) -> Assertion[]`
- **Fixture 6 (happy path) only**, end to end: normalize → candidates → attach → score → report
- Minimal API: submit, poll, fetch report
- **Deploy the skeleton to Azure today** — environment problems surface now, not on day 3
- Then add fixtures 1, 2, 5 if time remains
- **Milestone:** one fixture correct, running in the cloud

**Day 2 — Real sources, breadth first**
- Fixtures 3, 4, 7 and their assertions (morning, timeboxed — do not let the test suite eat the day)
- **SerpAPI adapters: Google Lens, Yandex reverse image, Google web search.** All three, because recall is the product. They share one client and one response shape, so the marginal cost of the 2nd and 3rd is small
- Per-assertion origin tagging on real results (default `unknown` where lineage is unclear)
- InsightFace embedding + cosine verification against the input photo
- **Run on Mike's photo and iterate on recall until 20+ assertions / 6+ origins.** If the number is low, add query expansion: name + company, name + "Social Detection", name + city, reverse-image on every returned profile photo
- Cache the result with its collection timestamp
- **Milestone:** the demo subject produces a dense, scored report — plus at least one wrong-person candidate that the system correctly demotes

**Day 3 — UI + delivery**
- Submit view: photo drop, or name/address/context form
- Results view, in the demo narrative order above: findings first and dense, then p / q / r per assertion, origin counts, conflicts, rejected candidates, per-source status
- **Rejected candidates must be visible, not hidden.** This is the screen that proves the thesis
- Fixture picker in the UI so reviewers can trigger each edge case themselves
- Deploy to Azure; README with tradeoffs and stated assumptions
- 3-minute recorded walkthrough. Cached replay **labeled as replay** with original collection time.

**If day 3 runs short, cut in this order:** fixture picker → conflict UI polish → per-source status panel. Never cut the findings density or the rejected-candidate view.

**Cut without apology (state in README):** multiple face providers, per-job cost accounting, copy detection via shared errors, general exclusivity modeling, queue-based orchestration, eligibility-use assessment.

**Pre-ship checks (write these as tests):**
1. Adding a mirror source produces no corroboration gain
2. Re-fetching an old assertion produces no freshness gain
3. Ambiguous identity preserves differences in conditional claim strength
4. Selecting a preferred value leaves every conflicting assertion visible in API and UI
5. A failed source and an empty source render differently

---

## 6. What separates a pass from a hire

*(For Zein. Does not go in the README.)*

| Ordinary submission | Strong submission |
|---|---|
| One confidence number | p and q estimated separately, r = p×q derived |
| `min(p,q)` capping | p and q preserved, r derived from both |
| "5 confirmations" | "5 publishers · 1 known origin · 2 unresolved" |
| Scores labeled as percentages | Labeled uncalibrated heuristics, with a reason |
| One `fetched_at` timestamp | observed / valid / fetched kept separate |
| Results dumped as a list | Conflicts stored, surfaced and explained |
| Provider results trusted | Provider results independently re-scored |
| Legal risk mentioned in README | Access category recorded per source |
| Works on the demo photo | Fails gracefully, shows why it failed |
| Scraper script | Pluggable adapters |
| Thin results, heavy theory | Dense results **and** the theory underneath |
| Only correct matches shown | Rejected candidates shown with the reason |

Pre-ship checks live in section 5; write them as tests.

---

## 7. Deliberate scope cuts

State these openly in the README. Naming your own limits reads as senior; having them found reads as sloppy.

- One face provider integrated fully rather than four partially
- No account creation, no logged-in access to private content, no pretexting — outside the scope of a public-source prototype, and it creates authentication and ToS problems this design does not attempt to solve
- No persistent face index — federated only, for the reasons above
- Scoring weights are tuned by hand, not learned; a real system would calibrate against labeled outcomes

---

## 8. Open questions for the call

- Does the existing engine collect or federate? Where is the line today?
- Is Loupe serving cached results or collecting live? "Minutes not days" implies pre-collection.
- What is Sentinel's cost per subject per month? Flat-rate pricing makes every wasted request margin.
- How is a wrong identity match caught today? Is there a human gate before a name enters a report?
- What breaks most often in collection right now?
