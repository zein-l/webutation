# Webutation

Finds what the open web says about a person, and is explicit about how much of
it is worth believing.

The hard part is not collection. It is refusing to state more than the evidence
supports: not merging two people because they share a name, not counting five
sites that copied one aggregator as five confirmations, and not presenting a
number as a fact when it is an uncalibrated guess.

## If you have ten minutes

1. **What it refuses to do** is the design. Skip to *Things the system refuses
   to do* — each entry is a failure this system hit and closed, with the
   arithmetic that closed it.
2. **The numbers are guesses.** *What the numbers mean* explains why `p` and `q`
   are capped below 1.00 and never presented as probabilities.
3. **What it still gets wrong** is in *Known limits*, including one that has no
   fix and one whose constant is set on precedent rather than measurement.
4. The test names are the specification: `pytest -q` runs 420 of them, and files
   like `tests/test_name_is_not_evidence.py` read as prose.

## Running it

```bash
python -m venv venv && ./venv/Scripts/pip install -r requirements.txt
cp .env.example .env          # then fill in SERPAPI_KEY
docker compose up -d          # Postgres with pgvector
python -m app.db              # create the schema
pytest
```

`create_all` only adds what is missing, so it will not pick up a change to an
existing table or to an enum type. There are no migrations by design; after a
model change, recreate:

```bash
python -m app.db --drop       # drops every table, then recreates
```

One subject, end to end, printing a report and nothing to the database:

```bash
python scripts/run_subject.py \
  --name "Marcus Webb" \
  --context "Austin, TX, logistics operations"
```

Every search goes through a local cache keyed by request, so re-running the
same subject costs no API calls. The report prints how many live calls a run
actually made.

The HTTP surface exists only to get a subject photo onto the web:

```bash
uvicorn app.api:app --reload
```

## Photo retention: 24 hours, by choice

An uploaded subject photo is **deleted 24 hours after it was last uploaded**.
A background sweep runs every 15 minutes, on the app's lifespan, and also once
at startup so a process that was down longer than the window does not serve
stale photos. `DELETE /subjects/photo/{digest}` removes one immediately.

This is a deliberate number, not a default nobody chose. Reverse image search
requires publishing a photograph of a person to a URL that a third party's
servers can fetch. That is not a neutral act, and the two bounds are separate
on purpose:

| Bound | Window | What it limits |
| --- | --- | --- |
| Signed URL lifetime | 15 minutes | how long a leaked link works |
| File retention | 24 hours | how long the photograph exists at all |

Twenty-four hours is long enough to re-run a search against the same subject
without asking for the file again, and short enough that an abandoned
investigation does not leave a face on disk indefinitely. Both are configurable
(`UPLOAD_RETENTION_SECONDS`, `PUBLIC_BASE_URL`, `UPLOAD_SIGNING_KEY`), and a
retention value that is absent or nonsense falls back to 24 hours rather than
to "keep forever".

Upload URLs carry an HMAC token in the query, so they are unguessable rather
than merely obscure, and an expired or wrong token returns 404 rather than 403
— whether a given photo exists is itself information about who has been
searched for.

**Tunnel hosts.** The signed route sets `Bypass-Tunnel-Reminder: true` and a
`Server` header so a tunnel edge is less likely to serve a click-through page
in place of the image. This is best effort and cannot be relied on: the check a
tunnel makes is on the *request*, and the request belongs to whichever search
engine is fetching. We can set that header on our own fetches and not on
theirs. A source that still cannot retrieve the photo is reported `failed` with
the reason, which is the honest outcome.

**Not implemented:** there is no retention on the search-response cache under
`cache/`, which holds the text and image artifacts a run collected. That is a
gap, not a decision.

## Deploying it

`render.yaml` is a Render blueprint describing three resources: the API as a
Docker web service, a Postgres database, and the frontend as a static site.

The face model is baked into the image at build time (`Dockerfile`), not
downloaded on first request. A cold container would otherwise stall for ~300MB
on the first search that carries a photo, and pay it again after every restart.
A build that cannot fetch the weights fails rather than producing an image that
will fail later on someone's request.

The frontend calls relative paths, and the blueprint rewrites them to the API —
the same shape as the Vite proxy used in development, so neither side needs CORS.

**The run endpoint is guarded, because it is the only one that spends money.**
Two separate controls, in `app/ratelimit.py`:

- `RUN_ACCESS_TOKEN` — a shared secret sent as `X-Run-Token`. It decides *who*
  may ask and stops a crawler that found the endpoint. It is baked into the
  frontend bundle, so it is public; treat it as a speed bump, not a secret.
- `MAX_RUNS_PER_DAY` (default 100) and `MAX_RUNS_PER_IP_HOURLY` (default 10) —
  these decide *how much* may be spent, and hold even once the token has leaked.
  The daily cap is the control that actually protects the budget.

Both are in-process, so they reset on restart and are not shared between
instances. That is honest for one Starter instance; scaling out needs this state
in Postgres or Redis.

Two things about the deployed service worth knowing before reading too much into
it. The HTTP path never opens a database session — runs are held in memory and
lost on restart, so the database and its schema exist for the persistence layer
that `scripts/` and the tests exercise, not because the API writes to it. And
the search cache and uploaded photos live on the container's ephemeral disk, so
a restart loses both: cached searches must be re-bought, and photos disappear
ahead of their retention window rather than after it.

## What the numbers mean

Two scores are estimated and one is derived from them.

- **p** — is this candidate the same human as the subject?
- **q** — *given the match is right*, is this claim true and current?
- **r = p × q** — derived, computed in one place, never estimated directly.

`q` is never capped at `p`. `min(p, q)` destroys signal: at p=0.40 it returns
0.40 whether q is 0.95 or 0.55, where the product separates those as 0.38 and
0.22. Nothing in scoring reads `p` while computing `q`.

**These are confidence scores, not probabilities.** They are uncalibrated
heuristics, and **`p` and `q` are both capped at 0.95** for that reason: every
input to either is a guess — reliability is a table of assumptions about
publishers, freshness is a decay curve nobody fitted, corroboration counts
websites — so 1.00 would assert a certainty none of them support. `r` is
therefore bounded at 0.9025. There is no labelled outcome set to calibrate against, so the
multiplication mirrors the chain rule without being one. `p` and `q` are always
displayed beside `r`, because the product is lossy — 0.4×0.9 and 0.9×0.4 both
give 0.36.

The face-match threshold (cosine similarity 0.45) is a conventional value for
`buffalo_l`, not a measurement against this task.

## Things the system refuses to do

**Count publishers as confirmations.** Corroboration counts distinct
`origin_key` values, never publishers and never rows. Five sites republishing
one aggregator is one origin. A suspected copy earns no independence bonus and
unknown lineage is never counted as independent. Reports render
`5 publishers, 2 unattributed, 1 distinct origin`, never `5 confirmations`.

**Merge on a shared name.** Blocking proposes; only evidence disposes. Records
sharing only a name-derived key stay separate candidates, and a candidate held
together by a name alone is multiplied down to 0.40 of its other signals. Two
people called Michael Petrie in different cities stay two people.

**Score context as a share of what was asked.** Context counts matching terms
toward a saturation point instead of dividing by however many terms the caller
typed. A share is unreachable — a search snippet cannot hold six context terms,
and the highest value ever observed was 0.50 against a 0.65 threshold — and it
punished precision: the same record, on identical evidence, scored 1.00 for a
caller who typed "Webutation" and 0.17 for one who typed "Webutation, private
investigator, insurance fraud, OSINT". The saturation value is set by precedent
rather than measurement; see the note on `anchor_context_saturation`.

**Anchor on a trade.** Context alone never admits a record. A name is a label
many people share and cannot admit on its own; a profession is a label far more
people share. A subject who supplies only context therefore gets no anchor at
all — "private investigator" identifies a job, not a person — and a photo-only
subject is anchored by the face, which is evidence about a human being rather
than about a category.

**Anchor on a shared name.** The same rule, in the one place that used to be
exempt. A record joins the anchor only if it carries the subject's name *and* a
signal that is not the name: an overlapping context term, an agreeing locality,
or a face match. A locality has to name a settlement to count — a shared state
scores 0.25 and cannot admit anything, because "Austin, TX" and "Houston, TX"
are not the same place and millions of people share a state. A locality that
disagrees refuses the record outright rather than being averaged away. The anchor
strength is a mean over the signals that could be computed, so a record whose
only comparable signal is its name scores exactly its name similarity — an exact
match on a common name reads 1.00 and settles nothing. Admitting on that put a
baseball roster, a real estate agent, an obituary, a professor and an
insider-trading filing inside one candidate as one man. A link used to satisfy
the check and no longer does: a page existing says nothing about which human it
describes. The consequence is deliberate — a search on a name alone, with no
address, context or photo, produces no anchor at all, because there is nothing
to verify it against.

**Print a perfect score.** p and q are both capped at 0.95. Every input to
either is an uncalibrated heuristic — reliability is a table of guesses about
publishers, freshness is a decay curve nobody fitted, corroboration counts
websites — so 1.00 asserts a certainty none of them support and contradicts the
"not a probability" disclaimer printed beside it. The caps are separate settings
from `score_ceiling`, which is what a fully present signal is worth; one field
for all of them would make any of them impossible to change alone.

**Mistake a crowd for a consensus.** Corroboration counts distinct origins that
*agree*, not origins present. It used to count presence, which meant a pile of
same-name strangers grew more convincing with every stranger added — the
inversion this system exists to prevent, arriving through the signal meant to
guard against it. Agreeing on the name or the locality does not count: those are
what put the records in one pile, so they cannot also confirm it.

**Hide the wrong answers.** Records that fail the subject anchor are kept,
clustered separately, and reported with the score they missed by. Showing which
same-name people were rejected, and why, is the point.

**Parse prose into facts.** A search snippet reads like structured data and is
not. Only the displayed name, the page URL and image URLs become assertions.
Snippets are kept as context against the record. An employer parsed out of a
sentence would be scored and corroborated as though a source had asserted it.

**Resolve conflicts.** Conflicting assertions are stored, never overwritten.
Different addresses over different periods are a move, not a conflict. Two
employers are compatible. A conflict may also mean the *grouping* was wrong
rather than the fact, which is why `possible_bad_merge` and
`possible_bad_anchor` are distinct reasons.

**Confuse empty with failed.** A source that returned nothing answered the
question. A source that errored did not. They never collapse into one outcome.

**Derive freshness from collection time.** Recency uses `observed_at`. When a
source did not say when it observed something, freshness is *unknown* — never
inferred from `fetched_at`. A mirror fetched today must not refresh a five-year
old claim.

## Two axes that are easy to conflate

`source_origin` says **where** a claim came from and feeds corroboration.
`evidence_kind` says **how** the source came to know it and feeds reliability.
They are independent: a first-party origin can relay something it never
observed. One field serving both would make every lineage judgement double as a
reliability judgement.

`access_category` (`PUBLIC_WEB` / `LICENSED` / `RESTRICTED`) is descriptive
metadata and is deliberately **not** an input to either score. Licensing is an
access right, not a reliability signal.

## Evidence preservation, precisely

Raw responses are stored with a SHA-256 and a `fetched_at` before any parsing,
and can be re-parsed without re-collection.

This preserves **that response**, not the original post it references. Hash and
timestamp support integrity checking. They do not establish authorship, truth,
or admissibility — FRE 901–902 authentication requirements still apply. Call
this "raw collection artifacts with integrity hashes", not chain of custody.

## Stated non-goals

**Eligibility use is unassessed.** FCRA compliance for employment or insurance
decisions involves permissible purpose, notices and adverse-action procedures
that a schema field cannot establish. Nothing here adjudicates permitted use,
and `access_category` records a characteristic rather than granting one.

**Copy detection is out of scope.** Inferring that one site republished another
needs a corpus of shared rare errors. Lineage is marked `unknown` and reported
as unknown rather than guessed. A consequence: several aggregator domains each
count as a distinct origin, which overstates independence. Their
`evidence_kind` is `republished`, so reliability drops; independence does not.

**One face provider, no persistent face index.** Faces come from InsightFace
`buffalo_l` and nothing else, and embeddings live only for the run that computed
them. There is no vector index, so faces cannot be searched across subjects or
across time, and a second provider's disagreement is never available as a check
on the first. The pgvector extension is installed and the column type exists;
the index is not built. A single provider means a single failure mode: when
`buffalo_l` is wrong about two photographs, nothing in the system is positioned
to notice.

**Review status is human-set.** An uncalibrated score crossing a threshold does
not make something a finding. Scores sort; people decide. `reviewed` means a
person looked, not that every claim was accepted.

## Known limits

- **Record fragmentation.** Web search supplies no shared distinctive
  attributes and no faces, so genuinely-same people stay separate candidates
  unless a photo or an email links them. The anchor collapses what it can; the
  rest is visible as a long tail of weak singletons rather than hidden by a
  wrong merge.
- **Name extraction is a title split.** A page title's leading segment becomes
  the name. A badly formed title yields a junk name, which then fails to block
  and lands unattached.
- **Context matching is unweighted bag-of-words.** Terms are lower-cased, split
  on whitespace, filtered for length and stopwords, and compared as a set. No
  weighting, no stemming, no synonyms, no phrases: "insurance fraud" and "claims
  litigation" are the same field to a human and share no tokens, while
  "Webutation" and "private" each count once and equally. A frequency-weighted
  version was tried and reverted — document frequency cannot distinguish a term
  that is everywhere because the search biased the results from one that is
  everywhere because the records really are all one person, and those need
  opposite treatment. See `_anchor_context_signal`.
- **`anchor_context_saturation = 2.0` is measured on one subject.** Two matching
  context terms count as full agreement. The evidence is a single corpus of 133
  records, where the change took admissions from 4 to 10 with all 10 verified as
  the subject. Precision 1.00 on ten admissions is encouraging, not proven, and
  the value follows `item_corroboration_saturation`'s precedent rather than any
  fit. It is load-bearing: at 2.0 the subject's own RocketReach page is admitted
  at 0.70, at 3.0 it is refused at 0.63. Re-measure before trusting it on a
  second subject.
- **A more complete anchor can report a lower `p`.** The anchor's name component
  is the *weakest* member's name agreement, on the reasoning that `p` answers
  "are these one person" and the least convincing member is what that claim
  rests on. So admitting a further record that is genuinely the subject, but
  whose page title contains rather than equals the name, lowers `p`. Observed:
  an anchor grew from 2 correct members to 8 correct members and `p` fell from
  0.57 to 0.49. The score got more honest and less flattering at the same time,
  which is defensible but reads backwards.
- **A namesake in the same line of work is admissible.** Anchoring asks for the
  subject's name plus one other signal, and a context term is one. Someone who
  shares the name *and* the profession satisfies both, so a second Michael
  Petrie who also investigates insurance fraud would join the anchor. No context
  rule fixes this — the terms that identify a person's field are exactly the
  terms their colleagues and namesakes in that field also match. Only evidence
  about the individual rather than the category separates them: a face, an
  email, a phone number. This is the cost of making context admit at all, and it
  is a deliberate trade against the false negatives described below.
- **Yandex reverse image search is off by default.** It refuses photo URLs
  served from a development tunnel — "the URL does not refer to an image, or the
  image is not publicly accessible" — while Google Lens fetches the same URL
  without complaint. Since failures are no longer cached, leaving it enabled
  buys that refusal on every run. It has different recall from Lens and is worth
  having: re-enable it after deploying to a real domain, where the rejection may
  not apply, with `SerpApiConfig(disabled_engines=frozenset())`.
- **`observed_at` is usually null** on web results, so most claims correctly
  report unknown freshness.
