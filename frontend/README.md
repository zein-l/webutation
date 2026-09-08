# Frontend

```bash
npm install
npm run dev          # http://127.0.0.1:5173
```

Needs the API on port 8000: `uvicorn app.api:app --port 8000`. Vite proxies
`/runs`, `/subjects`, `/fixtures`, `/uploads` and `/health` to it, so the app
is same-origin in development and needs no CORS configuration.

## Design notes

**Colour never encodes confidence.** A gauge that turned green at 0.80 would
hand a reviewer a verdict the score does not support, and the whole point of
this interface is to stop a confident-looking number being read as a fact.
Oxide red appears only where something failed, was refused, or contradicts
something else. Confidence is carried by bar geometry and a printed figure.

**The gauge is hatched on purpose.** Diagonal hatching over the filled portion
says the reading is estimated rather than measured; the tick marks at 0.25,
0.50 and 0.75 say it sits on a scale. Every gauge prints its number beside it,
because a bar alone invites eyeballing a precision nobody has.

**Freshness is a word.** When a source never said when it observed something,
the cell reads "unknown". A number there would be a lie a reader could not see
through.

**Rejected records are a finding.** A plausible wrong match that the system
caught and can explain is the strongest evidence that the rest of the report is
worth trusting, so it gets its own panel, ordered closest-first, rather than a
collapsed footnote.

**Empty and failed never look alike.** A source that returned nothing answered
the question; a source that errored did not. Failed carries an oxide rule, a
wash, and the printed reason. Empty carries a grey rule and the words "returned
nothing" — the two are distinguishable without relying on colour.

Type is Archivo for text and Courier Prime for values only — scores, origins,
digests, URLs. Courier is the vernacular of legal exhibits; it never sets a
label or a heading.
