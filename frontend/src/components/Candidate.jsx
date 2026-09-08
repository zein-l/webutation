import { useState } from "react";
import Gauge from "./Gauge.jsx";
import Source from "./Source.jsx";

/* One hypothesised person, with everything that holds them together.
 *
 * The anchor is visually distinct because it answers the question that was
 * asked; the others are people who happen to share a name. That difference is
 * carried by border weight and scale rather than by colour, so it survives
 * being printed and never reads as a verdict.
 */

const EVIDENCE_WORDS = {
  direct_observation: "observed it",
  self_reported: "self-reported",
  secondhand: "secondhand",
  republished: "republished",
  unknown: "unknown",
};

/* Keep both ends; lose the middle.
 *
 * A truncated URL still has to answer "which site, and which page", and those
 * sit at opposite ends. Cutting the end off, as the browser would, throws away
 * the half that identifies the record. A page title used as a name has the
 * person at the front and the site at the back, and the same applies.
 *
 * The table still records exactly what the source printed. The full string is
 * on the element and one button away.
 */
function shorten(value, max = 50) {
  if (!value || value.length <= max) return value;

  if (value.includes("://")) {
    let head;
    try {
      head = new URL(value).origin;
    } catch {
      head = value.slice(0, Math.floor(max / 2));
    }
    const tail = Math.max(10, max - head.length - 1);
    return `${head}…${value.slice(-tail)}`;
  }

  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${value.slice(0, head).trimEnd()}…${value.slice(-tail).trimStart()}`;
}

function LongValue({ value }) {
  const [copied, setCopied] = useState(false);
  const short = shorten(value);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  if (short === value) return <span className="urlcell__text">{value}</span>;


  return (
    <span className="urlcell">
      <span className="urlcell__text" title={value}>{short}</span>
      <button type="button" className="urlcell__copy" onClick={copy}>
        {copied ? "copied" : "copy"}
      </button>
    </span>
  );
}

/* Faces that were checked against the uploaded photo and matched.
 *
 * Every image here cleared the threshold. Images that failed, or that could
 * never be compared, are absent rather than shown with a warning: a stranger's
 * face under a person's name is the false attribution this system exists to
 * prevent, and no caption undoes a photograph. Nothing verified means nothing
 * rendered, not an empty frame.
 */
// What to call a candidate that is not the subject.
//
// The server decides this, because it is the only place that knows what was
// compared for *this* candidate. Two earlier versions derived it here and both
// were wrong: the first hardcoded "Other person, same name", and the second
// read the run-level `compared` block, which says a name was supplied and not
// that it matched anything. On a mixed search most candidates come from
// reverse image search and score 0.00 on the name — 79 of 116 in one live run,
// every one of them labelled as sharing a name it did not share.
//
// The fallback is for reports built before the server sent a stamp. It says
// only what is true of any non-anchor candidate, and claims no comparison.
function otherLabel(candidate) {
  return candidate?.stamp || "Other record, not matched";
}

function VerifiedFaces({ images }) {
  if (!images?.length) return null;

  return (
    <div className="faces">
      <div className="faces__strip">
        {images.map((img) => (
          <a
            key={img.image_url}
            className="faces__item"
            href={img.source_page}
            target="_blank"
            rel="noopener noreferrer"
            title={`Face similarity ${img.similarity.toFixed(2)} to the uploaded photo — ${img.publisher || "source"}`}
          >
            <img src={img.image_url} alt={`Matched face from ${img.publisher || "a source"}`} loading="lazy" />
            <span className="faces__score">{img.similarity.toFixed(2)}</span>
          </a>
        ))}
      </div>
      <p className="note">
        {images.length === 1 ? "One image" : `${images.length} images`} matched
        the uploaded photo. The figure is cosine similarity, an uncalibrated
        estimate. Images that did not match are not shown.
      </p>
    </div>
  );
}

function Freshness({ value }) {
  // Rendered as a word, never a number. A source that never said when it
  // looked has told us nothing about recency, and 0.50 would hide that.
  if (value === "unknown") return <span className="unknown">unknown</span>;
  return <span>{Number(value).toFixed(2)}</span>;
}

export default function Candidate({ candidate }) {
  const {
    is_anchor: isAnchor,
    name,
    display_name: displayName,
    localities,
    p,
    held_by: heldBy,
    assertions,
    corroboration,
    anchored_by: anchoredBy,
    merge_evidence: mergeEvidence,
    penalties,
    verified_images: verifiedImages,
  } = candidate;

  // Freshness is a per-claim column only when some claim actually has a date.
  // Web results almost never carry one, and a column of the word "unknown"
  // repeated seven times says less than one sentence does.
  const anyDated = assertions.some((a) => a.freshness !== "unknown");

  return (
    <article className={`cand${isAnchor ? " cand--anchor" : ""}`}>
      {isAnchor ? (
        <div className="cand__stamp">{candidate.stamp || "The person searched for"}</div>
      ) : (
        <div className="cand__stamp muted">{otherLabel(candidate)}</div>
      )}

      <h3 className="cand__name">{displayName || name || "unnamed"}</h3>
      <div className="note">
        {localities.length ? localities.join(" · ") : "no locality on record"}
      </div>

      <div className="cand__p">
        <Gauge value={p} size={isAnchor ? "lg" : "md"} label="identity confidence" />
      </div>
      <div className="note">
        Identity confidence. An uncalibrated estimate, not a probability.
      </div>

      <p className="cand__held">
        Held by: <b>{heldBy}</b>
      </p>

      {isAnchor && <VerifiedFaces images={verifiedImages} />}

      {Object.keys(penalties || {}).length > 0 && (
        <p className="note">
          Marked down for{" "}
          {Object.keys(penalties).map((k) => k.replace(/_/g, " ")).join(" and ")}.
        </p>
      )}

      {assertions.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Claim</th>
                <th>Value</th>
                <th className="num">q</th>
                <th className="num">r</th>
                <th className="origin">Origin</th>
                <th>How they know</th>
                {anyDated && <th>Freshness</th>}
              </tr>
            </thead>
            <tbody>
              {assertions.map((a, i) => (
                <tr key={`${a.record_ref}-${a.predicate}-${i}`}>
                  <td>{a.predicate.replace(/_/g, " ")}</td>
                  <td className="data">
                    <LongValue value={a.raw_value || a.value} />
                  </td>
                  <td className="num">
                    <Gauge value={a.q} size="sm" label={`claim confidence for ${a.predicate}`} />
                  </td>
                  <td className="num">{a.r.toFixed(2)}</td>
                  <td className="data origin">
                    {a.origin || <span className="unknown">unnamed</span>}
                  </td>
                  <td>{EVIDENCE_WORDS[a.evidence_kind] || a.evidence_kind}</td>
                  {anyDated && <td><Freshness value={a.freshness} /></td>}
                </tr>
              ))}
            </tbody>
          </table>
          {!anyDated && (
            <p className="legend">
              No source carried an observation date, so none of these claims
              can be dated.
            </p>
          )}
          <p className="legend">
            q is how far this claim is believed, given the identity is right.
            r is q scaled by that identity confidence. Both are estimates.
          </p>
        </div>
      )}

      <div className="corrob">
        <span className="corrob__counts">
          {corroboration.named_publishers} named publishers,{" "}
          {corroboration.unattributed} unattributed,{" "}
          {corroboration.distinct_origins} distinct origins
        </span>
        <div className="corrob__caveat">
          Publishers are not confirmations. Sites that republish one source
          count once.
        </div>
      </div>

      {(anchoredBy?.length > 0 || mergeEvidence?.length > 0) && (
        <details className="expand">
          <summary>Why these records were grouped</summary>
          {anchoredBy?.length > 0 && (
            <div className="rows">
              {anchoredBy.map((m) => (
                <div className="row" key={m.record_ref}>
                  <Gauge value={m.strength} size="sm" label="match against the subject" />
                  <div className="row__why">
                    {m.basis}
                    {m.face_ambiguous && (
                      <> <span className="flag">more than one face</span></>
                    )}
                  </div>
                  <Source source={m.source} fallbackRef={m.record_ref} />
                </div>
              ))}
            </div>
          )}
          {mergeEvidence?.map((line, i) => (
            <p className="note" key={i}>{line}</p>
          ))}
        </details>
      )}
    </article>
  );
}
