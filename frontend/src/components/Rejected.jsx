import Gauge from "./Gauge.jsx";
import Source from "./Source.jsx";

/* Records that did not match the subject.
 *
 * Open by default and ordered near-misses first. A plausible wrong match that
 * the system caught and can explain is the strongest evidence that the rest of
 * the report is worth trusting, so it is not hidden behind a collapsed panel.
 */
/* One rejected record.
 *
 * The bar and the figure are the record's overall match against the subject —
 * the number the decision was weighed against — not one component of it.
 *
 * Some records score high and are still refused, because a name was the only
 * signal they carried, and the strength of a lone signal is just that signal.
 * For those the score decided nothing, and saying so is the difference between
 * a report a reader trusts and one that looks like it contradicts itself.
 */
function Reject({ r, lead }) {
  const notDecisive = r.decided_by === "substance";
  return (
    <div className={`row${lead ? " reject-near" : ""}`}>
      <div>
        <Gauge value={r.strength} size="sm" label="overall match against the subject" />
        {notDecisive && <div className="row__caveat">score not decisive</div>}
      </div>
      <div className="row__why">{r.explanation}</div>
      <Source source={r.source} fallbackRef={r.record_ref} />
    </div>
  );
}

export default function Rejected({ rejected, collected }) {
  if (!rejected.length) {
    return (
      <p className="empty-state">
        {collected
          ? "Every record collected was used. Nothing was scored and set aside."
          : "No records were collected, so there was nothing to reject."}
      </p>
    );
  }

  // Two populations, and one ranking does not run through both. A record
  // refused by the threshold is ranked by how close it came. A record refused
  // on substance scored high and the score decided nothing, so 1.00 there
  // means less than 0.40 does above it. Presented as a single list called
  // "closest first", entries at 0.00 sat above entries at 1.00 and the section
  // contradicted itself on screen. They are labelled instead.
  const byThreshold = rejected.filter((r) => r.decided_by !== "substance");
  const onSubstance = rejected.filter((r) => r.decided_by === "substance");
  const near = byThreshold.slice(0, 8);
  const rest = byThreshold.slice(8);

  return (
    <>
      <div className="rejects">
        <p className="rejects__lead">
          {rejected.length === 1
            ? "One record looked like the subject and was not."
            : `${rejected.length} records looked like the subject and were not.`}{" "}
          Kept and scored rather than discarded.
        </p>

        {byThreshold.length > 0 && (
          <>
            <p className="note" style={{ marginBottom: "0.6rem" }}>
              {byThreshold.length === 1 ? "One scored" : `${byThreshold.length} scored`}{" "}
              below the threshold, closest first.
            </p>
            <div className="rows">
              {near.map((r, i) => (
                <Reject r={r} lead={i === 0} key={r.record_ref} />
              ))}
            </div>
          </>
        )}
      </div>

      {rest.length > 0 && (
        <details className="expand">
          <summary>{rest.length} more, scoring lower</summary>
          <div className="rows">
            {rest.map((r) => <Reject r={r} key={r.record_ref} />)}
          </div>
        </details>
      )}

      {onSubstance.length > 0 && (
        <div className="rejects" style={{ marginTop: "1rem" }}>
          <p className="note" style={{ marginBottom: "0.6rem" }}>
            {onSubstance.length === 1
              ? "One record was refused on substance, not on its score."
              : `${onSubstance.length} records were refused on substance, not on their scores.`}{" "}
            They score high on the name alone, and with nothing else to check
            that against the number is the name similarity and nothing more. It
            is shown, and it is not what decided them — so these are not ranked
            against the ones above.
          </p>
          <div className="rows">
            {onSubstance.map((r) => <Reject r={r} key={r.record_ref} />)}
          </div>
        </div>
      )}
    </>
  );
}
