import Gauge from "./Gauge.jsx";

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
      <div className="row__ref">{r.record_ref}</div>
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

  const near = rejected.slice(0, 8);
  const rest = rejected.slice(8);
  const onSubstance = rejected.filter((r) => r.decided_by === "substance").length;

  return (
    <>
      <div className="rejects">
        <p className="rejects__lead">
          {rejected.length === 1
            ? "One record looked like the subject and was not."
            : `${rejected.length} records looked like the subject and were not.`}{" "}
          Kept and scored rather than discarded, closest first.
        </p>
        {onSubstance > 0 && (
          <p className="note" style={{ marginBottom: "0.6rem" }}>
            {onSubstance === 1 ? "One of them scores" : `${onSubstance} of them score`}{" "}
            high on the name alone. With nothing else to check it against, that
            score is the name similarity and nothing more, so it was not what
            decided them.
          </p>
        )}
        <div className="rows">
          {near.map((r, i) => (
            <Reject r={r} lead={i === 0} key={r.record_ref} />
          ))}
        </div>
      </div>

      {rest.length > 0 && (
        <details className="expand">
          <summary>{rest.length} more, scoring lower</summary>
          <div className="rows">
            {rest.map((r) => <Reject r={r} key={r.record_ref} />)}
          </div>
        </details>
      )}
    </>
  );
}
