/* Per-source outcomes.
 *
 * Four answers, told apart by rule style and wording rather than by colour
 * alone. Failed carries a solid oxide rule and a printed reason; empty a
 * solid grey rule and "returned nothing"; skipped a dashed rule and "not
 * applicable", because it was never asked. A source that errored did not
 * answer the question, one that came back empty did, and one that was skipped
 * was never applicable to this subject.
 */
const WORDING = {
  found: (row) => `found ${row.result_count ?? 0}`,
  empty: () => "returned nothing",
  failed: () => "failed",
  // Not applicable, and nothing went wrong. A text search with no text was
  // never asked; saying "failed" sends someone hunting a fault that is not
  // there, and saying "empty" claims a search that never ran.
  skipped: () => "not applicable",
  searched: () => "searching",
};

export default function SourceRegister({ sources, planned, running }) {
  const pending = running
    ? planned.filter((q) => !sources.some((s) => s.query === q))
    : [];

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Source</th>
            <th>Query</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {sources.map((row, i) => (
            <tr key={`${row.source}-${row.query}-${i}`} className={`is-${row.status}`}>
              <td className="data">{row.source}</td>
              <td>{row.query}</td>
              <td>
                <span className={`status status--${row.status}`}>
                  {(WORDING[row.status] || (() => row.status))(row)}
                </span>
                {row.error_reason && (
                  // Oxide is reserved for failure and contradiction. A source
                  // that was never applicable is neither, so its reason is set
                  // as a plain note.
                  <div className={row.status === "skipped" ? "note" : "reason"}>
                    {row.error_reason}
                  </div>
                )}
              </td>
            </tr>
          ))}
          {pending.map((query) => (
            <tr key={`pending-${query}`} className="is-running">
              <td className="data muted">queued</td>
              <td className="muted">{query}</td>
              <td><span className="status muted">waiting</span></td>
            </tr>
          ))}
          {!sources.length && !pending.length && (
            <tr><td colSpan={3} className="empty-state">No sources have reported yet.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
