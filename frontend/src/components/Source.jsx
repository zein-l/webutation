/* Where a record came from, as something a person can read.
 *
 * The record reference is an internal identity, and for a result the source
 * returned without a URL it is a hash of that result's bytes —
 * "serpapi:google:sha256:0f085f7e010adeb4:10". Sound as an identity, and
 * unreadable on screen: it looks like a bug rather than a fact. The reference
 * is still in the report for tracing; this is what gets shown.
 */
export default function Source({ source, fallbackRef }) {
  if (!source) return <div className="row__ref">{fallbackRef}</div>;

  const { url, title, note } = source;
  if (url) {
    return (
      <div className="row__ref">
        <a href={url} target="_blank" rel="noopener noreferrer">{url}</a>
      </div>
    );
  }
  return (
    <div className="row__ref">
      <span className="row__title">{title || "untitled record"}</span>
      {note && <span className="row__note"> — {note}</span>}
    </div>
  );
}
