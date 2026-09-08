/* Where a record came from, as something a person can read.
 *
 * The record reference is an internal identity, and for a result the source
 * returned without a URL it is a hash of that result's bytes —
 * "serpapi:google:sha256:0f085f7e010adeb4:10". Sound as an identity, and
 * unreadable on screen: it looks like a bug rather than the fact it stands
 * for, which is that the source supplied no link.
 *
 * The server sends a `source` block that has already worked this out. This
 * component must still handle its absence, because a page left open across a
 * deploy holds an older report in memory and renders it through today's
 * component. The first version fell back to printing the raw reference, which
 * meant the one thing this component exists to prevent was one stale tab away.
 * Nothing here prints a reference; when the block is missing the same rule is
 * applied to the reference locally instead.
 */

/* The server's rule, restated for reports that predate it. Keep the two in
 * step: app/report.py, record_source(). */
function deriveSource(ref) {
  if (typeof ref !== "string" || !ref) {
    return { url: null, title: null, note: "no link supplied by the source" };
  }
  const parts = ref.split(":");
  const rest = parts.length > 2 ? parts.slice(2).join(":") : ref;
  if (rest.startsWith("sha256:")) {
    return { url: null, title: null, note: "no link supplied by the source" };
  }
  return { url: rest, title: null, note: null };
}

export default function Source({ source, fallbackRef }) {
  const { url, title, note } = source || deriveSource(fallbackRef);

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
