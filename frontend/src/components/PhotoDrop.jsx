import { useRef, useState } from "react";
import { apiUrl } from "../api";

/* Subject photo intake.
 *
 * The upload response says whether the photo is reachable from outside. That
 * is repeated here because it decides whether reverse image search can run at
 * all, and finding out later — as a source that mysteriously found nothing —
 * is the failure this wording exists to prevent.
 */
export default function PhotoDrop({ photo, onPhoto, onError }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef(null);

  async function send(file) {
    if (!file) return;
    setBusy(true);
    onError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(apiUrl("/subjects/photo?ttl_seconds=3600"), {
        method: "POST",
        body,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The upload was refused.");
      onPhoto({ ...data, preview: URL.createObjectURL(file), filename: file.name });
    } catch (error) {
      onError(error.message);
    } finally {
      setBusy(false);
    }
  }

  if (photo) {
    return (
      <div className="drop">
        <div className="drop__preview">
          <img src={photo.preview} alt="Subject photo as uploaded" />
          <div>
            <div className="drop__meta">{photo.filename}</div>
            <div className="drop__meta muted">
              {(photo.bytes / 1024).toFixed(0)} KB · {photo.content_type}
            </div>
            <p className="note" style={{ margin: "0.4rem 0" }}>
              {photo.publicly_reachable
                ? "Reachable from outside, so reverse image search can run."
                : "Not reachable from outside. Reverse image search will report failed; face comparison against images found by other sources still runs."}
            </p>
            <button
              type="button"
              className="button button--quiet"
              onClick={() => onPhoto(null)}
            >
              Remove photo
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <button
        type="button"
        className={`drop${over ? " drop--over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          send(e.dataTransfer.files?.[0]);
        }}
      >
        <div>{busy ? "Uploading the photo…" : "Drop a photo of the subject here"}</div>
        <div className="drop__hint">or choose a file — JPEG, PNG or WebP, up to 10 MB</div>
      </button>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(e) => send(e.target.files?.[0])}
      />
    </>
  );
}
