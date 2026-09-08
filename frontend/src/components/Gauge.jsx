/* A confidence reading, drawn as an instrument scale.
 *
 * Two things are deliberate. The fill is hatched, because these are
 * uncalibrated heuristics and a solid bar would read as a measurement. The
 * number is always printed beside it, because a bar alone invites eyeballing
 * a percentage that was never that precise.
 *
 * Colour is not used. A gauge that turned green at 0.8 would hand the reader
 * a verdict the score does not support.
 */
export default function Gauge({ value, size = "md", label }) {
  const known = typeof value === "number" && Number.isFinite(value);
  const pct = known ? Math.max(0, Math.min(1, value)) * 100 : 0;

  return (
    <div className={`gauge gauge--${size}`}>
      <div
        className="gauge__track"
        role="meter"
        aria-valuenow={known ? Number(value.toFixed(4)) : undefined}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-label={label || "confidence, an uncalibrated estimate"}
      >
        {known && <div className="gauge__fill" style={{ width: `${pct}%` }} />}
        <div className="gauge__ticks" aria-hidden="true">
          {[25, 50, 75].map((at) => (
            <span className="gauge__tick" key={at} style={{ left: `${at}%` }} />
          ))}
        </div>
      </div>
      <span className="gauge__value">{known ? value.toFixed(2) : "—"}</span>
    </div>
  );
}
