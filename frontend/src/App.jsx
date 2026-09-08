import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl, runHeaders } from "./api";
import Candidate from "./components/Candidate.jsx";
import Gauge from "./components/Gauge.jsx";
import PhotoDrop from "./components/PhotoDrop.jsx";
import Rejected from "./components/Rejected.jsx";
import SourceRegister from "./components/SourceRegister.jsx";

const SECTIONS = [
  ["sources", "Sources"],
  ["findings", "Findings"],
  ["rejected", "Not the subject"],
  ["conflicts", "Conflicts"],
  ["loose", "Unattached"],
];

export default function App() {
  const [form, setForm] = useState({ name: "", address: "", context: "" });
  const [photo, setPhoto] = useState(null);
  const [error, setError] = useState(null);
  const [run, setRun] = useState(null);
  const [active, setActive] = useState("sources");
  const timer = useRef(null);

  useEffect(() => () => clearTimeout(timer.current), []);

  const poll = useCallback((id) => {
    fetch(apiUrl(`/runs/${id}`))
      .then((r) => r.json())
      .then((snapshot) => {
        setRun(snapshot);
        if (snapshot.status === "running") {
          timer.current = setTimeout(() => poll(id), 600);
        }
      })
      .catch((e) => setError(`Lost contact with the run: ${e.message}`));
  }, []);

  async function submit(event) {
    event?.preventDefault();
    setError(null);
    try {
      const response = await fetch(apiUrl("/runs"), {
        method: "POST",
        headers: runHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          ...form,
          photo_url: photo?.public_url || "",
          // Sent so the server reads its own file for face comparison
          // rather than fetching the photo back over a public URL.
          photo_digest: photo?.digest || "",
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The search could not start.");
      setRun({ id: data.run_id, status: "running", sources: [], planned: [] });
      setActive("sources");
      poll(data.run_id);
    } catch (e) {
      setError(e.message);
    }
  }

  const report = run?.report;
  const anchor = report?.candidates?.find((c) => c.is_anchor);
  const others = report?.candidates?.filter((c) => !c.is_anchor) || [];

  // Name what actually separated them. With no name supplied there was no name
  // comparison, and claiming one would describe a check that never ran.
  // Reads the same report.compared the cards do. These two used to derive the
  // wording separately and drifted: the summary said "records whose faces did
  // not match" while every card beneath it said "Other person, same name".
  const separatedBy = (n) => {
    const compared = run?.report?.compared;
    if (compared?.name) return `${n} other ${n === 1 ? "person" : "people"} sharing this name`;
    if (compared?.face)
      return `${n} ${n === 1 ? "record whose face did not match" : "records whose faces did not match"}`;
    return `${n} other ${n === 1 ? "record" : "records"} that did not match`;
  };

  return (
    <div className="sheet">
      <header className="masthead">
        <span className="masthead__name">Webutation</span>
        <span className="masthead__role">identity resolution for insurance investigation</span>
      </header>

      <div className="frame">
        <nav className="rail" aria-label="Report sections">
          {run
            ? SECTIONS.map(([id, label], i) => (
                <button
                  key={id}
                  type="button"
                  className={`rail__item${active === id ? " rail__item--on" : ""}`}
                  onClick={() => {
                    setActive(id);
                    document.getElementById(id)?.scrollIntoView({ block: "start" });
                  }}
                >
                  <span className="rail__num">§{i + 1}</span>
                  {label}
                </button>
              ))
            : (
                <div className="rail__intro">
                  <p>Every score here is an estimate, not a measurement.</p>
                  <p>Records that did not match are kept and shown, with the
                  reason they were rejected.</p>
                  <p>Nothing is overwritten. Conflicting claims stay side by
                  side.</p>
                </div>
              )}

          {run && (
            <dl className="rail__meta">
              <dt>Subject</dt>
              <dd>{run.subject?.name || "photo only"}</dd>
              {run.subject?.context && (<><dt>Context</dt><dd>{run.subject.context}</dd></>)}
              {report && (
                <>
                  <dt>Records</dt>
                  <dd>{report.spend.drafts}</dd>
                </>
              )}
            </dl>
          )}
        </nav>

        <main className="body">
          {!run && (
            <Intake
              form={form}
              setForm={setForm}
              photo={photo}
              setPhoto={setPhoto}
              onSubmit={submit}
              error={error}
              setError={setError}
            />
          )}

          {run && (
            <>
              <div className="toolbar">
                <button
                  type="button"
                  className="button button--quiet"
                  onClick={() => { clearTimeout(timer.current); setRun(null); }}
                >
                  Open another case
                </button>
                {run.status === "running" && (
                  <span className="progress-note">searching…</span>
                )}
              </div>

              {error && <p className="error">{error}</p>}
              {run.status === "failed" && (
                <p className="error">The run stopped: {run.error}</p>
              )}

              <Section id="sources" num={1} title="Sources"
                count={run.status === "running" ? "running" : plural(run.sources.length, "search", "searches")}>
                <SourceRegister
                  sources={run.sources}
                  planned={run.planned || []}
                  running={run.status === "running"}
                />
                {run.face && (
                  <p className="note">
                    {run.face.available
                      ? `Subject photo: ${run.face.faces_detected} face detected.`
                      : `No face comparison: ${run.face.error}.`}
                    {run.face.ambiguous && " More than one face in the photo, so a match may be to the wrong person."}
                  </p>
                )}
              </Section>

              <Section id="findings" num={2} title="Findings"
                count={report ? plural(report.candidates.length, "candidate") : "waiting"}>
                {!report && <p className="progress-note">Collecting and scoring…</p>}
                {report && !anchor && (
                  <p className="empty-state">
                    No record matched the subject well enough to be the person
                    searched for. The run completed; this is an answer, not a
                    failure.
                  </p>
                )}
                {anchor && <Candidate candidate={anchor} />}
                {others.length > 0 && (
                  <details className="expand">
                    <summary>{separatedBy(others.length)}</summary>
                    {others.map((c) => (
                      <Candidate candidate={c} key={c.index} />
                    ))}
                  </details>
                )}
              </Section>

              <Section id="rejected" num={3} title="Not the subject"
                count={report ? plural(report.rejected.length, "record") : ""}>
                {report && (
                  <Rejected
                    rejected={report.rejected}
                    collected={report.spend.drafts > 0}
                  />
                )}
              </Section>

              <Section id="conflicts" num={4} title="Conflicts"
                count={report ? `${report.conflicts.length}` : ""}>
                {report && report.conflicts.length === 0 && (
                  <p className="empty-state">
                    Nothing contradicts anything else. Two employers or two
                    addresses over different periods are not conflicts.
                  </p>
                )}
                {report?.conflicts.map((c, i) => (
                  <div className="cand" key={i}>
                    <div className="cand__stamp">
                      <span className="flag">{c.potential ? "possible conflict" : "conflict"}</span>{" "}
                      {c.predicate.replace(/_/g, " ")}
                    </div>
                    <p style={{ marginBottom: "0.4rem" }}>
                      {c.values.join("  vs  ")}
                    </p>
                    <p className="note">{c.basis}</p>
                    <p className="note">
                      Status {c.status}. Both claims are kept; nothing was
                      overwritten and no preferred value was chosen.
                    </p>
                  </div>
                ))}
              </Section>

              <Section id="loose" num={5} title="Unattached"
                count={report ? `${report.unattached.length}` : ""}>
                {report && report.unattached.length === 0 && (
                  <p className="empty-state">Every claim was attached to a candidate.</p>
                )}
                {report?.unattached.length > 0 && (
                  <>
                    <p className="note">
                      These matched no candidate above threshold. Retained
                      rather than forced onto the nearest person.
                    </p>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr><th>Claim</th><th>Value</th><th>Origin</th></tr>
                        </thead>
                        <tbody>
                          {report.unattached.map((u, i) => (
                            <tr key={i}>
                              <td>{u.predicate.replace(/_/g, " ")}</td>
                              <td className="data">{u.value}</td>
                              <td className="data">{u.origin || <span className="unknown">unnamed</span>}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </Section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function plural(n, one, many) {
  return `${n} ${n === 1 ? one : many || `${one}s`}`;
}

function Section({ id, num, title, count, children }) {
  return (
    <section className="section" id={id}>
      <div className="section__head">
        <span className="section__num">§{num}</span>
        <h2 className="section__title">{title}</h2>
        {count && <span className="section__count">{count}</span>}
      </div>
      {children}
    </section>
  );
}

function Intake({ form, setForm, photo, setPhoto, onSubmit, error, setError }) {
  return (
    <form onSubmit={onSubmit}>
      <Section id="intake" num={1} title="Open a case">
        <p className="note" style={{ marginBottom: "1rem" }}>
          A photo or a name is enough. Supplying both narrows the search and
          gives the face check something to compare against.
        </p>

        {error && <p className="error">{error}</p>}

        <div className="split">
          <div className="stack">
            <PhotoDrop photo={photo} onPhoto={setPhoto} onError={setError} />
          </div>

          <div>
            <label className="field">
              <span className="field__label">Name</span>
              <input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="Michael Petrie"
                autoComplete="off"
              />
            </label>
            <label className="field">
              <span className="field__label">Address or locality</span>
              <input
                value={form.address}
                onChange={(e) => setForm({ ...form, address: e.target.value })}
                placeholder="Austin, TX"
                autoComplete="off"
              />
            </label>
            <label className="field">
              <span className="field__label">Context</span>
              <textarea
                value={form.context}
                onChange={(e) => setForm({ ...form, context: e.target.value })}
                placeholder="employer, industry, claim reference, anything that narrows it"
              />
              <span className="field__hint">
                Each comma-separated term becomes its own search.
              </span>
            </label>
            <button className="button" type="submit">Open the case</button>
          </div>
        </div>
      </Section>

    </form>
  );
}
