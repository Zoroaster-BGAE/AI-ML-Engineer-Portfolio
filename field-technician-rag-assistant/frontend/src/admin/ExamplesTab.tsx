import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, ApiError } from "../api/client";
import type { Example, FailureCause, GoldenSetRow } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useI18n, type TKey } from "../i18n";

const CAUSES: FailureCause[] = ["retrieval_miss", "wrong_machine", "answer_error", "manual_missing", "safety", "other"];
type Status = Example["status"];

/**
 * Golden-example review (design §5.4). The server sends a short, prioritised
 * batch -- safety first, then hard cases, then the least-covered machines -- so
 * two reviewers are never facing an endless list.
 */
export function ExamplesTab() {
  const { t } = useI18n();
  const { user } = useAuth();
  const [status, setStatus] = useState<Status>("candidate");
  const [rows, setRows] = useState<Example[] | null>(null);
  const [open, setOpen] = useState<Example | null>(null);

  const load = useCallback(() => {
    setRows(null);
    api<Example[]>(`/api/review/examples?status=${status}&limit=${status === "candidate" ? 10 : 100}`)
      .then(setRows).catch(() => setRows([]));
  }, [status]);
  useEffect(load, [load]);

  if (open) return <ExampleDetail ex={open} onClose={() => { setOpen(null); load(); }} />;

  return (
    <>
      <div className="admin-tabs" role="tablist" aria-label={t("examples")}>
        {(["candidate", "approved", "rejected"] as Status[]).map((s) => (
          <button key={s} role="tab" aria-selected={status === s} className={status === s ? "scope" : "mini"}
                  onClick={() => setStatus(s)}>
            {t(s === "candidate" ? "toReview" : s)}
          </button>
        ))}
      </div>
      {!rows && <div className="note">{t("loading")}</div>}
      {rows?.length === 0 && <div className="note">{t("noCandidates")}</div>}
      {rows && rows.length > 0 && (
        <div className="tw">
          <table>
            <thead><tr><th /><th>{t("machine")}</th><th>{t("question")}</th><th className="num">★</th><th /></tr></thead>
            <tbody>
              {rows.map((e) => (
                <tr key={e.id}>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {e.safety && <span className="tag tag--bad" style={{ marginRight: 4 }}>{t("safety")}</span>}
                    <span className={`tag ${e.kind === "hard" ? "tag--warn" : "tag--ok"}`}>{t(`kind_${e.kind}` as TKey)}</span>
                  </td>
                  <td>{e.machine ?? "—"}<div className="note">{new Date(e.created_at).toLocaleDateString()}</div></td>
                  <td>{e.snapshot.question ?? [...e.snapshot.turns].reverse().find((x) => x.role === "user")?.text ?? "—"}
                    {e.comment && <div className="note">“{e.comment}”</div>}</td>
                  <td className="num">{e.stars ?? "—"}</td>
                  <td><button className="btn" onClick={() => setOpen(e)}>{t("details")}</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {user?.role === "admin" && <GoldenSets />}
    </>
  );
}

function ExampleDetail({ ex, onClose }: { ex: Example; onClose: () => void }) {
  const { t } = useI18n();
  const [form, setForm] = useState({
    reference_answer: ex.reference_answer ?? "", rubric_notes: ex.rubric_notes ?? "", failure_cause: ex.failure_cause ?? "",
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const s = ex.snapshot;

  const save = async (status?: Status) => {
    setBusy(true); setError(null);
    try {
      await api(`/api/review/examples/${ex.id}`, { method: "PATCH", body: {
        reference_answer: form.reference_answer || null, rubric_notes: form.rubric_notes || null,
        failure_cause: form.failure_cause || null, ...(status ? { status } : {}),
      } });
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("errGeneric"));
    } finally { setBusy(false); }
  };

  return (
    <div className="review">
      <button className="btn" onClick={onClose}>← {t("backToQueue")}</button>
      <h3 style={{ margin: "12px 0 4px" }}>
        {ex.machine ?? "—"} · {ex.stars ? `${ex.stars}★` : ""}{" "}
        {ex.safety && <span className="tag tag--bad">{t("safety")}</span>}{" "}
        <span className={`tag ${ex.kind === "hard" ? "tag--warn" : "tag--ok"}`}>{t(`kind_${ex.kind}` as TKey)}</span>
      </h3>
      {ex.comment && <p><strong>{t("technicianComment")}:</strong> “{ex.comment}”</p>}

      <h4>{t("conversation")}</h4>
      <div className="review__turns">
        {s.turns.map((m, i) => (
          <div key={i} className={`review__turn review__turn--${m.role}`}>{m.text}</div>
        ))}
      </div>

      <h4>{t("answer")}</h4>
      <div className="review__answer md"><ReactMarkdown>{s.answer}</ReactMarkdown></div>
      {s.sources.length > 0 && (
        <div className="note">{t("sources")}: {s.sources.map((x) => `${x.title} · ${t("page")} ${x.page}`).join(" | ")}</div>
      )}
      {s.evidence && s.evidence.length > 0 && (
        <details style={{ marginTop: 6 }}>
          <summary className="note">{t("evidence")} ({s.evidence.length})</summary>
          <ul className="note">{s.evidence.map((h, i) => <li key={i}>{h.doc_label} · {t("page")} {h.page}</li>)}</ul>
        </details>
      )}
      {s.safety_review && s.safety_review.verdict !== "approve" && (
        <div className="note" style={{ marginTop: 6 }}>
          {t("verdict")}: <strong>{s.safety_review.verdict}</strong> — {s.safety_review.issues.join("; ")}
        </div>
      )}

      <div className="review__form">
        <label className="note" htmlFor="ref">{t("referenceAnswer")}</label>
        <textarea id="ref" className="field" rows={8} value={form.reference_answer}
                  onChange={(e) => setForm({ ...form, reference_answer: e.target.value })} />
        <label className="note" htmlFor="rubric">{t("rubricNotes")}</label>
        <textarea id="rubric" className="field" rows={3} value={form.rubric_notes}
                  onChange={(e) => setForm({ ...form, rubric_notes: e.target.value })} />
        <label className="note" htmlFor="cause">{t("failureCause")}</label>
        <select id="cause" className="field" value={form.failure_cause}
                onChange={(e) => setForm({ ...form, failure_cause: e.target.value })}>
          <option value="">—</option>
          {CAUSES.map((c) => <option key={c} value={c}>{t(`fc_${c}` as TKey)}</option>)}
        </select>
        {error && <div role="alert" className="tag tag--bad" style={{ whiteSpace: "normal" }}>{error}</div>}
        <div className="btnrow">
          <button className="btn" disabled={busy} onClick={() => void save()}>{t("save")}</button>
          <button className="btn" disabled={busy} onClick={() => void save("rejected")}>{t("reject")}</button>
          <button className="btn btn--primary" disabled={busy} onClick={() => void save("approved")}>{t("approve")}</button>
        </div>
      </div>
    </div>
  );
}

function GoldenSets() {
  const { t } = useI18n();
  const [sets, setSets] = useState<GoldenSetRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => { api<GoldenSetRow[]>("/api/review/golden-sets").then(setSets).catch(() => setSets([])); }, []);
  useEffect(load, [load]);

  const publish = async () => {
    setError(null);
    try { await api("/api/review/golden-sets", { method: "POST" }); load(); }
    catch (e) { setError(e instanceof ApiError ? e.message : t("errGeneric")); }
  };
  const download = async (v: number) => {
    const data = await api<unknown>(`/api/review/golden-sets/${v}`);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `golden-set-v${v}.json`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section style={{ marginTop: 20 }}>
      <h4 style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {t("goldenSets")}
        <button className="btn" onClick={() => void publish()}>{t("publishSet")}</button>
      </h4>
      {error && <div role="alert" className="tag tag--bad" style={{ whiteSpace: "normal" }}>{error}</div>}
      {sets.map((g) => (
        <div key={g.version} className="note" style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
          v{g.version} · {g.example_count} {t("examplesCount")} · {new Date(g.created_at).toLocaleDateString()}
          <button className="btn" style={{ minHeight: 30 }} onClick={() => void download(g.version)}>{t("download")}</button>
        </div>
      ))}
    </section>
  );
}
