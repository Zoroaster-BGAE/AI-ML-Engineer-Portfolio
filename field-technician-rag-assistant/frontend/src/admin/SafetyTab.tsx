import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { SafetyEventRow } from "../api/types";
import { useI18n, type TKey } from "../i18n";

const KIND_TAG: Record<SafetyEventRow["kind"], string> = {
  blocked: "tag--bad", amended: "tag--warn", tripwire: "tag--info", feedback: "tag--warn",
};

/** The lead technician's safety queue (design §3.6): blocked or amended answers, keyword hits, safety comments. */
export function SafetyTab() {
  const { t } = useI18n();
  const [resolved, setResolved] = useState(false);
  const [rows, setRows] = useState<SafetyEventRow[] | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});

  const load = useCallback(() => {
    setRows(null);
    api<SafetyEventRow[]>(`/api/review/safety-events?resolved=${resolved}`).then(setRows).catch(() => setRows([]));
  }, [resolved]);
  useEffect(load, [load]);

  const setState = (e: SafetyEventRow, value: boolean) =>
    api(`/api/review/safety-events/${e.id}`, { method: "PATCH", body: { resolved: value, note: notes[e.id] ?? e.note } })
      .then(load);

  return (
    <>
      <label className="note" style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 10 }}>
        <input type="checkbox" checked={resolved} onChange={(e) => setResolved(e.target.checked)} /> {t("showResolved")}
      </label>
      {!rows && <div className="note">{t("loading")}</div>}
      {rows?.length === 0 && <div className="note">{t("noSafety")}</div>}
      {rows?.map((e) => {
        const v = e.detail.verdict;
        const hits = [...(e.detail.question_hits ?? []), ...(e.detail.draft_hits ?? [])];
        return (
          <div key={e.id} className="card" style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <span className={`tag ${KIND_TAG[e.kind]}`}>{t(`sk_${e.kind}` as TKey)}</span>
              <strong>{e.machine ?? "—"}</strong>
              <span className="note">{new Date(e.created_at).toLocaleString()}</span>
            </div>
            {e.question && <p style={{ margin: "8px 0 4px" }}><strong>{t("question")}:</strong> {e.question}</p>}
            {v && (
              <div className="note">
                {t("verdict")}: <strong>{v.verdict}</strong>{v.issues.length > 0 && <> — {v.issues.join("; ")}</>}
                {v.warnings.length > 0 && <ul style={{ margin: "4px 0" }}>{v.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
              </div>
            )}
            {hits.length > 0 && <div className="note">{t("matched")}: {hits.join(", ")}</div>}
            {e.detail.comment && <p style={{ margin: "6px 0" }}><strong>{t("technicianComment")}</strong> ({e.detail.stars}★): “{e.detail.comment}”</p>}
            {(e.detail.draft || e.detail.answer) && (
              <details style={{ marginTop: 6 }}>
                <summary className="note">{e.detail.draft ? t("draftNeverShown") : t("answer")}</summary>
                <div className="review__answer" style={{ whiteSpace: "pre-wrap" }}>{e.detail.draft ?? e.detail.answer}</div>
              </details>
            )}
            <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
              <input className="field" style={{ flex: 1, minWidth: 200 }} placeholder={t("note")} aria-label={t("note")}
                     value={notes[e.id] ?? e.note ?? ""} onChange={(x) => setNotes({ ...notes, [e.id]: x.target.value })} />
              <button className="btn" onClick={() => void setState(e, !e.resolved)}>
                {e.resolved ? t("reopenEvent") : t("markResolved")}
              </button>
            </div>
          </div>
        );
      })}
    </>
  );
}
