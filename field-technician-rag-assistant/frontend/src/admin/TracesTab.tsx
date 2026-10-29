import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RunDetail, RunRow, StepRow } from "../api/types";
import { useI18n } from "../i18n";

const pill = (a: string | null) => `pill pill--${(a ?? "").toLowerCase()}`;
const ms = (n: number | null) => (n == null ? "" : n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${n} ms`);
const f = (n: number | null, d = 3) => (n == null ? "—" : n.toFixed(d));

function StepView({ s }: { s: StepRow }) {
  const { t } = useI18n();
  return (
    <li>
      <b>{s.node}</b> · {ms(s.latency_ms)}{s.model && s.model !== "fake-llm" ? ` · ${s.tokens_in ?? 0}→${s.tokens_out ?? 0} tok` : ""}
      {s.error && <div className="why" style={{ color: "var(--bad)" }}>error: {s.error}</div>}
      {s.rationale && <div className="why">{s.rationale}</div>}
      {s.hits.length > 0 && (
        <div className="tw">
          <table>
            <thead><tr><th>#</th><th>doc · p.</th><th className="num">vec</th><th className="num">fts</th><th className="num">rrf</th><th /></tr></thead>
            <tbody>
              {s.hits.map((h) => (
                <tr key={h.chunk_id + h.rank}>
                  <td>{h.rank}</td><td>{h.doc} · {h.page}</td>
                  <td className="num">{f(h.vec, 2)}</td><td className="num">{f(h.fts, 2)}</td><td className="num">{f(h.rrf, 3)}</td>
                  <td>{h.selected && <span className="tag tag--ok">used</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary className="note">{t("details")}</summary>
        {s.reasoning && <><div className="note"><b>{t("reasoning")}</b></div><pre className="trace-pre">{s.reasoning}</pre></>}
        {s.input != null && <><div className="note"><b>{t("input")}</b></div><pre className="trace-pre">{JSON.stringify(s.input, null, 2)}</pre></>}
        {s.output != null && <><div className="note"><b>{t("output")}</b></div><pre className="trace-pre">{JSON.stringify(s.output, null, 2)}</pre></>}
      </details>
    </li>
  );
}

export function TracesTab() {
  const { t } = useI18n();
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);

  useEffect(() => { api<RunRow[]>("/api/admin/traces?limit=100").then(setRuns).catch(() => setRuns([])); }, []);
  useEffect(() => {
    setDetail(null);
    if (sel) api<RunDetail>(`/api/admin/traces/${sel}`).then(setDetail).catch(() => setDetail(null));
  }, [sel]);

  return (
    <>
      <p className="note">{t("adminOnly")}</p>
      <div className="wf split-frame">
        <div className="split">
          <div>
            {runs === null && <div className="note">{t("loading")}</div>}
            {runs?.length === 0 && <div className="note">{t("noRuns")}</div>}
            {runs?.map((r) => (
              <button key={r.id} className="run" aria-selected={r.id === sel} onClick={() => setSel(r.id)}
                      style={{ display: "block", width: "100%", textAlign: "left", font: "inherit", color: "inherit" }}>
                <span className={pill(r.final_action)}>{r.final_action ?? "…"}</span> {new Date(r.started_at).toLocaleTimeString()}<br />
                <span className="note">{r.user} · {r.machine}</span>
              </button>
            ))}
          </div>
          <div>
            {!sel && <div className="note">{t("selectRun")}</div>}
            {sel && !detail && <div className="note">{t("loading")}</div>}
            {detail && (
              <>
                <p style={{ margin: "0 0 8px" }}><strong>{detail.question}</strong><br />
                  <span className="note">{t("latency")}: {ms(detail.latency_ms)} · LLM calls: {detail.llm_calls} · {detail.tokens_in}→{detail.tokens_out} tok</span></p>
                <ul className="timeline">{detail.steps.map((s) => <StepView key={s.seq} s={s} />)}</ul>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
