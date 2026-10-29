import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { LibClient, LibDoc } from "../api/types";
import { useI18n } from "../i18n";

const DOC_TYPES = ["operating_manual", "instructions", "wiring_diagram", "flow_diagram", "parameters", "parts", "datasheet", "mixed", "other"];
const statusTag = (s: string) => (s === "indexed" ? "tag--ok" : s === "failed" ? "tag--bad" : s === "skipped" ? "tag--mute" : "tag--warn");

function DocRow({ d }: { d: LibDoc }) {
  const { t } = useI18n();
  const [type, setType] = useState(d.doc_type);
  const [comp, setComp] = useState(d.component ?? "");
  const [saved, setSaved] = useState(false);
  const dirty = type !== d.doc_type || comp !== (d.component ?? "");
  return (
    <tr>
      <td>{d.title}</td>
      <td><select className="field" style={{ minHeight: 36 }} value={type} onChange={(e) => { setType(e.target.value); setSaved(false); }} aria-label={t("type")}>
        {DOC_TYPES.map((x) => <option key={x}>{x}</option>)}</select></td>
      <td><input className="field" style={{ minHeight: 36 }} value={comp} onChange={(e) => { setComp(e.target.value); setSaved(false); }} aria-label={t("component")} /></td>
      <td><span className={`tag ${statusTag(d.ingest_status)}`}>{d.ingest_status}</span>
        {d.ingest_note && <div className="note">{d.ingest_note}</div>}</td>
      <td className="num">{d.page_count ?? "—"} / {d.chunk_count}</td>
      <td><button className="btn" style={{ minHeight: 36 }} disabled={!dirty}
                  onClick={() => api(`/api/admin/library/documents/${d.id}`, { method: "PATCH", body: { doc_type: type, component: comp } }).then(() => setSaved(true))}>
        {saved ? "✓" : t("retag")}</button></td>
    </tr>
  );
}

export function LibraryTab() {
  const { t } = useI18n();
  const [lib, setLib] = useState<LibClient[] | null>(null);
  useEffect(() => { api<LibClient[]>("/api/admin/library").then(setLib).catch(() => setLib([])); }, []);
  if (!lib) return <div className="note">{t("loading")}</div>;
  return (
    <>
      {lib.map((c) => (
        <section key={c.id}>
          <h3 style={{ margin: "12px 0 6px" }}>{c.name} <span className="tag tag--mute">{c.slug}</span></h3>
          {c.machines.map((m) => (
            <details key={m.id} className="lib-machine">
              <summary>
                <strong>{m.label}</strong> <span className="tag tag--info">{m.kind}</span>{" "}
                <span className="note">{m.documents.filter((d) => d.ingest_status === "indexed").length}/{m.documents.length} {t("documents")}</span>
              </summary>
              <div className="tw">
                <table>
                  <thead><tr><th>{t("title")}</th><th>{t("type")}</th><th>{t("component")}</th><th>{t("status")}</th><th className="num">pp / {t("chunks")}</th><th /></tr></thead>
                  <tbody>{m.documents.map((d) => <DocRow key={d.id} d={d} />)}</tbody>
                </table>
              </div>
            </details>
          ))}
        </section>
      ))}
    </>
  );
}
