import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Gap } from "../api/types";
import { useI18n } from "../i18n";

export function GapsTab() {
  const { t } = useI18n();
  const [gaps, setGaps] = useState<Gap[] | null>(null);
  const load = useCallback(() => { api<Gap[]>("/api/admin/gaps").then(setGaps).catch(() => setGaps([])); }, []);
  useEffect(load, [load]);
  if (!gaps) return <div className="note">{t("loading")}</div>;
  if (gaps.length === 0) return <div className="note">{t("noGaps")}</div>;
  return (
    <div className="tw">
      <table>
        <thead><tr><th>{t("machine")}</th><th>{t("question")}</th><th>{t("missing")}</th><th /></tr></thead>
        <tbody>
          {gaps.map((g) => (
            <tr key={g.id} style={g.resolved ? { opacity: 0.55 } : undefined}>
              <td>{g.machine}<div className="note">{new Date(g.created_at).toLocaleString()}</div></td>
              <td>{g.question}</td><td>{g.missing}</td>
              <td><button className="btn" style={{ minHeight: 36 }}
                          onClick={() => api(`/api/admin/gaps/${g.id}?resolved=${!g.resolved}`, { method: "PATCH" }).then(load)}>
                {g.resolved ? t("reopen") : t("resolve")}</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
