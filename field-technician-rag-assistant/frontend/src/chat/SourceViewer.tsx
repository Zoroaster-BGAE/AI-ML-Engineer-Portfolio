import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { SourceExcerpt } from "../api/types";
import { useI18n } from "../i18n";
import { IconClose } from "../ui/Icon";
import { AuthImage } from "./Media";

/**
 * The exact manual passage behind reference [n] of an answer, so the technician
 * (or an admin checking the answer) can verify it. The server only serves what
 * this answer actually cited.
 */
export function SourceViewer({ messageId, n, onClose, onOpenImage }: {
  messageId: string; n: number; onClose: () => void; onOpenImage: (url: string, alt: string, caption?: string) => void;
}) {
  const { t } = useI18n();
  const [src, setSrc] = useState<SourceExcerpt | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<SourceExcerpt>(`/api/messages/${messageId}/sources/${n}`).then(setSrc)
      .catch((e) => setError(e instanceof ApiError && e.status !== 500 ? e.message : t("errGeneric")));
  }, [messageId, n, t]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  const caption = src ? `${src.title} · ${t("page")} ${src.page}` : "";
  return (
    <div className="dim" role="presentation" onClick={onClose}>
      <div className="modal source" role="dialog" aria-modal="true" aria-labelledby="src-title"
           onClick={(e) => e.stopPropagation()}>
        <div className="source__head">
          <h5 id="src-title">[{n}] {src ? src.title : t("loading")}</h5>
          <button type="button" className="ico" onClick={onClose} aria-label={t("close")} title={t("close")}><IconClose /></button>
        </div>
        {error && <div role="alert" className="note rating__error">{error}</div>}
        {src && (
          <>
            <div className="note">
              {src.doc_type ? `${src.doc_type} · ` : ""}{t("page")} {src.page}{src.section ? ` · ${src.section}` : ""}
            </div>
            {src.figure_id && (
              <AuthImage path={`/api/figures/${src.figure_id}`} alt={caption} className="source__figure"
                         onOpen={(u) => onOpenImage(u, caption, caption)} />
            )}
            <div className="source__text">
              {src.excerpts.map((x, i) => <p key={i}>{x}</p>)}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
