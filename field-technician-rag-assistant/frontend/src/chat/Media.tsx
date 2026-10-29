import { useEffect, useState } from "react";
import { authedObjectUrl } from "../api/client";
import type { BubbleAttachment } from "../api/types";
import { useI18n } from "../i18n";
import { IconClose, IconMic } from "../ui/Icon";

/** An image behind the API's auth: fetched as a blob, shown from an object URL, released on unmount. */
export function AuthImage({ path, alt, className, onOpen }: {
  path: string; alt: string; className?: string; onOpen?: (url: string) => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true, made: string | null = null;
    authedObjectUrl(path).then((u) => { made = u; if (live) setUrl(u); else URL.revokeObjectURL(u); })
      .catch(() => live && setFailed(true));
    return () => { live = false; if (made) URL.revokeObjectURL(made); };
  }, [path]);
  if (failed) return <span className="note">—</span>;
  if (!url) return <span className={`${className ?? ""} media--loading`} aria-busy="true" />;
  const img = <img src={url} alt={alt} className={className} />;
  return onOpen ? <button type="button" className="media__open" onClick={() => onOpen(url)} aria-label={alt}>{img}</button> : img;
}

/** Photos and voice-note markers on a bubble. */
export function Attachments({ items, onOpen }: { items: BubbleAttachment[]; onOpen: (url: string, alt: string) => void }) {
  const { t } = useI18n();
  if (!items.length) return null;
  return (
    <div className="media">
      {items.map((a, i) => a.kind === "audio" ? (
        <span key={i} className="tag tag--mute media__voice"><IconMic /> {t("voiceNote")}</span>
      ) : a.url ? (
        <button key={i} type="button" className="media__open" onClick={() => onOpen(a.url!, t("photo"))}
                aria-label={t("photo")}>
          <img src={a.url} alt={t("photo")} className="media__thumb" />
        </button>
      ) : (
        <AuthImage key={i} path={`/api/attachments/${a.id}`} alt={t("photo")} className="media__thumb"
                   onOpen={(u) => onOpen(u, t("photo"))} />
      ))}
    </div>
  );
}

/**
 * Full-size view, "as is". The image is shown at its real size inside a
 * scrolling pane, so a wiring sheet can be panned and read; "fit" shrinks it to
 * the screen for an overview.
 */
export function ImageViewer({ url, alt, caption, onClose }: {
  url: string; alt: string; caption?: string; onClose: () => void;
}) {
  const { t } = useI18n();
  const [fit, setFit] = useState(true);
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <div className="dim" role="presentation" onClick={onClose}>
      <div className="viewer" role="dialog" aria-modal="true" aria-label={alt} onClick={(e) => e.stopPropagation()}>
        <div className="viewer__bar">
          <span className="viewer__caption">{caption ?? alt}</span>
          <button type="button" className="btn" onClick={() => setFit((f) => !f)}>{fit ? t("actualSize") : t("fitScreen")}</button>
          <button type="button" className="ico" onClick={onClose} aria-label={t("close")} title={t("close")}><IconClose /></button>
        </div>
        <div className="viewer__pane">
          <img src={url} alt={alt} className={fit ? "viewer__img viewer__img--fit" : "viewer__img"} />
        </div>
      </div>
    </div>
  );
}
