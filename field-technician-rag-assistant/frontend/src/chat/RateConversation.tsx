import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Rating } from "../api/types";
import { useI18n } from "../i18n";
import { IconClose } from "../ui/Icon";

/**
 * Rate the whole conversation: 1-5 stars and an optional comment.
 *
 * Opened by "Finish & rate", or once when the technician leaves an unrated chat
 * that has answers (new chat, or another chat from the sidebar). Stars alone are a
 * complete rating; "Skip" leaves without one. Stars are 40px targets: this is
 * rated at the machine, often with gloves on.
 */
export function RateConversation({ sessionId, initial, onDone }: {
  sessionId: string;
  initial: Rating | null;
  /** Called with the saved rating, or null when skipped. */
  onDone: (saved: Rating | null) => void;
}) {
  const { t } = useI18n();
  const [stars, setStars] = useState(initial?.stars ?? 0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState(initial?.comment ?? "");
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onDone(null); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onDone]);

  const save = async () => {
    if (!stars) return;
    setSaving(true); setFailed(false);
    try {
      onDone(await api<Rating>(`/api/sessions/${sessionId}/rating`, {
        method: "PUT", body: { stars, comment: comment.trim() || null } }));
    } catch { setFailed(true); setSaving(false); }
  };

  return (
    <div className="dim" role="presentation" onClick={() => onDone(null)}>
      <div className="modal rate" role="dialog" aria-modal="true" aria-labelledby="rate-title"
           onClick={(e) => e.stopPropagation()}>
        <div className="source__head">
          <h5 id="rate-title">{t("ratePrompt")}</h5>
          <button type="button" className="ico" onClick={() => onDone(null)} aria-label={t("close")} title={t("close")}>
            <IconClose />
          </button>
        </div>
        <div className="rating__stars" role="radiogroup" aria-label={t("rateConversation")} onMouseLeave={() => setHover(0)}>
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} type="button" role="radio" aria-checked={stars === n} aria-label={`${n} / 5`}
                    title={`${n} / 5`} className={`star${(hover || stars) >= n ? " star--on" : ""}`}
                    onMouseEnter={() => setHover(n)} onClick={() => setStars(n)}>★</button>
          ))}
        </div>
        <textarea className="field" rows={3} maxLength={1000} value={comment}
                  onChange={(e) => setComment(e.target.value)} placeholder={t("commentPlaceholder")}
                  aria-label={t("addComment")} />
        {failed && <div role="alert" className="note rating__error">{t("ratingFailed")}</div>}
        <div className="btnrow">
          <button type="button" className="btn" onClick={() => onDone(null)}>{t("skip")}</button>
          <button type="button" className="btn btn--primary" disabled={!stars || saving} onClick={() => void save()}>
            {t("save")}
          </button>
        </div>
      </div>
    </div>
  );
}
