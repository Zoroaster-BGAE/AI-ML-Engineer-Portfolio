import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import { uploadAudio } from "../api/client";
import type { Bubble } from "../api/types";
import { useI18n } from "../i18n";
import { IconCheck, IconClose, IconCopy, IconMic, IconPaperclip, IconSend, IconStop } from "../ui/Icon";
import { Attachments, AuthImage, ImageViewer } from "./Media";
import { SourceViewer } from "./SourceViewer";
import { recorderMimeType, shrinkPhoto } from "./mediaUtils";
import { useChat, type TurnAction } from "./useChat";

/**
 * Copy a finished answer.
 *
 * On answers only. Selecting with the mouse is how text is taken from anywhere,
 * and that works; this is the shortcut for taking a whole answer at once, which
 * is the thing a technician does with it -- into a job note, or to a colleague.
 *
 * The rejection is caught: some in-app browsers refuse clipboard access outright,
 * and a button that silently does nothing is worse than no button. On refusal the
 * answer is selected instead, leaving the copy finishable by hand.
 */
function CopyAnswer({ text }: { text: string }) {
  const { t } = useI18n();
  const [done, setDone] = useState(false);

  const copy = async (e: React.MouseEvent<HTMLButtonElement>) => {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1600);
    } catch {
      const body = e.currentTarget.closest(".bubble")?.querySelector(".md");
      if (!body) return;
      const range = document.createRange();
      range.selectNodeContents(body);
      const sel = window.getSelection();
      sel?.removeAllRanges();
      sel?.addRange(range);
    }
  };

  return (
    <div className="bubble__tools">
      <button type="button" className="ico ico--sm" onClick={(e) => void copy(e)}
              title={done ? t("copied") : t("copy")} aria-label={done ? t("copied") : t("copy")}>
        {done ? <IconCheck /> : <IconCopy />}
      </button>
    </div>
  );
}

type OpenImage = (url: string, alt: string, caption?: string) => void;

/** [n] becomes a link to "#src-n", which the renderer below turns into a button that opens the passage. */
const linkCitations = (text: string) => text.replace(/\[(\d{1,2})\]/g, (_, n) => `[\\[${n}\\]](#src-${n})`);

function AnswerText({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  const { t } = useI18n();
  return (
    <div className="md">
      <ReactMarkdown components={{
        a: ({ href, children }) => href?.startsWith("#src-") && onCite ? (
          <button type="button" className="cite" onClick={() => onCite(Number(href.slice(5)))}
                  title={t("openSource")} aria-label={`${t("openSource")} ${href.slice(5)}`}>{children}</button>
        ) : <a href={href} target="_blank" rel="noreferrer">{children}</a>,
      }}>{onCite ? linkCitations(text) : text}</ReactMarkdown>
    </div>
  );
}

function BubbleView({ b, busy, onChip, onOpenImage, onCite }: {
  b: Bubble; busy: boolean; onChip: (label: string, index: number) => void; onOpenImage: OpenImage;
  onCite: (messageId: string, n: number) => void;
}) {
  const { t } = useI18n();
  // References open only once the answer is saved and numbered (not while streaming, not on old answers).
  const cite = b.messageId && !b.status && b.sources?.some((s) => s.n) ? (n: number) => onCite(b.messageId!, n) : undefined;
  return (
    <div className={`bubble bubble--${b.role}${b.stopped ? " bubble--stopped" : ""}`}
         role={b.error ? "alert" : undefined}>
      {b.status && !b.text && <div className="status" aria-live="polite">{b.status}</div>}
      {b.attachments && <Attachments items={b.attachments} onOpen={onOpenImage} />}
      {b.role === "assistant" ? (b.text && <AnswerText text={b.text} onCite={cite} />) : b.text}
      {/* Drawings the answer cites, shown as they are: a wiring sheet says more than any description of it. */}
      {!b.status && b.figures && b.figures.length > 0 && (
        <div className="media figures">
          {b.figures.map((f) => {
            const caption = `${t("wiringDiagram")}: ${f.title} · ${t("page")} ${f.page}`;
            return (
              <figure key={f.figure_id} className="figure">
                <AuthImage path={`/api/figures/${f.figure_id}`} alt={caption} className="figure__img"
                           onOpen={(u) => onOpenImage(u, caption, caption)} />
                <figcaption className="note">{f.n ? `[${f.n}] ` : ""}{f.title} · {t("page")} {f.page}</figcaption>
              </figure>
            );
          })}
        </div>
      )}
      {b.question && (
        <>
          <div className="chips" role="group" aria-label={b.question.question}>
            {b.question.options.map((o, i) => (
              <button key={i} type="button" disabled={busy}
                      className={`chip${i === b.question!.options.length - 1 ? " chip--ghost" : ""}`}
                      onClick={() => onChip(o.label, i)}>{o.label}</button>
            ))}
          </div>
          {b.question.allow_free_text && <div className="note" style={{ marginTop: 6 }}>{t("orOwnWords")}</div>}
        </>
      )}
      {/* Answers only, and only once finished -- there is nothing complete to
          copy from a half-streamed reply, a stopped one, or an error. */}
      {b.role === "assistant" && b.text && !b.status && !b.error && !b.stopped && <CopyAnswer text={b.text} />}
      {b.sources && b.sources.length > 0 && (
        <div className="sources">
          {t("sources")}: {b.sources.map((s, i) => {
            const label = `${s.title}${s.doc_type ? ` · ${s.doc_type}` : ""} · ${t("page")} ${s.page}`;
            return (
              <span key={i}>{i > 0 && " | "}
                {cite && s.n ? (
                  <button type="button" className="source-link" onClick={() => cite(s.n!)} title={t("openSource")}>
                    [{s.n}] {label}
                  </button>
                ) : label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

interface Props {
  sessionId: string | null;
  /** The machine this chat is about; null until the assistant has identified it. */
  scope: string | null;
  onScope: (machineLabel: string) => void;
  /** Called when a turn finishes, so the sidebar can pick up the new title and the app knows there are answers. */
  onTurnEnd?: (action: TurnAction) => void;
  /** Shown once there is something to rate: opens the conversation rating. */
  onFinish?: () => void;
  rated?: number | null;
}

export function ChatView({ sessionId, scope, onScope, onTurnEnd, onFinish, rated }: Props) {
  const { t, lang } = useI18n();
  const { bubbles, busy, loading, send, stop } = useChat(sessionId ?? "", onTurnEnd, onScope);
  const [draft, setDraft] = useState("");
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [photos, setPhotos] = useState<{ blob: Blob; url: string }[]>([]);
  const [audioId, setAudioId] = useState<string | null>(null);
  const [recorder, setRecorder] = useState<MediaRecorder | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [viewing, setViewing] = useState<{ url: string; alt: string; caption?: string } | null>(null);
  const [citing, setCiting] = useState<{ messageId: string; n: number } | null>(null);
  const openImage: OpenImage = (url, alt, caption) => setViewing({ url, alt, caption });

  // A voice note is transcribed first and the text lands in the box to be checked:
  // a misheard alarm code must be corrected before the assistant acts on it.
  async function transcribe(audio: Blob) {
    if (!sessionId) return;
    setTranscribing(true); setMediaError(null);
    try {
      const r = await uploadAudio(sessionId, audio, lang);
      setDraft((d) => (d.trim() ? `${d.trim()} ${r.transcript}` : r.transcript));
      setAudioId(r.audio_id);
      box.current?.focus();
    } catch { setMediaError(t("transcribeFailed")); }
    finally { setTranscribing(false); }
  }

  async function addFiles(files: FileList | null) {
    if (!files) return;
    setMediaError(null);
    for (const f of Array.from(files)) {
      if (f.type.startsWith("audio/")) { await transcribe(f); continue; }
      if (!f.type.startsWith("image/")) { setMediaError(t("unsupportedFile")); continue; }
      const blob = await shrinkPhoto(f);
      setPhotos((p) => (p.length >= 4 ? p : [...p, { blob, url: URL.createObjectURL(blob) }]));
    }
  }

  async function toggleRecording() {
    if (recorder) { recorder.stop(); return; }
    setMediaError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = recorderMimeType();
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const parts: Blob[] = [];
      rec.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop());
        setRecorder(null);
        void transcribe(new Blob(parts, { type: rec.mimeType || mimeType || "audio/webm" }));
      };
      rec.start();
      setRecorder(rec);
    } catch { setMediaError(t("micDenied")); }
  }

  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [bubbles]);

  // Grow the textarea with its content instead of scrolling inside one line. Done
  // in a layout effect so the height is set before paint -- in a plain effect the
  // box visibly jumps as you type past the first line.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [draft]);

  const canSend = !!sessionId && !transcribing && !recorder && (!!draft.trim() || photos.length > 0);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    if (!canSend || busy) return;
    void send(draft.trim(), undefined, { images: photos.map((p) => p.blob), audioId: audioId ?? undefined });
    setDraft(""); setPhotos([]); setAudioId(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter makes a newline. Skipped while composing, or an
    // Enter that accepts a Greek or German IME suggestion would send the message.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  const isEmpty = !loading && bubbles.length === 0;

  // Fixed once the assistant has identified the machine; a different machine is a new chat.
  const scopeBar = (
    <div className="scopebar">
      <span className="scope" aria-disabled="true" title={scope ?? t("machineUnknown")}>
        <span>{scope ?? t("machineUnknown")}</span>
      </span>
      {onFinish && (
        <button type="button" className="btn scopebar__rate" onClick={onFinish} disabled={busy}>
          {rated ? `★ ${rated}/5 · ${t("changeRating")}` : t("finishRate")}
        </button>
      )}
    </div>
  );

  // Declared once and placed in both branches, so the two layouts cannot drift.
  const composer = (
    <div className="composer-wrap">
      {(photos.length > 0 || transcribing || recorder || audioId || mediaError) && (
        <div className="composer__media">
          {photos.map((p, i) => (
            <span key={p.url} className="composer__photo">
              <img src={p.url} alt={t("photo")} />
              <button type="button" className="ico ico--sm" aria-label={t("removePhoto")} title={t("removePhoto")}
                      onClick={() => { URL.revokeObjectURL(p.url); setPhotos((all) => all.filter((_, j) => j !== i)); }}>
                <IconClose />
              </button>
            </span>
          ))}
          {recorder && <span className="tag tag--bad">{t("recording")}</span>}
          {transcribing && <span className="note">{t("transcribing")}</span>}
          {audioId && !transcribing && <span className="note">{t("checkTranscript")}</span>}
          {mediaError && <span role="alert" className="note rating__error">{mediaError}</span>}
        </div>
      )}
      <form className="composer" onSubmit={submit}>
        <input ref={fileInput} type="file" accept="image/*,audio/*" multiple hidden
               onChange={(e) => { void addFiles(e.target.files); e.target.value = ""; }} />
        <button type="button" className="composer__tool" onClick={() => fileInput.current?.click()}
                disabled={!sessionId || busy || photos.length >= 4} aria-label={t("attach")} title={t("attach")}>
          <IconPaperclip />
        </button>
        <button type="button" className={`composer__tool${recorder ? " composer__tool--on" : ""}`}
                onClick={() => void toggleRecording()} disabled={!sessionId || busy || transcribing}
                aria-pressed={!!recorder} aria-label={recorder ? t("stopRecording") : t("record")}
                title={recorder ? t("stopRecording") : t("record")}>
          <IconMic />
        </button>
        <textarea ref={box} className="composer__input" rows={1} value={draft}
                  onChange={(e) => setDraft(e.target.value)} onKeyDown={onKeyDown}
                  placeholder={t("typeMessage")} aria-label={t("typeMessage")}
                  autoComplete="off" spellCheck={false} disabled={!sessionId} />
        {/* While a turn is streaming the same button stops it. An answer takes
            tens of seconds, so without this a mistyped question has to be watched
            to the end before anything else can be asked. type="button" matters:
            inside a form, a default submit would send the draft. */}
        {busy ? (
          <button type="button" className="composer__send composer__send--stop" onClick={stop}
                  aria-label={t("stop")} title={t("stop")}>
            <IconStop />
          </button>
        ) : (
          <button className="composer__send" disabled={!canSend}
                  aria-label={t("send")} title={t("send")}>
            <IconSend />
          </button>
        )}
      </form>
      <div className="composer__hint">{t("composerHint")}</div>
    </div>
  );

  // An empty chat puts the greeting and the box together in the middle of the
  // screen; once there are messages the box drops to the bottom and the
  // conversation scrolls above it. Two layouts rather than one, because a
  // composer pinned to the bottom of an empty screen leaves the greeting
  // stranded at the top with nothing between them.
  if (isEmpty) {
    return (
      <div className="chat chat--empty">
        <div className="chat-col">
          {scopeBar}
          <div className="hero">
            <span className="brand__mark hero__mark">E</span>
            <h2>{t("heroTitle")}</h2>
            <p>{t("heroBody")}</p>
          </div>
          {composer}
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat-scroll">
        <div className="chat-col">
          {scopeBar}
          {loading && <div className="note">{t("loading")}</div>}
          {bubbles.map((b) => <BubbleView key={b.key} b={b} busy={busy} onChip={(label, i) => void send(label, i)}
                                          onOpenImage={openImage}
                                          onCite={(messageId, n) => setCiting({ messageId, n })} />)}
          <div ref={end} />
        </div>
      </div>
      {composer}
      {citing && <SourceViewer {...citing} onClose={() => setCiting(null)} onOpenImage={openImage} />}
      {viewing && <ImageViewer {...viewing} onClose={() => setViewing(null)} />}
    </div>
  );
}
