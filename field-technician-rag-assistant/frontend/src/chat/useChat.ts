import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamMessage } from "../api/client";
import type { Bubble, SessionDetail, StreamEvent } from "../api/types";
import { useI18n, type TKey } from "../i18n";

const STATUS_KEYS = ["identify", "understand", "route", "search", "assess", "expand", "answer", "safety"];
let seq = 0;
const key = () => `b${++seq}`;

export type TurnAction = "ANSWER" | "ASK" | "ESCALATE" | "ERROR" | null;

export function useChat(sessionId: string, onTurnEnd?: (action: TurnAction) => void,
                        onScope?: (machineLabel: string) => void) {
  const { t } = useI18n();
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const abort = useRef<AbortController | null>(null);
  // status lines are generic (no manual content); the UI language is applied here
  const statusText = useCallback((k: string, fallback: string) => (STATUS_KEYS.includes(k) ? t(`st_${k}` as TKey) : fallback), [t]);

  // (Re)open a session: restore messages, the scope is already fixed server-side, and any pending question.
  useEffect(() => {
    let live = true;
    setLoading(true); setBubbles([]);
    api<SessionDetail>(`/api/sessions/${sessionId}`).then((d) => {
      if (!live) return;
      const list: Bubble[] = d.messages.map((m) => ({
        key: m.id, role: m.role, text: m.text, sources: m.sources,
        messageId: m.id,
        attachments: m.attachments.map((a) => ({ id: a.id, kind: a.kind })),
        figures: m.figures,
      }));
      if (d.pending_question && list.length) list[list.length - 1].question = d.pending_question;
      setBubbles(list);
    }).finally(() => live && setLoading(false));
    return () => { live = false; abort.current?.abort(); };
  }, [sessionId]);

  const send = useCallback(async (text: string, optionIndex?: number, extras?: { images?: Blob[]; audioId?: string }) => {
    const images = extras?.images ?? [];
    if (busy || (!text.trim() && images.length === 0)) return;
    const live = key();
    setBusy(true);
    // The photos shown in the bubble are the local copies; history reloads fetch the stored ones.
    const attachments = [
      ...images.map((b) => ({ kind: "image" as const, url: URL.createObjectURL(b) })),
      ...(extras?.audioId ? [{ kind: "audio" as const, id: extras.audioId }] : []),
    ];
    setBubbles((b) => [
      ...b.map((x) => ({ ...x, question: null })),   // chips only ever live on the latest question
      { key: key(), role: "user", text, attachments },
      { key: live, role: "assistant", text: "", status: t("st_understand") },
    ]);
    const patch = (fn: (b: Bubble) => Bubble) => setBubbles((all) => all.map((b) => (b.key === live ? fn(b) : b)));
    let action: TurnAction = null;
    const onEvent = (e: StreamEvent) => {
      switch (e.event) {
        case "status": patch((b) => ({ ...b, status: statusText(e.data.key, e.data.text) })); break;
        case "token": patch((b) => ({ ...b, text: b.text + e.data, status: null })); break;
        case "question": patch((b) => ({ ...b, text: e.data.question, question: e.data, status: null })); break;
        case "scope": onScope?.(e.data.machine_label); break;
        case "done":
          action = e.data.action;
          patch((b) => ({
            ...b, status: null, sources: e.data.sources ?? b.sources, messageId: e.data.message_id,
            figures: e.data.figures ?? [],
            ...(e.data.action === "ERROR" ? { text: e.data.message ?? t("errGeneric"), error: true } : {}),
          }));
          break;
      }
    };
    abort.current = new AbortController();
    try {
      await streamMessage(sessionId, { text, option_index: optionIndex, images, audio_id: extras?.audioId },
                          onEvent, abort.current.signal);
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        // Stopping used to leave the bubble exactly as it was -- still showing
        // "Reading your question…" for ever, because the abort branch cleared
        // nothing. The status has to go, and a turn stopped before any text
        // arrived needs to say so, or the screen just shows an empty reply.
        patch((b) => ({
          ...b,
          status: null,
          ...(b.text ? {} : { text: t("stopped"), stopped: true }),
        }));
      } else {
        patch((b) => ({ ...b, status: null, text: b.text || t("errGeneric"), error: true }));
      }
    } finally {
      setBusy(false);
      // The server titles a session from its first message, so the sidebar's
      // copy is stale the moment a turn finishes -- it would keep saying "New
      // chat" for a chat that now has a name. Refreshing after every turn also
      // reorders the list by recency, which is what the sidebar claims to show.
      onTurnEnd?.(action);
    }
  }, [busy, sessionId, t, statusText, onTurnEnd, onScope]);

  /**
   * Cut the current turn short.
   *
   * Aborting the fetch is enough to stop the server too: the endpoint streams
   * through sse_starlette's EventSourceResponse, which watches for client
   * disconnect and cancels the generator. So this is not only a local cosmetic
   * stop -- the model stops being asked for tokens.
   *
   * What it leaves behind is the question, saved without an answer under it,
   * because the assistant message is persisted at the end of a turn. That is
   * existing behaviour -- the same thing happens when a session is switched
   * mid-answer -- and changing it would be server-side work.
   */
  const stop = useCallback(() => {
    abort.current?.abort();
  }, []);

  return { bubbles, busy, loading, send, stop };
}
