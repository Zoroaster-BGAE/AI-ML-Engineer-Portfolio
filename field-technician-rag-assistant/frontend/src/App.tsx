import { useCallback, useEffect, useRef, useState } from "react";
import { AdminConsole } from "./admin/AdminConsole";
import { api } from "./api/client";
import type { SessionSummary } from "./api/types";
import { useAuth } from "./auth/AuthContext";
import { Login } from "./auth/AuthScreens";
import { ChatView } from "./chat/ChatView";
import { RateConversation } from "./chat/RateConversation";
import { Sidebar } from "./chat/Sidebar";
import { useI18n } from "./i18n";
import { Controls } from "./ui/Controls";
import { IconMenu } from "./ui/Icon";

const LAST_KEY = "assistant.lastSession";
const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  del: (k: string) => { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

export default function App() {
  const { user, loading } = useAuth();
  const { t } = useI18n();
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [drawer, setDrawer] = useState(false);
  const [view, setView] = useState<"chat" | "admin">("chat");
  const [reloadKey, setReloadKey] = useState(0);

  const opened = useCallback((s: SessionSummary) => {
    setSession(s); setDrawer(false); setView("chat");
  }, []);

  // A new chat starts without a machine: the assistant asks which one. The server
  // hands back the current empty chat instead of making another, so this is safe
  // to call on every "New chat" and on every load with nothing to reopen.
  const newChat = useCallback(async () => {
    const s = await api<SessionSummary>("/api/sessions", { method: "POST" });
    opened(s);
    setReloadKey((k) => k + 1);
  }, [opened]);

  // The conversation rating: open for the current chat, and what to do once it closes.
  const [rating, setRating] = useState<{ next?: () => void } | null>(null);
  const asked = useRef(new Set<string>());

  // Leaving a chat that has answers and no rating asks for one -- once per chat, so
  // someone who skips is not asked again every time they switch back and forth.
  const leave = useCallback((next: () => void) => {
    if (session && session.has_answers && !session.rating && !asked.current.has(session.id)) {
      asked.current.add(session.id);
      setRating({ next });
    } else {
      next();
    }
  }, [session]);

  // Reopen the last chat if it still exists (FR-15); otherwise start one.
  useEffect(() => {
    if (!user) return;
    const last = store.get(LAST_KEY);
    (last ? api<SessionSummary[]>("/api/sessions").then((l) => l.find((s) => s.id === last) ?? null) : Promise.resolve(null))
      .catch(() => null)
      .then((s) => (s ? setSession(s) : newChat()))
      .catch(() => { /* the chat view shows an error on the first send */ });
  }, [user, newChat]);

  useEffect(() => { if (session) store.set(LAST_KEY, session.id); }, [session]);
  // Only once auth has settled: while a reload is still restoring the session `user` is briefly null, and
  // clearing then forgot the last chat on every reload. The view resets too, so an admin's console is not
  // what the next person to sign in on this browser lands on.
  useEffect(() => {
    if (!loading && !user) { setSession(null); setView("chat"); store.del(LAST_KEY); }
  }, [user, loading]);

  // Escape closes the drawer -- expected of any overlay, and one listener.
  useEffect(() => {
    if (!drawer) return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") setDrawer(false); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [drawer]);

  if (loading) return null;
  if (!user) return <Login />;

  return (
    <div className="app" data-drawer={drawer ? "open" : undefined}>
      <Sidebar activeId={session?.id ?? null}
               onOpen={(s) => (s.id === session?.id ? opened(s) : leave(() => opened(s)))}
               onNew={() => leave(() => void newChat())}
               onCloseDrawer={() => setDrawer(false)}
               reloadKey={reloadKey} />

      {/* A button, not a div: dismissing an overlay by tapping outside has to be
          reachable from a keyboard too. */}
      {drawer && <button className="scrim" aria-label={t("close")} onClick={() => setDrawer(false)} />}

      <div className="main">
        {/* Shown at every width, and in both views, so the controls never need
            scrolling to. The hamburger only appears where the sidebar is a drawer. */}
        <div className="topbar">
          <button className="ico topbar__menu" onClick={() => setDrawer(true)} aria-label={t("chats")} title={t("chats")}>
            <IconMenu />
          </button>
          <span className="brand"><span className="brand__mark">E</span>{t("appName")}</span>
          <span className="spacer" />
          <Controls view={view} onView={(v) => { setView(v); setDrawer(false); }} />
        </div>

        {view === "chat" ? (
          <ChatView
            key={session?.id ?? "none"}
            sessionId={session?.id ?? null}
            scope={session?.machine_label ?? null}
            onScope={(label) => setSession((s) => (s ? { ...s, machine_label: label } : s))}
            onTurnEnd={(action) => {
              setReloadKey((k) => k + 1);
              if (action === "ANSWER" || action === "ESCALATE") setSession((s) => (s ? { ...s, has_answers: true } : s));
            }}
            onFinish={session?.has_answers ? () => setRating({}) : undefined}
            rated={session?.rating?.stars ?? null}
          />
        ) : (
          <AdminConsole />
        )}
      </div>

      {rating && session && (
        <RateConversation sessionId={session.id} initial={session.rating} onDone={(saved) => {
          if (saved) { setSession((s) => (s ? { ...s, rating: saved } : s)); setReloadKey((k) => k + 1); }
          const next = rating.next;
          setRating(null);
          next?.();
        }} />
      )}
    </div>
  );
}
