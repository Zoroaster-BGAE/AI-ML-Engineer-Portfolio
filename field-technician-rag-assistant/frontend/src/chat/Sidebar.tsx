import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { SessionSummary } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useI18n } from "../i18n";
import { IconChat, IconClose, IconPin, IconPlus } from "../ui/Icon";

interface Props {
  activeId: string | null;
  onOpen: (s: SessionSummary) => void;
  onNew: () => void;
  onCloseDrawer: () => void;
  /** Bumped by the parent after a new chat starts, to refetch the list. */
  reloadKey: number;
}

// Matches history_max_sessions on the server: a full page means there may be more.
const PAGE = 30;

/**
 * Persistent chat list on the left, and a drawer on phones.
 *
 * Pinned chats are a separate group rather than just sorted first, because a list
 * that reorders itself when you pin something loses your place. Pinning also
 * exempts a chat from the 90-day expiry of unrated chats. Older chats are not
 * deleted when new ones are made; they page in with "Show older chats".
 */
export function Sidebar({ activeId, onOpen, onNew, onCloseDrawer, reloadKey }: Props) {
  const { t } = useI18n();
  const { user } = useAuth();
  const [items, setItems] = useState<SessionSummary[] | null>(null);
  const [older, setOlder] = useState<SessionSummary[]>([]);
  const [more, setMore] = useState(false);

  const load = useCallback(() => {
    api<SessionSummary[]>("/api/sessions")
      .then((l) => { setItems(l); setOlder([]); setMore(l.length >= PAGE); })
      .catch(() => setItems([]));
  }, []);
  useEffect(load, [load, reloadKey]);

  const loadOlder = () => {
    const offset = (items?.length ?? 0) + older.length;
    api<SessionSummary[]>(`/api/sessions?offset=${offset}`)
      .then((l) => { setOlder((o) => [...o, ...l]); setMore(l.length >= PAGE); })
      .catch(() => setMore(false));
  };

  const togglePin = async (s: SessionSummary) => {
    // Optimistic: pinning should feel instant, and the only way it fails is a
    // dead session, which the reload below corrects.
    setItems((prev) => prev?.map((x) => (x.id === s.id ? { ...x, pinned: !x.pinned } : x)) ?? prev);
    try {
      await api(`/api/sessions/${s.id}`, { method: "PATCH", body: { pinned: !s.pinned } });
    } finally {
      load();
    }
  };

  const pinned = items?.filter((s) => s.pinned) ?? [];
  const recent = items?.filter((s) => !s.pinned) ?? [];

  const row = (s: SessionSummary) => (
    <div key={s.id} className="srow" aria-current={s.id === activeId ? "true" : undefined}>
      <button className="srow__text" onClick={() => onOpen(s)} style={{ border: 0, background: "none", font: "inherit", color: "inherit", cursor: "pointer", textAlign: "left", padding: 0 }}>
        <span className="srow__title">{s.title ?? t("untitledChat")}</span>
        <span className="srow__sub">{s.machine_label ?? t("machineUnknown")}{s.rating ? ` · ★ ${s.rating.stars}` : ""}</span>
      </button>
      <button className="ico ico--sm srow__pin" aria-pressed={s.pinned}
              title={s.pinned ? t("unpin") : t("pin")} aria-label={s.pinned ? t("unpin") : t("pin")}
              onClick={() => void togglePin(s)}>
        <IconPin filled={s.pinned} />
      </button>
    </div>
  );

  return (
    <aside className="side" aria-label={t("chats")}>
      <div className="side__head">
        <span className="brand"><span className="brand__mark">E</span>{t("appName")}</span>
        <span className="spacer" />
        <button className="ico side__close" onClick={onCloseDrawer} aria-label={t("close")} title={t("close")}>
          <IconClose />
        </button>
      </div>

      <button className="newchat" onClick={onNew}>
        <IconPlus /> {t("newGuide")}
      </button>

      <div className="side__list">
        {items === null && <div className="note" style={{ padding: "8px" }}>{t("loading")}</div>}
        {items?.length === 0 && (
          <div className="note" style={{ padding: "8px", display: "flex", gap: 8, alignItems: "center" }}>
            <IconChat /> {t("noHistory")}
          </div>
        )}
        {pinned.length > 0 && <div className="side__group">{t("pinned")}</div>}
        {pinned.map(row)}
        {recent.length > 0 && pinned.length > 0 && <div className="side__group">{t("recent")}</div>}
        {recent.map(row)}
        {older.filter((s) => !s.pinned).map(row)}
        {more && (
          <button className="btn" style={{ margin: "8px", minHeight: 34 }} onClick={loadOlder}>{t("olderChats")}</button>
        )}
      </div>

      <div className="side__foot">
        <div className="note" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {user?.email}
        </div>
      </div>
    </aside>
  );
}
