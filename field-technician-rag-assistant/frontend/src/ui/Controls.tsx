import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { useI18n } from "../i18n";
import { IconChat, IconMoon, IconPower, IconSliders, IconSun } from "./Icon";

type Theme = "light" | "dark";

/**
 * One button, two states.
 *
 * There used to be a third, "follow the device", which meant reaching dark from
 * light took two presses and the button had no single meaning. The device is
 * still respected — it decides the theme on a first visit — but it is not a stop
 * on the cycle. Once someone chooses, the choice sticks.
 */
function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem("assistant.theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch { /* private mode */ }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("assistant.theme", theme); } catch { /* private mode */ }
  }, [theme]);

  return [theme, () => setTheme((x) => (x === "dark" ? "light" : "dark"))] as const;
}

/**
 * Admin, language, theme and sign-out.
 *
 * These live in the top bar rather than at the foot of the sidebar, and the top
 * bar is shown at every width. In the sidebar they were correct on a wide screen
 * and awkward everywhere else: in the admin console you had to leave what you
 * were reading to reach them, and on a phone the sidebar is a drawer, so they
 * were two taps away and invisible until you went looking.
 */
export function Controls({ view, onView }: { view: "chat" | "admin"; onView: (v: "chat" | "admin") => void }) {
  const { t, lang, setLang } = useI18n();
  const { user, logout } = useAuth();
  const [theme, toggleTheme] = useTheme();

  // The icon and the label name what the press will do, not the state it is in —
  // on a single toggle that is the only reading with one meaning.
  const nextTheme = theme === "dark" ? t("themeLight") : t("themeDark");
  const ThemeIcon = theme === "dark" ? IconSun : IconMoon;
  // A flagged reviewer (the lead technician) gets the same console, limited to review.
  const consoleLabel = user?.role === "admin" ? t("admin") : t("review");

  return (
    <div className="controls">
      {user?.can_review && (
        <button className="ico" aria-pressed={view === "admin"}
                onClick={() => onView(view === "chat" ? "admin" : "chat")}
                title={view === "chat" ? consoleLabel : t("chat")}
                aria-label={view === "chat" ? consoleLabel : t("chat")}>
          {view === "chat" ? <IconSliders /> : <IconChat />}
        </button>
      )}

      {/* Two letters instead of a globe: a globe says "language" without saying
          which one you get, which is the one thing a two-language switch must say. */}
      <button className="ico ico--text" onClick={() => setLang(lang === "en" ? "el" : "en")}
              title={lang === "en" ? "Ελληνικά" : "English"} aria-label={lang === "en" ? "Ελληνικά" : "English"}>
        {lang === "en" ? "EL" : "EN"}
      </button>

      <button className="ico" onClick={toggleTheme} title={nextTheme} aria-label={nextTheme}>
        <ThemeIcon />
      </button>

      <button className="ico" onClick={logout} title={t("signOut")} aria-label={t("signOut")}>
        <IconPower />
      </button>
    </div>
  );
}
