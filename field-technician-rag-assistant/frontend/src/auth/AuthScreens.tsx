import { useState, type FormEvent } from "react";
import { ApiError } from "../api/client";
import { useI18n } from "../i18n";
import { useAuth } from "./AuthContext";

export function Login() {
  const { login, expired } = useAuth();
  const { t } = useI18n();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      await login(email.trim(), password);
    } catch (err) {
      // Cognito's own wording distinguishes a wrong password from a disabled or
      // locked-out account. Collapsing all three into "invalid email or password"
      // sends someone to reset a password that was never the problem.
      setError(err instanceof ApiError && err.message ? err.message : t("errGeneric"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <form className="login__card" onSubmit={submit}>
        <div className="login__brand">
          <span className="brand__mark" style={{ width: 40, height: 40, fontSize: "1.3rem", borderRadius: 11 }}>E</span>
          <h1>{t("appName")}</h1>
        </div>

        {/* Shown when the session ended by itself rather than by signing out.
            Five-minute id tokens make this the normal way a session ends: the
            refresh was rejected, meaning the account was disabled or the refresh
            window ran out. Without it the login screen simply reappears, which
            reads as a bug. */}
        {expired && !error && <div role="status" className="banner banner--warn">{t("sessionExpired")}</div>}

        <div className="login__field">
          <label className="note" htmlFor="email">{t("email")}</label>
          <input id="email" className="field" type="email" inputMode="email" autoComplete="username"
                 autoCapitalize="none" spellCheck={false} value={email}
                 onChange={(e) => setEmail(e.target.value)} autoFocus required />
        </div>

        <div className="login__field">
          <label className="note" htmlFor="password">{t("password")}</label>
          <input id="password" className="field" type="password" autoComplete="current-password"
                 value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>

        {error && <div role="alert" className="banner banner--bad">{error}</div>}

        <button className="btn btn--primary btn--block" disabled={busy || !email || !password}>
          {busy ? t("signingIn") : t("signIn")}
        </button>
      </form>
    </div>
  );
}
