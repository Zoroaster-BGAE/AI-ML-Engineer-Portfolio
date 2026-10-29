import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, ApiError, cognitoRefresh, cognitoSignIn, cognitoSignOut, tokenStore } from "../api/client";
import type { User } from "../api/types";

interface Auth {
  user: User | null;
  loading: boolean;
  /** True when a session ended by itself rather than by signing out. */
  expired: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
}
const Ctx = createContext<Auth | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [expired, setExpired] = useState(false);
  // Either token means there may be a session to restore, so show the loading
  // state rather than flashing the login screen before the refresh resolves.
  const [loading, setLoading] = useState<boolean>(!!tokenStore.get() || !!tokenStore.getRefresh());

  const refresh = useCallback(async () => {
    // A stored id token is almost always expired -- they last five minutes and
    // people close tabs for longer than that. The refresh token is what carries
    // a session across days, so try it before concluding nobody is signed in.
    if (!tokenStore.get() && !(await cognitoRefresh())) {
      setUser(null); setLoading(false); return;
    }
    try { setUser(await api<User>("/api/auth/me")); }
    catch (e) { if (e instanceof ApiError && e.status !== 401) console.error(e); setUser(null); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  // any 401 from the API (disabled user, expired token) drops us back to the login screen at once
  useEffect(() => {
    const h = () => { setUser(null); setExpired(true); };
    window.addEventListener("assistant:unauthorized", h);
    return () => window.removeEventListener("assistant:unauthorized", h);
  }, []);

  // Straight to Cognito: no password reaches our API, and there is no login
  // endpoint on it any more. /api/auth/me is then the first authenticated call,
  // and the one that creates this user's app_user row if it is their first
  // sign-in.
  const login = useCallback(async (email: string, password: string) => {
    setExpired(false);
    await cognitoSignIn(email.trim().toLowerCase(), password);
    await refresh();
  }, [refresh]);

  const logout = useCallback(() => {
    // Fire and forget: the local tokens are cleared synchronously so the UI
    // returns to the login screen at once, while the refresh token is revoked
    // at Cognito in the background.
    void cognitoSignOut();
    setExpired(false);   // signing out is deliberate, so it is not an expiry
    setUser(null);
  }, []);
  const value = useMemo(() => ({ user, loading, expired, login, logout, refresh }),
                        [user, loading, expired, login, logout, refresh]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth(): Auth {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth outside AuthProvider");
  return v;
}
