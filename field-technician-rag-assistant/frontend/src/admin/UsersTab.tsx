import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { AdminUser } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useI18n } from "../i18n";

export function UsersTab() {
  const { t } = useI18n();
  const { user: me } = useAuth();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ email: "", password: "", role: "technician", expires: "" });

  const load = useCallback(() => { api<AdminUser[]>("/api/admin/users").then(setUsers).catch(() => setError(t("errGeneric"))); }, [t]);
  useEffect(load, [load]);

  const run = async (fn: () => Promise<unknown>) => {
    setError(null);
    try { await fn(); load(); } catch (e) { setError(e instanceof ApiError ? e.message : t("errGeneric")); }
  };

  const create = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      await api("/api/admin/users", { method: "POST", body: {
        email: form.email, password: form.password, role: form.role,
        access_expires: form.expires ? new Date(form.expires).toISOString() : null,
      } });
      setForm({ email: "", password: "", role: "technician", expires: "" });
    });
  };

  return (
    <>
      {error && <div role="alert" className="tag tag--bad" style={{ whiteSpace: "normal", marginBottom: 8 }}>{error}</div>}
      <form onSubmit={create} className="admin-form">
        <input className="field" placeholder={t("email")} value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
        <input className="field" type="password" placeholder={t("newTempPassword")} minLength={6} value={form.password}
               onChange={(e) => setForm({ ...form, password: e.target.value })} required />
        <select className="field" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })} aria-label={t("role")}>
          <option value="technician">technician</option><option value="admin">admin</option><option value="customer">customer</option>
        </select>
        <input className="field" type="date" value={form.expires} onChange={(e) => setForm({ ...form, expires: e.target.value })} aria-label={t("expires")} />
        <button className="btn btn--primary">{t("create")}</button>
      </form>
      <div className="tw" style={{ marginTop: 12 }}>
        <table>
          <thead><tr><th>{t("email")}</th><th>{t("role")}</th><th>{t("reviewer")}</th><th>{t("status")}</th><th>{t("expires")}</th><th /></tr></thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.username}>
                <td>{u.email}</td><td>{u.role}</td>
                <td>
                  {/* Golden-example reviewer: the lead technician. Admins can review without it, but
                      hard and safety examples need a flagged reviewer to approve. */}
                  <input type="checkbox" checked={u.is_reviewer} aria-label={`${t("reviewer")}: ${u.email}`}
                         onChange={(e) => void run(() => api(`/api/admin/users/${u.username}`, {
                           method: "PATCH", body: { is_reviewer: e.target.checked } }))} />
                </td>
                <td><span className={`tag ${u.is_active ? "tag--ok" : "tag--bad"}`}>{u.is_active ? t("active") : t("disabled")}</span></td>
                <td>{u.access_expires ? new Date(u.access_expires).toLocaleDateString() : t("noExpiry")}</td>
                <td style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <button className="btn" disabled={u.email === me?.email}
                          onClick={() => void run(() => api(`/api/admin/users/${u.username}`, { method: "PATCH", body: { is_active: !u.is_active } }))}>
                    {u.is_active ? t("disable") : t("enable")}
                  </button>
                  <button className="btn" onClick={() => {
                    const p = window.prompt(t("newTempPassword"));
                    if (p) void run(() => api(`/api/admin/users/${u.username}/reset-password`, { method: "POST", body: { password: p } }));
                  }}>{t("resetPassword")}</button>
                  <button className="btn" disabled={u.email === me?.email}
                          onClick={() => {
                            // Deletes the Cognito account outright, so it is worth
                            // one confirmation. Disable is the reversible option
                            // and sits next to it.
                            if (window.confirm(t("confirmDeleteUser").replace("{email}", u.email))) {
                              void run(() => api(`/api/admin/users/${u.username}`, { method: "DELETE" }));
                            }
                          }}>{t("deleteUser")}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
