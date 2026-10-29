import { useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { useI18n, type TKey } from "../i18n";
import { ExamplesTab } from "./ExamplesTab";
import { GapsTab } from "./GapsTab";
import { LibraryTab } from "./LibraryTab";
import { SafetyTab } from "./SafetyTab";
import { TracesTab } from "./TracesTab";
import { UsersTab } from "./UsersTab";

type Tab = "users" | "library" | "traces" | "gaps" | "examples" | "safety";
const ADMIN_TABS: Tab[] = ["examples", "safety", "users", "library", "traces", "gaps"];
// A flagged reviewer who is not an admin -- the lead technician -- sees only the review work.
const REVIEWER_TABS: Tab[] = ["examples", "safety"];

export function AdminConsole() {
  const { t } = useI18n();
  const { user } = useAuth();
  const tabs = user?.role === "admin" ? ADMIN_TABS : REVIEWER_TABS;
  const [tab, setTab] = useState<Tab>("examples");
  return (
    <div className="admin">
      <div className="admin-tabs" role="tablist">
        {tabs.map((id) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "scope" : "mini"} onClick={() => setTab(id)}>
            {t(id as TKey)}
          </button>
        ))}
      </div>
      {tab === "examples" && <ExamplesTab />}
      {tab === "safety" && <SafetyTab />}
      {tab === "users" && <UsersTab />}
      {tab === "library" && <LibraryTab />}
      {tab === "traces" && <TracesTab />}
      {tab === "gaps" && <GapsTab />}
    </div>
  );
}
