import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

export type Lang = "en" | "el";

const en = {
  appName: "Field Technician AI Assistant",
  signIn: "Sign in", email: "Email", password: "Password", signingIn: "Signing in…", signOut: "Sign out",
  badLogin: "Invalid email or password.",
  machine: "Machine", documents: "documents", machineUnknown: "Machine not identified yet",
  loading: "Loading…", newGuide: "New chat", chats: "Chats", noHistory: "No chats yet.",
  pinned: "Pinned", recent: "Recent", pin: "Pin chat", unpin: "Unpin chat", untitledChat: "New chat",
  close: "Close",
  typeMessage: "Type a message…", send: "Send", stop: "Stop", stopped: "Stopped.", copy: "Copy response", copied: "Copied", sources: "Sources", orOwnWords: "Or describe it in your own words.",
  page: "p.", chat: "Chat", admin: "Admin", users: "Users", library: "Library", traces: "History", gaps: "Gaps",
  heroTitle: "How may I assist you?",
  heroBody: "Say which machine you are at and what is happening: a fault code, a problem or a procedure.",
  composerHint: "Answers cite the manual and page. Always verify before working on live equipment.",
  st_identify: "Finding out which machine this is…", st_understand: "Reading your question…", st_route: "Checking the documents…", st_search: "Searching the documents…",
  st_assess: "Checking what I found…", st_expand: "Looking a little wider in this machine's documents…",
  st_answer: "Preparing your answer…", st_safety: "Checking the answer for safety…",
  errGeneric: "Something went wrong. Please try again.", sessionExpired: "Your session expired. Please sign in again.",
  themeLight: "Light mode", themeDark: "Dark mode",
  rateConversation: "Rate this conversation", ratePrompt: "How did this conversation go?",
  finishRate: "Finish & rate", changeRating: "change", addComment: "Add a comment",
  commentPlaceholder: "What was good or wrong? (optional)", skip: "Skip", save: "Save",
  ratingFailed: "The rating was not saved. Please try again.", olderChats: "Show older chats",
  attach: "Add a photo or voice file", record: "Record a voice note", stopRecording: "Stop recording",
  recording: "Recording…", transcribing: "Transcribing…", checkTranscript: "Check the transcript, then send.",
  transcribeFailed: "The voice note could not be transcribed. Please type your message.",
  unsupportedFile: "Only photos and voice recordings can be added.", micDenied: "The microphone is not available.",
  openSource: "Show the manual passage", wiringDiagram: "Wiring diagram",
  photo: "Photo", voiceNote: "Voice note", removePhoto: "Remove photo", actualSize: "Actual size", fitScreen: "Fit to screen",
  // review (admins and reviewers)
  review: "Review", examples: "Examples", safety: "Safety", reviewer: "Reviewer",
  toReview: "To review", approved: "Approved", rejected: "Rejected", noCandidates: "Nothing to review right now.",
  kind_positive: "Good answer", kind_hard: "Needs a correct answer",
  conversation: "Conversation", answer: "Answer", technicianComment: "Technician's comment", evidence: "Evidence used",
  referenceAnswer: "Reference answer", rubricNotes: "What a good answer must contain", failureCause: "What went wrong",
  approve: "Approve", reject: "Reject", backToQueue: "Back to candidates",
  fc_retrieval_miss: "Search missed the right page", fc_wrong_machine: "Wrong machine", fc_answer_error: "Answer was wrong",
  fc_manual_missing: "Manual not uploaded", fc_safety: "Safety problem", fc_other: "Other",
  goldenSets: "Golden sets", publishSet: "Publish golden set", examplesCount: "examples", download: "Download",
  noSafety: "No open safety events.", showResolved: "Show resolved", markResolved: "Mark resolved", reopenEvent: "Reopen",
  note: "Note", draftNeverShown: "Draft (never shown to the technician)", verdict: "Verdict", matched: "Matched keywords",
  sk_blocked: "Blocked", sk_amended: "Warnings added", sk_tripwire: "Keyword", sk_feedback: "Technician comment",
  // admin
  create: "Create", role: "Role", active: "Active", disabled: "Disabled", expires: "Access expires", disable: "Disable",
  enable: "Enable", resetPassword: "Reset password", newTempPassword: "New temporary password", noExpiry: "No expiry",
  deleteUser: "Delete", confirmDeleteUser: "Delete {email} permanently? Disable is reversible; this is not.",
  status: "Status", title: "Title", type: "Type", component: "Component", chunks: "Chunks", reason: "Reason",
  noRuns: "No runs yet.", selectRun: "Select a run to see its steps.", resolve: "Resolve", reopen: "Reopen",
  noGaps: "No retrieval gaps recorded.", question: "Question", missing: "Missing", retag: "Save tags",
  latency: "Latency", details: "Details", reasoning: "Model reasoning summary", input: "Input", output: "Output",
  adminOnly: "Admin-only view: history contains prompts and manual references.",
};
type Dict = typeof en;

const el: Dict = {
  appName: "Field Technician AI Assistant",
  signIn: "Σύνδεση", email: "Email", password: "Κωδικός", signingIn: "Σύνδεση…", signOut: "Αποσύνδεση",
  badLogin: "Λάθος email ή κωδικός.",
  machine: "Μηχάνημα", documents: "έγγραφα", machineUnknown: "Το μηχάνημα δεν έχει εντοπιστεί ακόμη",
  loading: "Φόρτωση…", newGuide: "Νέα συζήτηση", chats: "Συζητήσεις", noHistory: "Δεν υπάρχουν συζητήσεις.",
  pinned: "Καρφιτσωμένα", recent: "Πρόσφατα", pin: "Καρφίτσωμα", unpin: "Ξεκαρφίτσωμα",
  untitledChat: "Νέα συζήτηση", close: "Κλείσιμο",
  typeMessage: "Γράψτε ένα μήνυμα…", send: "Αποστολή", stop: "Διακοπή", stopped: "Διακόπηκε.", copy: "Αντιγραφή απάντησης", copied: "Αντιγράφηκε", sources: "Πηγές", orOwnWords: "Ή περιγράψτε το με δικά σας λόγια.",
  page: "σ.", chat: "Συνομιλία", admin: "Διαχείριση", users: "Χρήστες", library: "Βιβλιοθήκη", traces: "Ιστορικότητα", gaps: "Κενά",
  heroTitle: "Πώς μπορώ να βοηθήσω;",
  heroBody: "Πείτε σε ποιο μηχάνημα βρίσκεστε και τι συμβαίνει: κωδικός βλάβης, πρόβλημα ή διαδικασία.",
  composerHint: "Οι απαντήσεις αναφέρουν εγχειρίδιο και σελίδα. Επιβεβαιώνετε πάντα πριν εργαστείτε σε ενεργό εξοπλισμό.",
  st_identify: "Εντοπίζω ποιο μηχάνημα είναι…", st_understand: "Διαβάζω την ερώτησή σας…", st_route: "Ελέγχω τα έγγραφα…", st_search: "Αναζητώ στα έγγραφα…",
  st_assess: "Ελέγχω τι βρήκα…", st_expand: "Ψάχνω λίγο ευρύτερα στα έγγραφα του μηχανήματος…",
  st_answer: "Ετοιμάζω την απάντησή σας…", st_safety: "Ελέγχω την απάντηση για θέματα ασφάλειας…",
  errGeneric: "Κάτι πήγε στραβά. Δοκιμάστε ξανά.", sessionExpired: "Η συνεδρία έληξε. Συνδεθείτε ξανά.",
  themeLight: "Φωτεινό θέμα", themeDark: "Σκούρο θέμα",
  rateConversation: "Βαθμολογήστε τη συζήτηση", ratePrompt: "Πώς πήγε αυτή η συζήτηση;",
  finishRate: "Ολοκλήρωση & βαθμολογία", changeRating: "αλλαγή", addComment: "Προσθήκη σχολίου",
  commentPlaceholder: "Τι ήταν σωστό ή λάθος; (προαιρετικό)", skip: "Παράλειψη", save: "Αποθήκευση",
  ratingFailed: "Η βαθμολογία δεν αποθηκεύτηκε. Δοκιμάστε ξανά.", olderChats: "Παλαιότερες συζητήσεις",
  attach: "Προσθήκη φωτογραφίας ή ηχητικού", record: "Ηχογράφηση φωνητικού μηνύματος", stopRecording: "Τέλος ηχογράφησης",
  recording: "Ηχογράφηση…", transcribing: "Μεταγραφή…", checkTranscript: "Ελέγξτε τη μεταγραφή και μετά στείλτε.",
  transcribeFailed: "Το φωνητικό μήνυμα δεν μεταγράφηκε. Παρακαλώ πληκτρολογήστε το μήνυμά σας.",
  unsupportedFile: "Μπορείτε να προσθέσετε μόνο φωτογραφίες και ηχογραφήσεις.", micDenied: "Το μικρόφωνο δεν είναι διαθέσιμο.",
  openSource: "Εμφάνιση του αποσπάσματος του εγχειριδίου", wiringDiagram: "Ηλεκτρολογικό σχέδιο",
  photo: "Φωτογραφία", voiceNote: "Φωνητικό μήνυμα", removePhoto: "Αφαίρεση φωτογραφίας", actualSize: "Πραγματικό μέγεθος",
  fitScreen: "Προσαρμογή στην οθόνη",
  review: "Έλεγχος", examples: "Παραδείγματα", safety: "Ασφάλεια", reviewer: "Ελεγκτής",
  toReview: "Προς έλεγχο", approved: "Εγκεκριμένα", rejected: "Απορριφθέντα", noCandidates: "Δεν υπάρχει κάτι για έλεγχο αυτή τη στιγμή.",
  kind_positive: "Καλή απάντηση", kind_hard: "Χρειάζεται σωστή απάντηση",
  conversation: "Συζήτηση", answer: "Απάντηση", technicianComment: "Σχόλιο τεχνικού", evidence: "Πηγές που χρησιμοποιήθηκαν",
  referenceAnswer: "Απάντηση αναφοράς", rubricNotes: "Τι πρέπει να περιέχει μια καλή απάντηση", failureCause: "Τι πήγε στραβά",
  approve: "Έγκριση", reject: "Απόρριψη", backToQueue: "Πίσω στη λίστα",
  fc_retrieval_miss: "Η αναζήτηση δεν βρήκε τη σωστή σελίδα", fc_wrong_machine: "Λάθος μηχάνημα", fc_answer_error: "Λάθος απάντηση",
  fc_manual_missing: "Δεν έχει ανέβει το εγχειρίδιο", fc_safety: "Πρόβλημα ασφάλειας", fc_other: "Άλλο",
  goldenSets: "Σετ αναφοράς", publishSet: "Δημοσίευση σετ αναφοράς", examplesCount: "παραδείγματα", download: "Λήψη",
  noSafety: "Δεν υπάρχουν ανοιχτά συμβάντα ασφάλειας.", showResolved: "Εμφάνιση επιλυμένων", markResolved: "Επίλυση", reopenEvent: "Επαναφορά",
  note: "Σημείωση", draftNeverShown: "Προσχέδιο (δεν εμφανίστηκε στον τεχνικό)", verdict: "Απόφαση", matched: "Λέξεις-κλειδιά",
  sk_blocked: "Μπλοκαρίστηκε", sk_amended: "Προστέθηκαν προειδοποιήσεις", sk_tripwire: "Λέξη-κλειδί", sk_feedback: "Σχόλιο τεχνικού",
  create: "Δημιουργία", role: "Ρόλος", active: "Ενεργός", disabled: "Ανενεργός", expires: "Λήξη πρόσβασης", disable: "Απενεργοποίηση",
  enable: "Ενεργοποίηση", resetPassword: "Επαναφορά κωδικού", newTempPassword: "Νέος προσωρινός κωδικός", noExpiry: "Χωρίς λήξη",
  deleteUser: "Διαγραφή", confirmDeleteUser: "Οριστική διαγραφή του {email}; Η απενεργοποίηση είναι αναστρέψιμη, αυτό όχι.",
  status: "Κατάσταση", title: "Τίτλος", type: "Τύπος", component: "Εξάρτημα", chunks: "Τμήματα", reason: "Αιτιολόγηση",
  noRuns: "Δεν υπάρχουν εκτελέσεις ακόμη.", selectRun: "Επιλέξτε μια εκτέλεση για να δείτε τα βήματά της.", resolve: "Επίλυση",
  reopen: "Επαναφορά", noGaps: "Δεν έχουν καταγραφεί κενά ανάκτησης.", question: "Ερώτηση", missing: "Λείπει", retag: "Αποθήκευση ετικετών",
  latency: "Καθυστέρηση", details: "Λεπτομέρειες", reasoning: "Περίληψη σκέψης του μοντέλου", input: "Είσοδος", output: "Έξοδος",
  adminOnly: "Προβολή μόνο για διαχειριστές: η ιστορικότητα περιέχει prompts και αναφορά εγχειριδίων.",
};

export type TKey = keyof Dict;
const dicts: Record<Lang, Dict> = { en, el };

interface I18n { lang: Lang; setLang: (l: Lang) => void; t: (k: TKey) => string }
const Ctx = createContext<I18n | null>(null);

function initialLang(): Lang {
  try {
    const s = localStorage.getItem("assistant.lang");
    if (s === "en" || s === "el") return s;
  } catch { /* ignore */ }
  return navigator.language?.toLowerCase().startsWith("el") ? "el" : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    document.documentElement.lang = l;
    try { localStorage.setItem("assistant.lang", l); } catch { /* ignore */ }
  }, []);
  const value = useMemo<I18n>(() => ({ lang, setLang, t: (k) => dicts[lang][k] }), [lang, setLang]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18n {
  const v = useContext(Ctx);
  if (!v) throw new Error("useI18n outside I18nProvider");
  return v;
}
