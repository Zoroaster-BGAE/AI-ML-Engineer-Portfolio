export interface User {
  id: string;
  email: string;
  role: "admin" | "technician" | "customer";
  is_active: boolean;
  access_expires: string | null;
  /** Admins, and users flagged as golden-example reviewers (the lead technician). */
  can_review: boolean;
}
/**
 * A user as the admin console sees them: a Cognito account, plus the one field
 * that only exists locally. Identified by `username` -- the pool's opaque uuid --
 * because that is what the pool calls them; `id` is the local row's key and does
 * not exist for anyone who has never signed in.
 */
export interface AdminUser {
  username: string;
  email: string;
  role: "admin" | "technician" | "customer";
  is_active: boolean;
  status: string;
  access_expires: string | null;
  has_local_row: boolean;
  is_reviewer: boolean;
}

export interface Client { id: string; name: string }
export interface Machine {
  id: string; label: string; kind: "machine" | "line"; year: number | null;
  document_count: number; component_count: number;
}
export interface SessionSummary {
  /** Null until the assistant has identified the machine. */
  id: string; title: string | null; machine_label: string | null; kind: string | null;
  /** Pinned chats group first in the sidebar and are exempt from the 90-day expiry of unrated chats. */
  pinned: boolean;
  /** The technician's rating of the whole conversation, if given. */
  rating: Rating | null;
  /** True once the assistant has answered something: there is something to rate. */
  has_answers: boolean;
  created_at: string; updated_at: string;
}
/** `n` matches the [n] marker in the answer (absent on answers written before references were numbered). */
export interface Source {
  n?: number | null; title: string; doc_type?: string | null; page: number; section?: string | null;
  /** Set when the cited page is a drawing: shown as it is. */
  figure_id?: string | null;
}
export interface SourceExcerpt {
  n: number; title: string; doc_type: string | null; page: number; section: string | null; excerpts: string[];
  figure_id?: string | null;
}
export interface Question { question: string; options: { label: string }[]; allow_free_text: boolean }
export interface Rating { stars: number; comment: string | null }
export interface AttachmentRef { id: string; kind: "image" | "audio" }
/** A drawing shown with an answer, as it is. `n` is set when the answer cited that page. */
export interface FigureRef { figure_id: string; title: string; page: number; n: number | null }
export interface Message {
  id: string; role: "user" | "assistant"; text: string; sources: Source[]; created_at: string;
  attachments: AttachmentRef[]; figures: FigureRef[];
}
/** A photo or voice note on a bubble: stored (`id`) or, just sent, a local object URL (`url`). */
export interface BubbleAttachment { kind: "image" | "audio"; id?: string; url?: string }
export interface SessionDetail extends SessionSummary {
  client_name: string | null; messages: Message[]; pending_question: Question | null;
}

/** A chat bubble as rendered (persisted messages plus the live, streaming one). */
export interface Bubble {
  key: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  question?: Question | null;
  status?: string | null;   // generic progress line while streaming
  error?: boolean;
  /** Set when the turn was stopped by the user rather than finished or failed. */
  stopped?: boolean;
  /** The persisted assistant message, once known: what a rating is attached to. */
  messageId?: string;
  attachments?: BubbleAttachment[];
  figures?: FigureRef[];
}

export type StreamEvent =
  | { event: "status"; data: { key: string; text: string } }
  | { event: "token"; data: string }
  | { event: "question"; data: Question }
  | { event: "scope"; data: { machine_label: string } }
  | { event: "done"; data: {
      action: "ANSWER" | "ASK" | "ESCALATE" | "ERROR"; sources?: Source[]; message?: string; run_id?: string;
      message_id?: string; figures?: FigureRef[];
    } };

// ---- admin ----
export interface RunRow {
  id: string; started_at: string; final_action: "ANSWER" | "ASK" | "ESCALATE" | "ERROR" | null; question: string | null;
  user: string | null; machine: string | null; latency_ms: number | null; llm_calls: number;
  tokens_in: number; tokens_out: number;
}
export interface HitRow {
  rank: number; doc: string | null; page: number | null; vec: number | null; fts: number | null; rrf: number | null;
  selected: boolean; chunk_id: string;
}
export interface StepRow {
  seq: number; node: string; model: string | null; prompt_version: string | null; latency_ms: number | null;
  tokens_in: number | null; tokens_out: number | null; rationale: string | null; reasoning: string | null;
  input: unknown; output: unknown; error: string | null; hits: HitRow[];
}
export interface RunDetail extends RunRow { steps: StepRow[] }
export interface Gap {
  id: string; run_id: string | null; machine: string; question: string; missing: string | null; resolved: boolean;
  created_at: string;
}
export interface LibDoc {
  id: string; title: string; doc_type: string; component: string | null; languages: string[]; page_count: number | null;
  ingest_status: string; ingest_note: string | null; chunk_count: number;
}
export interface LibMachine { id: string; label: string; kind: string; documents: LibDoc[] }
export interface LibClient { id: string; name: string; slug: string; machines: LibMachine[] }

// ---- review (admins and flagged reviewers) ----
export type FailureCause = "retrieval_miss" | "wrong_machine" | "answer_error" | "manual_missing" | "safety" | "other";
export interface ExampleSnapshot {
  question?: string | null;
  answer: string;
  action?: string | null;
  machine?: string | null;
  turns: { role: "user" | "assistant"; text: string }[];
  sources: Source[];
  evidence?: { doc_label: string | null; page: number | null; rank: number }[];
  safety_verdict?: string | null;
  safety_review?: { verdict: string; issues: string[]; warnings: string[]; rationale: string };
}
export interface Example {
  id: string; status: "candidate" | "approved" | "rejected"; kind: "positive" | "hard"; safety: boolean;
  machine: string | null; stars: number | null; comment: string | null; snapshot: ExampleSnapshot;
  reference_answer: string | null; rubric_notes: string | null; failure_cause: FailureCause | null;
  reviewed_by: string | null; reviewed_at: string | null; created_at: string;
}
export interface SafetyEventRow {
  id: string; kind: "blocked" | "amended" | "tripwire" | "feedback"; machine: string | null; question: string | null;
  detail: {
    verdict?: { verdict: string; issues: string[]; warnings: string[]; rationale: string };
    draft?: string; draft_hits?: string[]; question_hits?: string[]; comment?: string; stars?: number; answer?: string;
  };
  resolved: boolean; note: string | null; created_at: string;
}
export interface GoldenSetRow { version: number; example_count: number; created_at: string }
