"""The fixed outer state machine (PRD section 6). The LLM decides inside each node; code owns the transitions
and the budgets, so the flow is testable with a scripted FakeLLM.

    scope locked -> understand -> route -> retrieve -> assess
        assess: ANSWER -> synthesize (+ verbatim guard) -> safety review -> shown (approved / with warnings)
                                                                         or blocked -> escalation text
                EXPAND (max 2) -> retrieve
                ASK (max 2 turns) -> clarify chips -> (technician reply) -> understand
                ESCALATE -> honest "not in library" + retrieval_gap
"""
from __future__ import annotations

from collections.abc import Generator, Iterator
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent import safety
from app.agent.context import TurnContext, msg
from app.agent.nodes.assess import Assessment, assess
from app.agent.nodes.clarify import build_question
from app.agent.nodes.identify import MachineCatalog, identify
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.route import route
from app.agent.nodes.synthesize import SynthResult, select_context, synthesize
from app.agent.nodes.understand import understand
from app.agent.prompts import v1
from app.agent.schemas import Reformulation
from app.config import Settings
from app.config import settings as default_settings
from app.domain.enums import AgentAction
from app.domain.models import MachineLine
from app.domain.scope import Scope
from app.domain.support import AppUser, ChatMessage, ChatSession, RetrievalGap, SafetyEvent
from app.llm.base import ImageInput, LLMClient, LLMError
from app.llm.embed import Embedder, EmbeddingUnavailable
from app.obs import metrics
from app.obs.logging import get_logger
from app.obs.recorder import RunRecorder
from app.quality.nominate import nominate_blocked
from app.retrieval.catalog import Catalog
from app.retrieval.repository import ScopedRepository

log = get_logger()
Event = dict  # {"event": "status"|"token"|"question"|"done", "data": ...}
_EMIT_CHARS = 48  # a reviewed answer is released in small pieces so the UI keeps its streaming feel


def _status(key: str, text: str) -> Event:
    return {"event": "status", "data": {"key": key, "text": text}}


def _pieces(text: str) -> Iterator[Event]:
    for i in range(0, len(text), _EMIT_CHARS):
        yield {"event": "token", "data": text[i:i + _EMIT_CHARS]}


def apply_effect(memory: dict, effect: dict) -> None:
    """A chip's routing effect narrows later routing (component / documents / doc type)."""
    f = memory.setdefault("filters", {})
    for k in ("component", "doc_ids", "doc_type"):
        if effect.get(k):
            f[k] = effect[k]


class Agent:
    def __init__(self, db: Session, llm: LLMClient, embedder: Embedder, settings: Settings = default_settings):
        self.db, self.llm, self.embedder, self.settings = db, llm, embedder, settings

    # ------------------------------------------------------------------------------------------------
    def run_turn(self, session: ChatSession, user: AppUser, text: str, option_index: int | None = None,
                 images: list[ImageInput] | None = None, user_message_id=None) -> Iterator[Event]:
        """One technician turn. `user_message_id` is set when the API already stored the message (with its
        attachments); otherwise the message is stored here."""
        db = self.db
        images = list(images or [])
        memory = dict(session.working_memory or {})
        pending = session.pending_question

        shown_text, effect = text, {}
        if option_index is not None and pending and 0 <= option_index < len(pending.get("options", [])):
            opt = pending["options"][option_index]
            effect = opt.get("effect") or {}
            shown_text = opt["label"]
        session.pending_question = None
        session.updated_at = datetime.now(UTC)

        if user_message_id is None:
            user_msg = ChatMessage(session_id=session.id, role="user", content=shown_text)
            db.add(user_msg)
            db.flush()
            user_message_id = user_msg.id
        if not session.title:
            session.title = shown_text[:60]
        db.commit()
        db.refresh(session, attribute_names=["messages"])

        rec = RunRecorder(db, self.llm, session_id=session.id, user_id=user.id, client_id=session.client_id,
                          machine_line_id=session.machine_line_id, message_id=user_message_id, question=shown_text)
        try:
            if session.machine_line_id is None:
                locked = yield from self._identify(session, rec, memory, shown_text, effect, images)
                if not locked:
                    return
                issue = memory.pop("pre_scope_issue", None)
                if not issue:  # the machine is known but nothing has been asked yet
                    yield from self._say(session, rec, memory, msg("machine_locked", memory.get("lang", "en"),
                                                                   machine=self._machine_label(session)), "ASK")
                    return
                # Hand-off: the question asked before the machine was known is now answered in its scope.
                text, shown_text, pending, effect = issue, issue, None, {}
            if effect:
                apply_effect(memory, effect)
            prev_issue = memory.get("issue")
            if not pending:
                memory["issue"] = text  # a fresh issue starts here
            scope = Scope(session.client_id, session.machine_line_id)  # from the session row, never the payload
            ctx = TurnContext(db, self.llm, self.embedder, ScopedRepository(db, scope), rec, session, self.settings,
                              self._machine_label(session), memory, shown_text, prev_issue=prev_issue, images=images)
            yield from self._flow(ctx)
        except (LLMError, EmbeddingUnavailable) as exc:
            db.rollback()
            user_message = getattr(exc, "user_message", "The assistant is temporarily unavailable. Please try again.")
            log.error("turn_failed", run_id=str(rec.run_id), error=str(exc))
            rec.finish(AgentAction.ERROR.value, error=str(exc))
            yield {"event": "done", "data": {"action": "ERROR", "message": user_message, "run_id": str(rec.run_id)}}
        except Exception as exc:  # noqa: BLE001 - a turn must always end with a terminal event
            db.rollback()
            log.exception("turn_crashed", run_id=str(rec.run_id))
            rec.finish(AgentAction.ERROR.value, error=repr(exc))
            yield {"event": "done", "data": {"action": "ERROR", "run_id": str(rec.run_id),
                                             "message": "Something went wrong on our side. Please try again."}}

    # ---- the Questioner agent (unscoped chats) -------------------------------------------------------
    def _machine_label(self, session: ChatSession) -> str:
        return self.db.get(MachineLine, session.machine_line_id).label

    def _identify(self, session: ChatSession, rec: RunRecorder, memory: dict, text: str, effect: dict,
                  images: list[ImageInput]) -> Generator[Event, None, bool]:
        """Returns True once the scope is locked (the caller then hands off to the RAG flow)."""
        yield _status("identify", "Finding out which machine this is…")
        catalog = MachineCatalog.load(self.db)
        chosen = catalog.by_id(effect["machine_line_id"]) if effect.get("machine_line_id") else None
        ident = None
        if chosen is None:
            ident = identify(self.db, rec, catalog, session, text, images)
            memory["lang"] = ident.guess.language
            if ident.guess.has_issue and text.strip():
                prev = memory.get("pre_scope_issue")
                memory["pre_scope_issue"] = f"{prev}\n{text}" if prev else text
            chosen = ident.machine
        if chosen is not None:
            # Written once: the database trigger refuses any later change (migration 0008).
            session.client_id, session.machine_line_id = chosen.client_id, chosen.id
            rec.run.client_id, rec.run.machine_line_id = chosen.client_id, chosen.id
            rec.code_step("lock_scope", inp={"via": "chip" if ident is None else "questioner"},
                          out={"machine": chosen.label, "machine_line_id": str(chosen.id)})
            session.working_memory = memory
            self.db.commit()
            yield {"event": "scope", "data": {"machine_label": chosen.label}}
            return True

        lang = memory.get("lang", "en")
        options = [{"label": f"{c.label} · {c.client}", "effect": {"machine_line_id": str(c.id)}}
                   for c in ident.candidates]
        if options:
            options.append({"label": msg("none_of_these", lang), "effect": {}})
        question = ident.guess.question.strip() or msg("q_machine", lang)
        memory["identify_turns"] = memory.get("identify_turns", 0) + 1
        session.pending_question = {"kind": "machine", "question": question, "options": options} if options else None
        public = {"question": question, "options": [{"label": o["label"]} for o in options], "allow_free_text": True}
        self.db.add(ChatMessage(session_id=session.id, role="assistant", content=question,
                                meta={"question": public}, run_id=rec.run_id))
        session.working_memory = memory
        self.db.commit()
        rec.finish("ASK")
        yield {"event": "question", "data": public}
        yield {"event": "done", "data": {"action": "ASK", "run_id": str(rec.run_id)}}
        return False

    def _say(self, session: ChatSession, rec: RunRecorder, memory: dict, text: str, action: str) -> Iterator[Event]:
        """A code-authored reply outside the RAG flow (no sources, nothing to review)."""
        session.working_memory = memory
        self.db.add(ChatMessage(session_id=session.id, role="assistant", content=text, meta={"action": action},
                                run_id=rec.run_id))
        self.db.commit()
        rec.finish(action)
        yield from _pieces(text)
        yield {"event": "done", "data": {"action": action, "run_id": str(rec.run_id)}}

    # ------------------------------------------------------------------------------------------------
    def _flow(self, ctx: TurnContext) -> Iterator[Event]:
        s = self.settings
        docs = ctx.repo.documents()
        if not docs:
            text = (msg("no_docs", "en", machine=ctx.machine_label) + "\n\n"
                    + msg("no_docs", "el", machine=ctx.machine_label))
            yield from self._escalate(ctx, "no documents are indexed for this machine", text=text)
            return
        ctx.catalog = Catalog(docs)

        yield _status("understand", "Reading your question…")
        u = understand(ctx)

        if u.intent == "smalltalk":
            if ctx.prev_issue:  # a greeting must not replace the open issue
                ctx.memory["issue"] = ctx.prev_issue
            else:
                ctx.memory.pop("issue", None)
            yield from self._finish_text(ctx, msg("smalltalk", u.language, machine=ctx.machine_label), [], "ANSWER")
            return

        asked = ctx.memory.get("asked", [])
        f = ctx.memory.get("filters", {})
        no_detail = not (u.slots.error_codes or u.slots.symptom or u.slots.component_hint or f)
        if (u.intent == "troubleshoot" and no_detail and "symptom" not in asked
                and ctx.memory.get("ask_turns", 0) < s.max_ask_turns):
            yield from self._ask(ctx, None, "symptom", reason="a required slot is missing (no code or symptom)")
            return

        yield _status("route", f"Checking the {ctx.machine_label} documents…")
        route(ctx)

        yield _status("search", "Searching the documents…")
        ctx.hits = retrieve(ctx)
        while True:
            yield _status("assess", "Checking what I found…")
            a = assess(ctx)
            if a.action == AgentAction.EXPAND:
                ctx.expand_count += 1
                yield _status("expand", "Looking a little wider in this machine's documents…")
                ctx.hits = self._expand(ctx, a)
                continue
            break

        if a.action == AgentAction.ASK:
            yield from self._ask(ctx, a.facet, a.facet.name, reason=a.reason)
        elif a.action == AgentAction.ANSWER:
            yield from self._answer(ctx, a)
        else:
            yield from self._escalate(ctx, "; ".join(a.grade.missing) if a.grade and a.grade.missing else a.reason)

    # ---- transitions --------------------------------------------------------------------------------
    def _expand(self, ctx: TurnContext, a: Assessment):
        u = ctx.understanding
        top = "\n".join(f"- {h.doc_title} p.{h.page}" for h in ctx.hits[:5]) or "(nothing relevant)"
        user = (f"Question: {u.standalone_query}\nError codes: {', '.join(u.slots.error_codes) or '-'}\n"
                f"Missing: {'; '.join(a.grade.missing) if a.grade else 'no relevant results'}\nBest results so far:\n{top}")
        r = ctx.rec.llm_step("expand", v1.REFORMULATE, user, Reformulation, max_tokens=1200)
        cat = ctx.catalog
        plan_ids = [c.id for c in ctx.plan]
        if r.widen == "all_types":
            ids = [c.id for c in cat.docs]
        elif r.widen == "siblings":
            comps = {c.component for c in ctx.plan}
            ids = [c.id for c in cat.docs if c.component in comps or c.id in plan_ids]
        else:
            ids = plan_ids
        queries = [q for q in r.queries if q.strip()][:3] or [u.standalone_query]
        fresh = retrieve(ctx, queries=queries, doc_ids=ids)
        return sorted({h.chunk_id: h for h in [*ctx.hits, *fresh]}.values(), key=lambda h: h.rrf_score, reverse=True)[:30]

    def _reset_issue(self, ctx: TurnContext) -> None:
        comp = (ctx.memory.get("filters") or {}).get("component")
        ctx.memory = {"last_component": comp} if comp else {}

    def _ask(self, ctx: TurnContext, facet, facet_name: str, *, reason: str) -> Iterator[Event]:
        q = build_question(ctx, facet, facet_name)
        ctx.memory["asked"] = [*ctx.memory.get("asked", []), facet_name]
        ctx.memory["ask_turns"] = ctx.memory.get("ask_turns", 0) + 1
        metrics.INTERVIEW_TURNS.labels(facet_name).inc()
        ctx.rec.code_step("clarify_out", inp={"facet": facet_name, "reason": reason},
                          out={"question": q["question"], "options": [o["label"] for o in q["options"]]})
        ctx.session.pending_question = q
        ctx.session.working_memory = ctx.memory
        public = {"question": q["question"], "options": [{"label": o["label"]} for o in q["options"]],
                  "allow_free_text": True}
        ctx.db.add(ChatMessage(session_id=ctx.session.id, role="assistant", content=q["question"],
                               meta={"question": public}, run_id=ctx.rec.run_id))
        ctx.db.commit()
        ctx.rec.finish("ASK")
        yield {"event": "question", "data": public}
        yield {"event": "done", "data": {"action": "ASK", "run_id": str(ctx.rec.run_id)}}

    def _answer(self, ctx: TurnContext, a: Assessment) -> Iterator[Event]:
        chunks = select_context(ctx.hits, self.settings.context_chunks)
        ctx.rec.code_step("select_context", inp={"budget": self.settings.context_chunks},
                          out={"selected": len(chunks)}, hits=chunks, selected={h.chunk_id for h in chunks})
        yield _status("answer", "Preparing your answer…")
        # Buffered, never streamed: nothing reaches the technician before the safety review has judged it.
        result = SynthResult()
        for _ in synthesize(ctx, a.partial, result):
            pass
        yield _status("safety", "Checking the answer for safety…")
        r = safety.review(ctx, result.text, chunks)
        text = safety.render(ctx, r, result.text)
        self._reset_issue(ctx)
        yield from _pieces(text)
        if r.blocked:
            ctx.db.add(RetrievalGap(run_id=ctx.rec.run_id, machine_line_id=ctx.session.machine_line_id,
                                    question=ctx.user_text[:2000],
                                    missing=("safety review blocked the answer: "
                                             + "; ".join(r.verdict.issues))[:1000]))
            yield from self._persist_done(ctx, text, [], "ESCALATE", review=r, draft=result.text)
        else:
            yield from self._persist_done(ctx, text, result.sources, "ANSWER", review=r, draft=result.text,
                                          figures=result.figures)

    def _escalate(self, ctx: TurnContext, missing: str, text: str | None = None) -> Iterator[Event]:
        ctx.db.add(RetrievalGap(run_id=ctx.rec.run_id, machine_line_id=ctx.session.machine_line_id,
                                question=ctx.user_text[:2000], missing=missing[:1000]))
        text = text or msg("escalate", ctx.lang, machine=ctx.machine_label)
        self._reset_issue(ctx)
        yield from self._finish_text(ctx, text, [], "ESCALATE")

    def _finish_text(self, ctx: TurnContext, text: str, sources: list, action: str) -> Iterator[Event]:
        yield {"event": "token", "data": text}
        if action == "ANSWER":  # smalltalk keeps any open issue untouched
            pass
        else:
            self._reset_issue(ctx)
        yield from self._persist_done(ctx, text, sources, action)

    def _persist_done(self, ctx: TurnContext, text: str, sources: list, action: str, *,
                      review: safety.Review | None = None, draft: str | None = None,
                      figures: list | None = None) -> Iterator[Event]:
        ctx.session.working_memory = ctx.memory
        message_id = uuid4()  # returned in `done`, so the answer's references can be opened without a reload
        meta = {"sources": sources, "action": action, "figures": figures or []}
        if review is not None:
            meta["safety"] = review.verdict.verdict
        ctx.db.add(ChatMessage(id=message_id, session_id=ctx.session.id, role="assistant", content=text,
                               meta=meta, run_id=ctx.rec.run_id))
        if review is not None and review.needs_attention:
            ctx.db.flush()  # no ORM relationship orders these inserts, and the event references the message
            self._safety_event(ctx, review, draft or "", message_id)
        ctx.db.commit()
        ctx.rec.finish(action)
        if review is not None and review.blocked:
            nominate_blocked(ctx.db, message_id)
        public = [{k: s.get(k) for k in ("n", "title", "doc_type", "page", "section", "figure_id")} for s in sources]
        yield {"event": "done", "data": {"action": action, "sources": public, "figures": figures or [],
                                         "run_id": str(ctx.rec.run_id), "message_id": str(message_id)}}

    def _safety_event(self, ctx: TurnContext, r: safety.Review, draft: str, message_id) -> None:
        kind = "blocked" if r.blocked else "amended" if r.amended else "tripwire"
        ctx.db.add(SafetyEvent(
            kind=kind, run_id=ctx.rec.run_id, session_id=ctx.session.id, message_id=message_id,
            user_id=ctx.rec.run.user_id, machine_line_id=ctx.session.machine_line_id, question=ctx.user_text[:2000],
            detail={"verdict": r.verdict.model_dump(), "strict": r.strict, "question_hits": r.question_hits,
                    "draft_hits": r.draft_hits, "draft": draft[:12000]}))
