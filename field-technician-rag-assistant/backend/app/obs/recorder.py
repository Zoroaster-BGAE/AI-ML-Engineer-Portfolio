"""Trace recorder: every node of every turn is persisted (agent_run / agent_step / retrieval_hit) so any
turn can be replayed from its run_id (FR-19). Text stored here is admin-only and purged after the
retention window (ADR 0001 / D6)."""
from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.agent.prompts.v1 import PROMPT_VERSION
from app.domain.support import AgentRun, AgentStep, RetrievalHit
from app.llm.base import LLMClient, LLMError, Usage
from app.obs import metrics
from app.obs.logging import get_logger
from app.retrieval.repository import Hit

log = get_logger()
_MAX_STORED = 12000


class RunRecorder:
    def __init__(self, db: Session, llm: LLMClient, *, session_id: UUID, user_id: UUID, client_id: UUID | None,
                 machine_line_id: UUID | None, message_id: UUID | None, question: str):
        self.db, self.llm = db, llm
        self.t0 = time.perf_counter()
        self.run = AgentRun(session_id=session_id, user_id=user_id, client_id=client_id,
                            machine_line_id=machine_line_id, message_id=message_id, question=question[:2000])
        db.add(self.run)
        db.commit()
        self.seq = 0
        self.usage = Usage()
        self.llm_calls = 0
        self.last_step: AgentStep | None = None

    @property
    def run_id(self) -> UUID:
        return self.run.id

    def _add(self, node, *, inp, out, rationale, latency_ms, model=None, reasoning=None, usage: Usage | None = None,
             error=None, prompt_version=None) -> AgentStep:
        self.seq += 1
        step = AgentStep(run_id=self.run.id, seq=self.seq, node=node, llm_model=model, prompt_version=prompt_version,
                         input=inp, output=out, rationale=(rationale or "")[:2000] or None, reasoning_raw=reasoning,
                         latency_ms=latency_ms, tokens_in=usage.tokens_in if usage else None,
                         tokens_out=usage.tokens_out if usage else None, error=error)
        self.db.add(step)
        self.db.commit()
        metrics.NODE_SECONDS.labels(node).observe(latency_ms / 1000)
        self.last_step = step
        return step

    def llm_step(self, node: str, system: str, user: str, schema, *, effort: str | None = None,
                 max_tokens: int = 4000, note: dict | None = None, fast: bool = False, images=None):
        t = time.perf_counter()
        inp = {"user": user[:_MAX_STORED], **(note or {}), **({"images": len(images)} if images else {})}
        try:
            res = self.llm.structured(node=node, system=system, user=user, schema=schema, effort=effort,
                                      max_tokens=max_tokens, fast=fast, images=images)
        except LLMError as exc:
            self._add(node, inp=inp, out=None, rationale=None, latency_ms=int((time.perf_counter() - t) * 1000),
                      model=getattr(self.llm, "model", None), error=str(exc)[:1000], prompt_version=PROMPT_VERSION)
            raise
        self.llm_calls += 1
        self.usage.add(res.usage)
        metrics.LLM_TOKENS.labels("in").inc(res.usage.tokens_in)
        metrics.LLM_TOKENS.labels("out").inc(res.usage.tokens_out)
        out = res.parsed.model_dump()
        self._add(node, inp=inp, out=out, rationale=out.get("rationale"),
                  latency_ms=int((time.perf_counter() - t) * 1000), model=res.model, reasoning=res.reasoning,
                  usage=res.usage, prompt_version=PROMPT_VERSION)
        return res.parsed

    def code_step(self, node: str, *, inp: dict, out: dict, rationale: str | None = None, t_start: float | None = None,
                  hits: list[Hit] | None = None, selected: set | None = None) -> AgentStep:
        ms = int((time.perf_counter() - (t_start or time.perf_counter())) * 1000)
        step = self._add(node, inp=inp, out=out, rationale=rationale, latency_ms=ms)
        if hits:
            selected = selected or set()
            for rank, h in enumerate(hits, start=1):
                self.db.add(RetrievalHit(step_id=step.id, chunk_id=h.chunk_id, document_id=h.document_id, page=h.page,
                                         doc_label=h.doc_title, rank=rank, vector_distance=h.vector_distance,
                                         fts_rank=h.fts_rank, rrf_score=h.rrf_score, selected=h.chunk_id in selected))
            self.db.commit()
        return step

    def add_llm_usage(self, usage: Usage, latency_ms: int, node: str = "synthesize", model: str | None = None,
                      out: dict | None = None) -> None:
        self.llm_calls += 1
        self.usage.add(usage)
        metrics.LLM_TOKENS.labels("in").inc(usage.tokens_in)
        metrics.LLM_TOKENS.labels("out").inc(usage.tokens_out)
        self._add(node, inp=None, out=out, rationale=None, latency_ms=latency_ms, model=model, usage=usage,
                  prompt_version=PROMPT_VERSION)

    def finish(self, action: str, error: str | None = None) -> None:
        self.run.ended_at = datetime.now(UTC)
        self.run.final_action = action
        self.run.llm_calls = self.llm_calls
        self.run.tokens_in, self.run.tokens_out = self.usage.tokens_in, self.usage.tokens_out
        self.run.latency_ms = int((time.perf_counter() - self.t0) * 1000)
        self.db.commit()
        metrics.DECISIONS.labels(action).inc()
        metrics.TURN_SECONDS.observe(self.run.latency_ms / 1000)
        log.info("turn_finished", run_id=str(self.run.id), session_id=str(self.run.session_id), action=action,
                 latency_ms=self.run.latency_ms, llm_calls=self.llm_calls, error=error)
