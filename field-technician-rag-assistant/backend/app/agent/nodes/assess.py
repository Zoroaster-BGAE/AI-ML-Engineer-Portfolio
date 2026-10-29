"""assess: deterministic signals + LLM grader -> ANSWER / EXPAND / ASK / ESCALATE (the blended gate)."""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.agent import facets as fx
from app.agent.context import TurnContext
from app.agent.gate import GateSignals, compute_signals, decide, hits_with_code
from app.agent.prompts import v1
from app.agent.schemas import Grade
from app.domain.enums import AgentAction


@dataclass
class Assessment:
    action: AgentAction
    reason: str
    partial: bool
    signals: GateSignals
    grade: Grade | None
    facet: fx.Facet | None      # the facet to ask about when action == ASK


def excerpts(hits, n: int, width: int = 900) -> str:
    return "\n\n".join(f"[{i}] {h.doc_title} p.{h.page}\n{h.content[:width]}" for i, h in enumerate(hits[:n], start=1))


def assess(ctx: TurnContext) -> Assessment:
    s, u = ctx.settings, ctx.understanding
    t = time.perf_counter()
    plan_ids = {c.id for c in ctx.plan}
    sig = compute_signals(ctx.hits, u.slots.error_codes, plan_ids, max_distance=s.gate_max_distance)

    grade = None
    if ctx.hits and sig.det_ok:  # nothing to grade when the deterministic gate already failed
        user = (f"Question: {u.standalone_query}\nError codes: {', '.join(u.slots.error_codes) or '-'}\n\n"
                f"EXCERPTS\n{excerpts(ctx.hits, s.context_chunks)}")
        grade = ctx.rec.llm_step("assess_grade", v1.ASSESS, user, Grade, max_tokens=1200)
    ctx.grade = grade

    # Ambiguity is judged on the hits that carry the exact code when there are any, else on the top hits.
    # Only search-ranked hits vote: pages fetched for context (read_pages) must not tilt the facet distribution.
    ranked = [h for h in ctx.hits if not h.via_read]
    pool = hits_with_code(ranked, u.slots.error_codes) or ranked[:10]
    facets = fx.candidate_facets(pool, use_doc_type=True)
    asked = ctx.memory.get("asked", [])
    ambiguous = fx.pick_ambiguous([f for f in facets if f.name != "doc_type"], asked, s.gate_margin)
    fallback = fx.any_askable(facets, asked)
    action, reason, partial = decide(
        sig, grade, ambiguous=ambiguous is not None, any_facet=fallback is not None,
        expand_count=ctx.expand_count, ask_turns=ctx.memory.get("ask_turns", 0),
        max_expand=s.max_expand_loops, max_ask=s.max_ask_turns)
    facet = ambiguous or fallback if action == AgentAction.ASK else None
    ctx.rec.code_step("assess", inp={"signals": sig.as_dict(), "asked": asked, "expand_count": ctx.expand_count},
                      out={"action": action.value, "partial": partial,
                           "grade": grade.model_dump() if grade else None,
                           "facets": [{"name": f.name, "entropy": round(f.entropy, 3), "margin": round(f.margin, 3),
                                       "options": [o.label for o in f.options[:6]]} for f in facets]},
                      rationale=reason, t_start=t)
    return Assessment(action, reason, partial, sig, grade, facet)
