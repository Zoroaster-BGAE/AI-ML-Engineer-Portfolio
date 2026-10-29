"""Confidence gate (PRD section 6): a small model's self-reported confidence is poorly calibrated, so the
decision blends deterministic signals with the LLM grader. Pure functions; thresholds live in config."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.agent.schemas import Grade
from app.domain.enums import AgentAction
from app.retrieval.repository import Hit


def code_pattern(code: str) -> re.Pattern | None:
    """'E4' also matches 'E 4', 'E-4' and 'E.4', but not inside longer tokens such as 'BE41'."""
    chars = re.sub(r"[\s\-_.]", "", code.strip().upper())
    if len(chars) < 2:
        return None
    body = r"[\s\-_.]?".join(re.escape(c) for c in chars)
    return re.compile(rf"(?<![A-Z0-9]){body}(?![A-Z0-9])")


def hits_with_code(hits: list[Hit], codes: list[str]) -> list[Hit]:
    pats = [p for p in map(code_pattern, codes) if p]
    return [h for h in hits if any(p.search(h.content.upper()) for p in pats)] if pats else []


@dataclass
class GateSignals:
    n_hits: int
    best_distance: float | None
    exact_code_hit: bool
    plan_coverage: float
    det_ok: bool

    def as_dict(self) -> dict:
        return asdict(self)


def compute_signals(hits: list[Hit], codes: list[str], plan_doc_ids: set, *, max_distance: float) -> GateSignals:
    dists = [h.vector_distance for h in hits if h.vector_distance is not None]
    best = min(dists) if dists else None
    exact = bool(hits_with_code(hits, codes))
    covered = {h.document_id for h in hits} & set(plan_doc_ids)
    coverage = len(covered) / len(plan_doc_ids) if plan_doc_ids else 0.0
    det_ok = bool(hits) and (exact or (best is not None and best <= max_distance))
    return GateSignals(len(hits), best, exact, coverage, det_ok)


def decide(sig: GateSignals, grade: Grade | None, *, ambiguous: bool, any_facet: bool, expand_count: int,
           ask_turns: int, max_expand: int, max_ask: int) -> tuple[AgentAction, str, bool]:
    """-> (action, reason, partial). `partial` marks an ANSWER given on incomplete evidence."""
    insufficient = (not sig.det_ok) or (grade is not None and not grade.sufficient)
    if ambiguous and ask_turns < max_ask:
        return AgentAction.ASK, "candidates are too close to call; asking the technician", False
    if not insufficient:
        return AgentAction.ANSWER, "deterministic gate and grader agree the evidence is sufficient", False
    if expand_count < max_expand:
        return AgentAction.EXPAND, "evidence insufficient; widening within scope", False
    if any_facet and ask_turns < max_ask:
        return AgentAction.ASK, "budget spent and a facet remains; asking", False
    if sig.det_ok:
        return AgentAction.ANSWER, "budget spent; best grounded partial answer", True
    return AgentAction.ESCALATE, "no sufficiently relevant evidence in this machine's library", False
