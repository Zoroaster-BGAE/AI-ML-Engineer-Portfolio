"""clarify: code chooses the facet and builds the chips; the LLM only phrases the question."""
from __future__ import annotations

from app.agent import facets as fx
from app.agent.context import TurnContext, msg
from app.agent.prompts import v1
from app.agent.schemas import Clarify
from app.llm.base import LLMError

_LANG = {"el": "Greek", "en": "English", "de": "German", "other": "the technician's language"}


def build_question(ctx: TurnContext, facet: fx.Facet | None, facet_name: str) -> dict:
    lang = ctx.lang
    options = [{"label": o.label, "effect": o.effect} for o in (facet.options[:6] if facet else [])]
    user = (f"Language: {_LANG.get(lang, 'English')}\nFacet: {facet_name}\n"
            f"Candidates: {', '.join(o['label'] for o in options) or '-'}\n"
            f"Machine: {ctx.machine_label}\nIssue: {ctx.memory.get('issue') or ctx.user_text}")
    try:
        c = ctx.rec.llm_step("clarify", v1.CLARIFY, user, Clarify, max_tokens=600)
        question = c.question.strip()
        if facet_name == "symptom":
            options = [{"label": o, "effect": {}} for o in c.options[:4]]
    except LLMError:  # phrasing is cosmetic: never lose the interview to a phrasing failure
        question = msg(f"q_{facet_name}", lang)
    options.append({"label": msg("not_sure", lang), "effect": {}})
    return {"facet": facet_name, "question": question, "options": options, "allow_free_text": True,
            "original_query": ctx.memory.get("issue") or ctx.user_text}
