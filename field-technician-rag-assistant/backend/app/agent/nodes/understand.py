from app.agent.context import TurnContext
from app.agent.prompts import v1
from app.agent.schemas import Understanding
from app.retrieval.catalog import Catalog


def _recent(ctx: TurnContext, limit: int = 6) -> str:
    msgs = [m for m in ctx.session.messages][-(limit + 1):-1]  # everything before the just-saved user message
    return "\n".join(f"{m.role}: {m.content[:400]}" for m in msgs) or "(none)"


def catalog_view(ctx: TurnContext) -> Catalog:
    """The scope's catalog, narrowed by anything the technician already confirmed (chip effects)."""
    f = ctx.memory.get("filters", {})
    return ctx.catalog.restrict(component=f.get("component"), doc_ids=f.get("doc_ids"), doc_type=f.get("doc_type"))


def understand(ctx: TurnContext) -> Understanding:
    f = ctx.memory.get("filters", {})
    confirmed = "; ".join(f"{k}={v}" for k, v in f.items() if v and k != "doc_ids") or "(none)"
    view = catalog_view(ctx)
    user = (f"Machine or line: {ctx.machine_label}\n"
            f"Open issue: {ctx.memory.get('issue') or '(none)'}\n"
            f"Technician-confirmed context: {confirmed}\n"
            f"Recent conversation:\n{_recent(ctx)}\n\n"
            f"Latest message: {ctx.user_text}\n\n"
            f"CATALOG\n{view.render()}")
    if ctx.images:
        user += (f"\n\n{len(ctx.images)} photo(s) from the technician are attached. Read any alarm code, display "
                 "text, indicator state or nameplate in them and put codes into slots.error_codes.")
    u = ctx.rec.llm_step("understand", v1.UNDERSTAND, user, Understanding, max_tokens=2500,
                         note={"catalog_size": len(view.docs)}, images=ctx.images or None)
    ctx.understanding = u
    return u
