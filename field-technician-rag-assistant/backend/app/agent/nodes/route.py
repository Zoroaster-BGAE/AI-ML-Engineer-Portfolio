"""route: validate the plan the model proposed. Every id it named must exist in this scope's catalog; unknown or
foreign ids are dropped and logged (FR-07). The model proposes, code enforces."""
import time

from app.agent.context import TurnContext
from app.agent.nodes.understand import catalog_view
from app.obs.logging import get_logger
from app.retrieval.catalog import CatalogDoc

log = get_logger()


def route(ctx: TurnContext) -> list[CatalogDoc]:
    t = time.perf_counter()
    view = catalog_view(ctx)
    resolved: list[CatalogDoc] = []
    dropped: list[str] = []
    hints: list[str] = []
    for item in ctx.understanding.plan:
        doc = view.resolve(item.document_id)
        if doc is None:
            dropped.append(item.document_id)
        elif doc not in resolved:
            resolved.append(doc)
            hints.extend(item.section_hints[:2])
    fallback = not resolved
    if fallback:  # nothing usable: search the whole (restricted) scope rather than give up
        resolved = list(view.docs)
    if dropped:
        log.warning("route_dropped_foreign_ids", run_id=str(ctx.rec.run_id), ids=dropped)
    ctx.rec.code_step("route", inp={"proposed": [i.document_id for i in ctx.understanding.plan],
                                    "catalog_size": len(view.docs)},
                      out={"resolved": [c.alias for c in resolved], "titles": [c.title for c in resolved],
                           "dropped_ids": dropped, "fallback_to_full_scope": fallback},
                      rationale=ctx.understanding.rationale, t_start=t)
    ctx.plan, ctx.hints = resolved, hints
    return resolved
