"""retrieve: hybrid search + read_pages, restricted to the scope AND the plan, within a tool-call budget."""
from __future__ import annotations

import time

from app.agent.context import TurnContext
from app.agent.prompts import v1
from app.agent.schemas import RetrieveDecision
from app.retrieval.hybrid import hybrid_search
from app.retrieval.repository import Hit

_READ_BOOST = 0.02  # context pages sit mid-list; they are excluded from ambiguity scoring (see assess)


def _merge(*groups: list[Hit]) -> list[Hit]:
    best: dict = {}
    for g in groups:
        for h in g:
            cur = best.get(h.chunk_id)
            if cur is None:
                best[h.chunk_id] = h
            elif not h.via_read:  # context pages never add score to a chunk that search already ranked
                cur.rrf_score += h.rrf_score
                dists = [d for d in (cur.vector_distance, h.vector_distance) if d is not None]
                ranks = [r for r in (cur.fts_rank, h.fts_rank) if r is not None]
                cur.vector_distance = min(dists) if dists else None
                cur.fts_rank = max(ranks) if ranks else None
    return sorted(best.values(), key=lambda h: h.rrf_score, reverse=True)


def search(ctx: TurnContext, query: str, doc_ids, *, node: str = "retrieve", why: str | None = None) -> list[Hit]:
    t = time.perf_counter()
    ids = ctx.repo.filter_doc_ids(doc_ids)  # defence in depth: only documents inside the scope survive
    hits = hybrid_search(ctx.repo, ctx.embedder, query, doc_ids=ids or None, top_k=ctx.settings.retrieval_candidate_k)
    ctx.rec.code_step(node, inp={"query": query, "documents": [str(i) for i in ids]},
                      out={"n_hits": len(hits)}, rationale=why, t_start=t, hits=hits[:12])
    return hits


def read_pages(ctx: TurnContext, doc_id, page_from: int, page_to: int) -> list[Hit]:
    t = time.perf_counter()
    cap = ctx.settings.read_pages_cap
    page_from = max(1, page_from)
    page_to = min(max(page_to, page_from), page_from + cap - 1)
    ids = ctx.repo.filter_doc_ids([doc_id])
    hits = ctx.repo.read_pages(next(iter(ids)), page_from, page_to) if ids else []
    for h in hits:
        h.rrf_score, h.via_read = _READ_BOOST, True
    ctx.rec.code_step("read_pages", inp={"document_id": str(doc_id), "pages": [page_from, page_to]},
                      out={"n_chunks": len(hits)}, t_start=t, hits=hits[:12])
    return hits


def _snippets(hits: list[Hit], ctx: TurnContext, n: int = 8) -> str:
    alias = {c.id: c.alias for c in ctx.plan}
    return "\n".join(f"- {alias.get(h.document_id, '?')} ({h.doc_title}) p.{h.page} [{h.section_path or '-'}]: "
                     f"{' '.join(h.content.split())[:220]}" for h in hits[:n]) or "(no results)"


def retrieve(ctx: TurnContext, *, queries: list[str] | None = None, doc_ids=None) -> list[Hit]:
    """One retrieval round. `queries`/`doc_ids` let the EXPAND loop reuse it with reformulations."""
    s = ctx.settings
    u = ctx.understanding
    ids = doc_ids if doc_ids is not None else [c.id for c in ctx.plan]
    first_q = (queries or [u.standalone_query + " " + " ".join(ctx.hints[:3])])[0].strip()
    groups = [search(ctx, first_q, ids)]
    used = 1
    for extra in (queries or [])[1:]:
        if used >= s.max_tool_calls:
            break
        groups.append(search(ctx, extra, ids, why="reformulated query"))
        used += 1

    if queries is None and not s.retrieval_controller and groups[0]:
        top = groups[0][0]  # deterministic completion: a procedure usually continues on the next page
        if top.content_type != "wiring_sheet":
            groups.append(read_pages(ctx, top.document_id, top.page, top.page + 1))
        return _merge(*groups)[:30]

    if queries is None:  # first round only: let the controller complete the evidence
        remaining = s.max_tool_calls - used
        user = (f"Question: {u.standalone_query}\nError codes: {', '.join(u.slots.error_codes) or '-'}\n"
                f"Actions allowed: at most {remaining}\nDocuments: "
                f"{'; '.join(f'{c.alias}={c.title}' for c in ctx.plan)}\n\nTOP RESULTS\n{_snippets(groups[0], ctx)}")
        decision = ctx.rec.llm_step("retrieve_plan", v1.RETRIEVE, user, RetrieveDecision, max_tokens=1500)
        view = ctx.catalog
        for act in decision.actions[:remaining]:
            if act.action == "finish":
                break
            if act.action == "search" and act.query:
                groups.append(search(ctx, act.query, ids, why=act.rationale))
            elif act.action == "read_pages" and act.page_from:
                doc = view.resolve(act.document_id)  # unknown / foreign alias -> None -> ignored
                if doc is not None:
                    groups.append(read_pages(ctx, doc.id, act.page_from, act.page_to or act.page_from))
            used += 1
    return _merge(*groups)[:30]
