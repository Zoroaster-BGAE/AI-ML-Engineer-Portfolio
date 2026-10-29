"""synthesize: grounded, safety-first answer streamed through the verbatim guard; citations = title + page."""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from uuid import UUID

from sqlalchemy import select

from app.agent import facets as fx
from app.agent.context import TurnContext, msg
from app.agent.guard import SentenceGuard
from app.agent.prompts import v1
from app.domain.models import Figure
from app.obs import metrics
from app.retrieval.repository import Hit

_LANG = {"el": "Greek", "en": "English", "de": "German", "other": "the same language the technician wrote in"}
_CTX_CHARS = 14000


def select_context(hits: list[Hit], n: int) -> list[Hit]:
    """Top chunks by fused score, capped by count and total size; the numbering is what the model cites."""
    out, size = [], 0
    for h in hits:
        if len(out) >= n or size + len(h.content) > _CTX_CHARS:
            break
        out.append(h)
        size += len(h.content)
    return out


def _context_block(chunks: list[Hit]) -> str:
    parts = [f"[{i}] {h.doc_title} · p. {h.page}" + (f" · section: {h.section_path}" if h.section_path else "")
             + f"\n{h.content}" for i, h in enumerate(chunks, start=1)]
    return "<manual_excerpts>\n" + "\n\n".join(parts) + "\n</manual_excerpts>"


_MARK = re.compile(r"\[(\d{1,2})\]")
_REPEAT = re.compile(r"(\[\d{1,2}\])(?:\s*\1)+")


def source_entries(cited: list[int], chunks: list[Hit]) -> tuple[list[dict], dict[int, int]]:
    """Sources shown to the technician, numbered in order of first citation, one per (document, page); several
    cited chunks of one page share its number. Returns the entries and excerpt-number -> source-number.
    Falls back to the top chunks, unnumbered in the text, if the answer cited nothing."""
    picked = [(k, chunks[k - 1]) for k in cited if 1 <= k <= len(chunks)] or [(0, h) for h in chunks[:3]]
    by_page: dict[tuple, dict] = {}
    mapping: dict[int, int] = {}
    for k, h in picked:
        key = (h.document_id, h.page)
        if key not in by_page:
            by_page[key] = {"n": len(by_page) + 1, "title": h.doc_title,
                            "doc_type": fx.DOC_TYPE_LABEL.get(h.doc_type, h.doc_type), "page": h.page,
                            "section": h.section_path, "document_id": str(h.document_id), "chunk_ids": []}
        entry = by_page[key]
        if str(h.chunk_id) not in entry["chunk_ids"]:
            entry["chunk_ids"].append(str(h.chunk_id))
        if k:
            mapping[k] = entry["n"]
    return list(by_page.values()), mapping


MAX_FIGURES = 3


def attach_figures(db, sources: list[dict], chunks: list[Hit]) -> list[dict]:
    """Drawings to show with the answer, as they are: cited drawing pages first (their source gets `figure_id`),
    then drawing pages among the passages the answer was written from, even if it did not cite them -- a
    retrieved wiring sheet is worth showing. At most MAX_FIGURES."""
    docs = {UUID(s["document_id"]) for s in sources if s.get("document_id")} | {h.document_id for h in chunks}
    if not docs:
        return []
    found = {(str(f.document_id), f.page): str(f.id)
             for f in db.scalars(select(Figure).where(Figure.document_id.in_(docs)))}
    figures, seen = [], set()
    for s in sources:
        fid = found.get((s.get("document_id"), s["page"]))
        if fid:
            s["figure_id"] = fid
            if fid not in seen:
                seen.add(fid)
                figures.append({"figure_id": fid, "title": s["title"], "page": s["page"], "n": s["n"]})
    for h in chunks:
        fid = found.get((str(h.document_id), h.page))
        if fid and fid not in seen:
            seen.add(fid)
            figures.append({"figure_id": fid, "title": h.doc_title, "page": h.page, "n": None})
    return figures[:MAX_FIGURES]


def renumber(text: str, mapping: dict[int, int]) -> str:
    """Excerpt numbers -> source numbers, so [n] in the answer opens the n-th source. Unknown numbers go."""
    out = _MARK.sub(lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else "", text)
    return _REPEAT.sub(r"\1", out)


class SynthResult:
    def __init__(self):
        self.text = ""
        self.sources: list[dict] = []
        self.figures: list[dict] = []
        self.selected: set = set()
        self.dropped = 0


def synthesize(ctx: TurnContext, partial: bool, result: SynthResult) -> Iterator[str]:
    """Yields guarded text pieces; fills `result` (text, sources, selected chunk ids)."""
    s = ctx.settings
    chunks = select_context(ctx.hits, s.context_chunks)
    result.selected = {h.chunk_id for h in chunks}
    system = v1.SYNTHESIZE.format(machine=ctx.machine_label, language=_LANG.get(ctx.lang, "English"),
                                  partial=v1.PARTIAL_NOTE if partial else "")
    user = f"{_context_block(chunks)}\n\nTechnician's question: {ctx.understanding.standalone_query}"
    guard = SentenceGuard((h.content for h in ctx.hits), s.verbatim_ngram, keep_markers=True)
    t = time.perf_counter()
    if ctx.images:
        user += f"\n\n{len(ctx.images)} photo(s) from the technician are attached (panel, display or machine)."
    stream = ctx.llm.stream_text(system=system, user=user, effort=s.effort_synthesis,
                                 max_tokens=s.synthesis_max_tokens, fast=True, images=ctx.images or None)
    for delta in stream:
        for piece in guard.feed(delta):
            result.text += piece
            yield piece
    for piece in guard.flush():
        result.text += piece
        yield piece
    result.dropped = guard.dropped
    metrics.GUARD_DROPS.inc(guard.dropped)
    result.sources, mapping = source_entries(guard.cited, chunks)
    result.figures = attach_figures(ctx.db, result.sources, chunks)
    # The answer is buffered for the safety review, so the markers can be renumbered before anyone sees them.
    result.text = renumber(result.text, mapping)
    ctx.rec.add_llm_usage(stream.usage, int((time.perf_counter() - t) * 1000), "synthesize", stream.model,
                          out={"chars": len(result.text), "guard_dropped": guard.dropped, "cited": guard.cited,
                               "sources": result.sources})
    if not result.text.strip():
        result.text = msg("guarded", ctx.lang)
        yield result.text
