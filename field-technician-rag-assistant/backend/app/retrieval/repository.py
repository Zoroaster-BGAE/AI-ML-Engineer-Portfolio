"""The ONLY module that queries `chunk`. Every query is pinned to a Scope (machine_line + client)
inside SQL, so no caller, plan or LLM output can widen it (PRD FR-05 / FR-08)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, literal_column, select, text
from sqlalchemy.orm import Session

from app.domain.enums import IngestStatus
from app.domain.models import Chunk, Document
from app.domain.scope import Scope

_TOKEN = re.compile(r"\w{2,}", re.UNICODE)


@dataclass
class Hit:
    chunk_id: UUID
    document_id: UUID
    page: int
    section_path: str | None
    content: str
    content_type: str | None
    doc_title: str
    doc_type: str
    component: str | None
    vector_distance: float | None = None
    fts_rank: float | None = None
    rrf_score: float = 0.0
    via_read: bool = False  # fetched by read_pages for context, not ranked by search


def fts_query_string(query: str) -> str | None:
    """OR-of-terms tsquery text from free text (word characters only, so it cannot inject operators)."""
    toks = list(dict.fromkeys(t.lower() for t in _TOKEN.findall(query)))[:24]
    return " | ".join(toks) or None


class ScopedRepository:
    def __init__(self, db: Session, scope: Scope):
        if not isinstance(scope, Scope):
            raise TypeError("ScopedRepository requires a Scope; unscoped retrieval is not allowed")
        self.db, self.scope = db, scope

    # ---- scope predicates --------------------------------------------------------------------------
    def _chunk_scope(self):
        return (Chunk.machine_line_id == self.scope.machine_line_id) & (Chunk.client_id == self.scope.client_id)

    def _doc_scope(self):
        return (Document.machine_line_id == self.scope.machine_line_id) & (Document.client_id == self.scope.client_id)

    # ---- documents ---------------------------------------------------------------------------------
    def documents(self, indexed_only: bool = True) -> list[Document]:
        q = select(Document).where(self._doc_scope())
        if indexed_only:
            q = q.where(Document.ingest_status == IngestStatus.indexed)
        return list(self.db.scalars(q.order_by(Document.component.nulls_first(), Document.title)))

    def filter_doc_ids(self, ids) -> set[UUID]:
        """Subset of `ids` that really belong to this scope. Foreign or made-up ids vanish here."""
        parsed = set()
        for i in ids or []:
            try:
                parsed.add(i if isinstance(i, UUID) else UUID(str(i)))
            except (ValueError, AttributeError):
                continue
        if not parsed:
            return set()
        rows = self.db.scalars(select(Document.id).where(self._doc_scope(), Document.id.in_(parsed)))
        return set(rows)

    # ---- search ------------------------------------------------------------------------------------
    def _base(self, *cols):
        return (select(*cols).join(Document, Document.id == Chunk.document_id)
                .where(self._chunk_scope(), Document.ingest_status == IngestStatus.indexed))

    def _hit(self, row, **kw) -> Hit:
        c = row[0]
        d = row[1]
        return Hit(chunk_id=c.id, document_id=c.document_id, page=c.page, section_path=c.section_path,
                   content=c.content, content_type=c.content_type, doc_title=d.title, doc_type=d.doc_type,
                   component=d.component, **kw)

    def vector_search(self, embedding: list[float], k: int, doc_ids=None, content_types=None) -> list[Hit]:
        dist = Chunk.embedding.cosine_distance(embedding)
        q = self._base(Chunk, Document, dist.label("dist")).where(Chunk.embedding.is_not(None))
        q = self._narrow(q, doc_ids, content_types).order_by(dist).limit(k)
        try:  # pgvector >= 0.8: a filtered HNSW scan must not return fewer than k rows
            with self.db.begin_nested():
                self.db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
                rows = self.db.execute(q).all()
        except Exception:  # noqa: BLE001 - older pgvector: fall back to the plain query
            rows = self.db.execute(q).all()
        return [self._hit(r, vector_distance=float(r.dist)) for r in rows]

    def fts_search(self, query: str, k: int, doc_ids=None, content_types=None) -> list[Hit]:
        qs = fts_query_string(query)
        if not qs:
            return []
        tsq = func.to_tsquery(literal_column("'simple'::regconfig"), qs)
        rank = func.ts_rank_cd(Chunk.tsv, tsq)
        q = self._base(Chunk, Document, rank.label("rank")).where(Chunk.tsv.op("@@")(tsq))
        q = self._narrow(q, doc_ids, content_types).order_by(rank.desc()).limit(k)
        return [self._hit(r, fts_rank=float(r.rank)) for r in self.db.execute(q).all()]

    def _narrow(self, q, doc_ids, content_types):
        if doc_ids:
            q = q.where(Chunk.document_id.in_(list(doc_ids)))  # AND-ed with the scope predicate above
        if content_types:
            q = q.where(Chunk.content_type.in_(list(content_types)))
        return q

    def read_pages(self, document_id: UUID, page_from: int, page_to: int) -> list[Hit]:
        q = (self._base(Chunk, Document).where(Chunk.document_id == document_id,
                                               Chunk.page.between(page_from, page_to))
             .order_by(Chunk.page, Chunk.id))
        return [self._hit(r) for r in self.db.execute(q).all()]
