"""Router catalog: what the LLM is shown about the scope's documents. Short aliases (D1, D2...) stand in
for UUIDs so the model cannot mistype an id, and anything it names that is not in the catalog is dropped."""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.domain.models import Document


@dataclass
class CatalogDoc:
    alias: str
    id: UUID
    title: str
    doc_type: str
    component: str | None
    languages: list[str]
    page_count: int | None
    headings: list[str] = field(default_factory=list)


class Catalog:
    def __init__(self, docs: list[Document], max_headings: int = 10):
        self.docs: list[CatalogDoc] = []
        for i, d in enumerate(docs, start=1):
            heads = [o["title"] for o in (d.outline or []) if o.get("level", 1) <= 2][:max_headings]
            self.docs.append(CatalogDoc(f"D{i}", d.id, d.title, d.doc_type, d.component, d.languages or [],
                                        d.page_count, heads))
        self._by_alias = {c.alias.upper(): c for c in self.docs}
        self._by_id = {str(c.id): c for c in self.docs}

    def resolve(self, ref: str | None) -> CatalogDoc | None:
        """Alias or UUID string -> catalog entry, or None (foreign / invented ids are rejected here)."""
        if not ref:
            return None
        ref = str(ref).strip()
        return self._by_alias.get(ref.upper()) or self._by_id.get(ref)

    def by_id(self, doc_id: UUID) -> CatalogDoc | None:
        return self._by_id.get(str(doc_id))

    def restrict(self, *, component: str | None = None, doc_ids: list[str] | None = None,
                 doc_type: str | None = None) -> Catalog:
        """A narrowed view (technician-confirmed filters). A filter that would empty the catalog is ignored."""
        keep = self.docs
        for pred in (
            (lambda c: c.component == component) if component else None,
            (lambda c: str(c.id) in set(doc_ids)) if doc_ids else None,
            (lambda c: c.doc_type in (doc_type, "mixed")) if doc_type else None,
        ):
            if pred and (narrowed := [c for c in keep if pred(c)]):
                keep = narrowed
        view = Catalog.__new__(Catalog)
        view.docs = keep
        view._by_alias = {c.alias.upper(): c for c in keep}
        view._by_id = {str(c.id): c for c in keep}
        return view

    def render(self) -> str:
        lines = []
        for c in self.docs:
            head = f" | headings: {'; '.join(c.headings)}" if c.headings else ""
            lines.append(f"{c.alias} | {c.title} | type={c.doc_type} | component={c.component or '-'} | "
                         f"langs={','.join(c.languages) or '-'} | pages={c.page_count or '?'}{head}")
        return "\n".join(lines)
