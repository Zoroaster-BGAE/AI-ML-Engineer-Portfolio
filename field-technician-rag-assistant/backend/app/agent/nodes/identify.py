"""The Questioner agent: find out which machine an unscoped chat is about (design §2.2).

The model sees a catalog of every machine under short aliases (M1, M2, ...) and may only answer with those;
anything else is dropped here, so it cannot invent or widen a scope. Code, not the model, decides when the
guess is good enough to lock, and a technician's tap on a machine chip locks it without asking the model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import facets as fx
from app.agent.prompts import v1
from app.agent.schemas import MachineGuess
from app.domain.enums import IngestStatus
from app.domain.models import Client, Document, MachineLine
from app.domain.support import ChatSession
from app.llm.base import ImageInput
from app.obs.recorder import RunRecorder

MAX_CHIPS = 4


@dataclass
class CatalogMachine:
    alias: str
    id: UUID
    client_id: UUID
    label: str
    client: str
    line: str


@dataclass
class MachineCatalog:
    machines: list[CatalogMachine] = field(default_factory=list)

    @classmethod
    def load(cls, db: Session) -> MachineCatalog:
        rows = db.execute(select(MachineLine, Client).join(Client, Client.id == MachineLine.client_id)
                          .where(Client.is_active.is_(True)).order_by(Client.name, MachineLine.label)).all()
        docs: dict[UUID, dict[str, int]] = defaultdict(dict)
        comps: dict[UUID, set[str]] = defaultdict(set)
        for ml_id, doc_type, component, n in db.execute(
                select(Document.machine_line_id, Document.doc_type, Document.component, func.count())
                .where(Document.ingest_status == IngestStatus.indexed)
                .group_by(Document.machine_line_id, Document.doc_type, Document.component)):
            docs[ml_id][doc_type] = docs[ml_id].get(doc_type, 0) + n
            if component:
                comps[ml_id].add(component)
        out = cls()
        for i, (ml, client) in enumerate(rows, start=1):
            kinds = ", ".join(f"{fx.DOC_TYPE_LABEL.get(t, t)} x{n}" for t, n in sorted(docs[ml.id].items())) or "none yet"
            parts = [f"M{i}", f"client: {client.name}", ml.label, f"kind={ml.kind}"]
            if ml.model_label:
                parts.append(f"model={ml.model_label}")
            if ml.year:
                parts.append(f"year={ml.year}")
            if comps[ml.id]:
                parts.append("components: " + "; ".join(sorted(comps[ml.id])))
            parts.append(f"documents: {kinds}")
            out.machines.append(CatalogMachine(f"M{i}", ml.id, ml.client_id, ml.label, client.name, " | ".join(parts)))
        return out

    def resolve(self, alias: str | None) -> CatalogMachine | None:
        a = (alias or "").strip().upper()
        return next((m for m in self.machines if m.alias == a), None)

    def by_id(self, machine_id) -> CatalogMachine | None:
        return next((m for m in self.machines if str(m.id) == str(machine_id)), None)

    def render(self) -> str:
        return "\n".join(m.line for m in self.machines) or "(the library is empty)"


@dataclass
class Identification:
    machine: CatalogMachine | None
    candidates: list[CatalogMachine]
    guess: MachineGuess


def _recent(session: ChatSession, limit: int = 8) -> str:
    msgs = list(session.messages)[-(limit + 1):-1]
    return "\n".join(f"{m.role}: {m.content[:400]}" for m in msgs) or "(none)"


def identify(db: Session, rec: RunRecorder, catalog: MachineCatalog, session: ChatSession, text: str,
             images: list[ImageInput]) -> Identification:
    user = (f"CATALOG\n{catalog.render()}\n\nConversation so far:\n{_recent(session)}\n\n"
            f"Latest message: {text}" + (f"\n({len(images)} photo(s) attached)" if images else ""))
    guess = rec.llm_step("identify", v1.IDENTIFY, user, MachineGuess, max_tokens=1500, images=images or None,
                         note={"catalog_size": len(catalog.machines)})
    machine = catalog.resolve(guess.machine)
    seen, cands = set(), []
    for a in ([guess.machine] if machine else []) + guess.candidates:
        m = catalog.resolve(a)
        if m and m.id not in seen:
            seen.add(m.id)
            cands.append(m)
    rec.code_step("identify_validate", inp={"machine": guess.machine, "candidates": guess.candidates},
                  out={"locked": machine.label if machine else None, "candidates": [c.label for c in cands]},
                  rationale=guess.rationale)
    return Identification(machine, cands[:MAX_CHIPS], guess)
