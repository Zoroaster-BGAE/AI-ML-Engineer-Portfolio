"""Folder tree -> rows. `<client>/<machine-or-line>/[<component>/]<file>` is the schema (PRD section 5).

    python -m app.cli.onboard <client-folder> [--dry-run] [--only SUBSTR] [--slug S] [--name N]

Idempotent: a document is identified by (machine_line, sha256); re-running changes nothing.
"""
from __future__ import annotations

import hashlib
import mimetypes
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.enums import IngestStatus, MachineKind
from app.domain.models import Chunk, Client, Document, Figure, MachineLine
from app.ingest.chunking import TextChunk
from app.ingest.parse import ParsedDoc, parse_pdf
from app.ingest.rules import (
    ClientConfig,
    classify_doc_type,
    load_client_config,
    parse_machine_label,
    pretty_component,
    slugify,
)
from app.llm.embed import Embedder
from app.media.storage import get_storage

_EMBED_CHARS = 500  # embedding cost is linear in text; the head of a chunk carries its meaning
_IGNORED = {"client.yaml", "thumbs.db", ".ds_store", "desktop.ini"}


@dataclass
class PlanDoc:
    path: Path
    component: str | None
    title: str
    doc_type: str
    media_type: str
    sha256: str
    skip_reason: str | None = None


@dataclass
class PlanMachine:
    slug: str
    label: str
    kind: str
    meta: dict
    docs: list[PlanDoc] = field(default_factory=list)


@dataclass
class Plan:
    client: ClientConfig
    machines: list[PlanMachine]
    unassigned: list[Path]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_plan(client_dir: Path, slug: str | None = None, name: str | None = None,
               only: str | None = None) -> Plan:
    cfg = load_client_config(client_dir, slug, name)
    machines: list[PlanMachine] = []
    unassigned = [p for p in client_dir.iterdir() if p.is_file() and p.name.lower() not in _IGNORED]
    for mdir in sorted(p for p in client_dir.iterdir() if p.is_dir()):
        if only and only.lower() not in mdir.name.lower():
            continue
        files = sorted(p for p in mdir.rglob("*") if p.is_file() and p.name.lower() not in _IGNORED)
        has_components = any(p.parent != mdir for p in files)
        pm = PlanMachine(slug=slugify(mdir.name), label=mdir.name,
                         kind=(MachineKind.line if has_components else MachineKind.machine).value,
                         meta=parse_machine_label(mdir.name))
        for f in files:
            rel = f.relative_to(mdir)
            component = pretty_component(rel.parts[0]) if len(rel.parts) > 1 else None
            media = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            pdf = f.suffix.lower() == ".pdf"
            pm.docs.append(PlanDoc(
                path=f, component=component, title=f.stem, media_type=media,
                doc_type=classify_doc_type(f.name, cfg.rules) if pdf else "other",
                sha256=_sha256(f),
                skip_reason=None if pdf else f"unsupported file type ({f.suffix.lower() or 'none'}); not indexed in MVP",
            ))
        machines.append(pm)
    return Plan(cfg, machines, unassigned)


def render_plan(plan: Plan) -> str:
    lines = [f"CLIENT  {plan.client.slug}  <-  {plan.client.name}"]
    for m in plan.machines:
        comps = sorted({d.component for d in m.docs if d.component})
        lines.append(f"  {m.kind.upper():7} {m.slug}  ({m.label})  docs={len(m.docs)}"
                     + (f"  components={comps}" if comps else ""))
        for d in m.docs:
            tag = "SKIP " if d.skip_reason else "INDEX"
            lines.append(f"      {tag} [{d.doc_type:14}] {d.title}" + (f"  -- {d.skip_reason}" if d.skip_reason else ""))
    for p in plan.unassigned:
        lines.append(f"  (unassigned, ignored) {p.name}")
    total = sum(len(m.docs) for m in plan.machines)
    idx = sum(1 for m in plan.machines for d in m.docs if not d.skip_reason)
    lines.append(f"TOTAL  machines/lines={len(plan.machines)}  files={total}  to-index={idx}  skipped={total - idx}")
    return "\n".join(lines)


def apply_plan(db: Session, plan: Plan, embedder: Embedder, log=print, workers: int = 3, reindex: bool = False) -> dict:
    """`reindex` re-parses documents that are already indexed (e.g. to render their figures after an upgrade)."""
    stats = {"created_docs": 0, "indexed": 0, "skipped": 0, "failed": 0, "already": 0}
    client = db.scalar(select(Client).where(Client.slug == plan.client.slug))
    if client is None:
        client = Client(slug=plan.client.slug, name=plan.client.name)
        db.add(client)
        db.flush()
    todo: list[tuple[Document, Path]] = []
    for pm in plan.machines:
        ml = db.scalar(select(MachineLine).where(MachineLine.client_id == client.id, MachineLine.slug == pm.slug))
        if ml is None:
            ml = MachineLine(client_id=client.id, slug=pm.slug, label=pm.label, kind=pm.kind, **pm.meta)
            db.add(ml)
            db.flush()
        for pd in pm.docs:
            doc = db.scalar(select(Document).where(Document.machine_line_id == ml.id, Document.sha256 == pd.sha256))
            if doc is None:
                doc = Document(
                    client_id=client.id, machine_line_id=ml.id, component=pd.component, title=pd.title,
                    doc_type=pd.doc_type, media_type=pd.media_type, storage_uri=str(pd.path), sha256=pd.sha256,
                    ingest_status=IngestStatus.skipped if pd.skip_reason else IngestStatus.pending,
                    ingest_note=pd.skip_reason,
                )
                db.add(doc)
                db.flush()
                stats["created_docs"] += 1
            if doc.ingest_status == IngestStatus.indexed and not reindex:
                stats["already"] += 1
            elif doc.ingest_status == IngestStatus.skipped:
                stats["skipped"] += 1
            else:  # pending or failed -> (re)try
                todo.append((doc, pd.path))
        db.commit()

    # Parsing (pdfplumber) is CPU-bound and independent per file: fan it out to worker processes while the
    # main process embeds and stores whatever has finished parsing.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_pdf, path): doc for doc, path in todo}
        for fut in as_completed(futures):
            doc = futures[fut]
            try:
                parsed = fut.result()
            except Exception as exc:  # noqa: BLE001
                parsed = exc
            ok = _index_document(db, doc, parsed, embedder, log)
            stats["indexed" if ok else "failed"] += 1
            db.commit()
    return stats


def _store_figures(db: Session, doc: Document, parsed: ParsedDoc) -> None:
    db.query(Figure).filter(Figure.document_id == doc.id).delete()
    storage = get_storage()
    for f in parsed.figures:
        key = f"derived/{doc.id}/p{f.page}.png"
        storage.put(key, f.png, "image/png")
        db.add(Figure(document_id=doc.id, machine_line_id=doc.machine_line_id, client_id=doc.client_id,
                      page=f.page, kind=f.kind, storage_key=key, width=f.width, height=f.height))


def _index_document(db: Session, doc: Document, parsed: ParsedDoc | Exception, embedder: Embedder, log) -> bool:
    try:
        if isinstance(parsed, Exception):
            raise parsed
        if not parsed.chunks and parsed.figures:
            # A scanned or drawing-only PDF: nothing to read as text, but every sheet can be shown. One keyword
            # chunk per sheet makes it findable by title, component and page, and links a hit to its figure.
            label = f"{doc.title} ({doc.component})" if doc.component else doc.title
            parsed.chunks = [TextChunk(page=f.page, content=f"[Drawing] {label}, sheet on page {f.page}",
                                       kind="drawing") for f in parsed.figures]
        if not parsed.chunks:
            raise ValueError("no extractable text and no drawings (OCR is out of MVP scope)")
        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        _store_figures(db, doc, parsed)
        to_embed = [c for c in parsed.chunks if c.kind != "wiring_sheet"]  # drawings: keyword search only
        vectors = dict(zip(map(id, to_embed), embedder.embed([c.content[:_EMBED_CHARS] for c in to_embed]),
                           strict=True)) if to_embed else {}
        for c in parsed.chunks:
            db.add(Chunk(document_id=doc.id, machine_line_id=doc.machine_line_id, client_id=doc.client_id,
                         page=c.page, section_id=None, section_path=c.section, content_type=c.kind,
                         content=c.content, embedding=vectors.get(id(c))))
        drawings = len(parsed.chunks) - len(to_embed)
        doc.page_count, doc.outline, doc.languages = parsed.page_count, parsed.outline, parsed.languages
        doc.ingest_status = IngestStatus.indexed
        doc.ingest_note = (f"{len(parsed.chunks)} chunks" + (f" ({drawings} drawing pages, keyword-only)" if drawings else "")
                           + (f"; {len(parsed.figures)} figures" if parsed.figures else "")
                           + (f"; {parsed.empty_pages} textless pages" if parsed.empty_pages else ""))
        log(f"    indexed {doc.title}: {doc.ingest_note}")
        return True
    except Exception as exc:  # noqa: BLE001 - record and continue; never lose the rest of the client
        db.rollback()
        doc = db.merge(doc)
        doc.ingest_status, doc.ingest_note = IngestStatus.failed, f"{type(exc).__name__}: {exc}"[:500]
        log(f"    FAILED  {doc.title}: {doc.ingest_note}")
        return False
