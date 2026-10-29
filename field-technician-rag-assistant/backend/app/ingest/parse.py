"""PDF ingestion (PRD FR-22): pdfplumber text + tables, chunk, outline -- and page images of drawings.

Drawing pages (wiring diagrams, schematics: dense vector graphics) and scanned pages (one big image, no text)
are also rendered to PNG so they can be shown to the technician as they are. Text alone loses what a wiring
sheet is for.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from app.ingest.chunking import TextChunk, build_outline, chunk_page, last_heading
from app.ingest.rules import detect_languages

_DENSE_GRAPHICS = 300  # lines+rects above this = a drawing; skip table detection (slow, mostly noise)
_RENDER_DPI = 150
_MAX_EDGE_PX = 3000     # an A1 sheet at 150 dpi is ~5000 px; this keeps a figure under ~1.5 MB and readable
_SCANNED_COVER = 0.5    # one image over half the page and almost no text = a scanned page


@dataclass
class Figure:
    page: int
    kind: str           # drawing | scanned
    png: bytes
    width: int
    height: int


@dataclass
class ParsedDoc:
    page_count: int
    chunks: list[TextChunk] = field(default_factory=list)
    outline: list[dict] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    empty_pages: int = 0
    figures: list[Figure] = field(default_factory=list)


def _table_text(page) -> str:
    rows = []
    for table in page.extract_tables() or []:
        for row in table:
            cells = [" ".join((c or "").split()) for c in row]
            if any(cells):
                rows.append(" | ".join(cells))
    return "\n".join(rows)


def _figure_kind(page, text: str) -> str | None:
    if len(page.lines) + len(page.rects) >= _DENSE_GRAPHICS:
        return "drawing"
    if len(text.strip()) < 5 and page.images:
        area = float(page.width * page.height) or 1.0
        cover = max((abs((i["x1"] - i["x0"]) * (i["bottom"] - i["top"])) for i in page.images), default=0.0)
        if cover / area >= _SCANNED_COVER:
            return "scanned"
    return None


def render(page) -> tuple[bytes, int, int]:
    dpi = min(_RENDER_DPI, _MAX_EDGE_PX * 72 / max(float(page.width), float(page.height), 1.0))
    im = page.to_image(resolution=dpi).original.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue(), im.width, im.height


def parse_pdf(path: Path, render_figures: bool = True) -> ParsedDoc:
    chunks: list[TextChunk] = []
    figures: list[Figure] = []
    section: str | None = None
    sample: list[str] = []
    empty = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
                if (len(page.lines) + len(page.rects)) < _DENSE_GRAPHICS:
                    tt = _table_text(page)
                    if tt:
                        chunks.append(TextChunk(page=i, content=f"[Table]\n{tt}"[:2500], section=section, kind="table"))
                kind = _figure_kind(page, text) if render_figures else None
                if kind:
                    png, w, h = render(page)
                    figures.append(Figure(i, kind, png, w, h))
            except Exception:  # noqa: BLE001 - one bad page must not sink a 300-page manual
                text = ""
            if len(text.strip()) < 5:
                empty += 1
                continue
            chunks.extend(chunk_page(i, text, section))
            section = last_heading(text) or section
            if len(sample) < 12:
                sample.append(text[:800])
            page.flush_cache()
    return ParsedDoc(page_count=page_count, chunks=chunks, outline=build_outline(chunks, page_count),
                     languages=detect_languages("\n".join(sample)), empty_pages=empty, figures=figures)
