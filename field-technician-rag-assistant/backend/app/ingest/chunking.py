"""Split document text into embed-sized chunks, preserving page + section for citations.

Strategy: respect natural boundaries. Split a page into paragraphs, then pack whole paragraphs
(or sentences, for long ones) into chunks up to a target size, carrying a short overlap so an
answer that straddles a boundary is still retrievable. This beats blind fixed-width cuts, which
slice mid-sentence and hurt both retrieval and citation quality.

We also track the nearest **section heading** while walking the text and attach it to each chunk
(and prefix the chunk content with it), so retrieval has structural context and citations can say
"page 109, section 5.2 Drum motor" instead of just a page number.
"""

import re
from dataclasses import dataclass

CHUNK_SIZE = 1000  # target max characters per chunk (~250-300 tokens)
CHUNK_OVERLAP = 150  # characters of continuity carried between adjacent chunks

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n+")

# A numbered heading like "5", "5.2", "5.2.1" followed by a short title.
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S.{0,80}$")

# Running page header/footer, not a real heading: "12 | Maintenance" (chapter marker) or
# "132 / 276 5S0049_Instructions" (page-of-total + filename). Both repeat on every page of a
# section/chapter and would otherwise be misdetected as the heading in effect, orphaning the
# real content under a wrong, less specific label.
_RUNNING_HEADER_FOOTER = re.compile(r"^\d+\s*[|/]\s")


@dataclass
class TextChunk:
    page: int
    content: str
    section: str | None = None
    kind: str = "text"


def _is_heading(line: str) -> bool:
    """Heuristic: a numbered heading, or a short ALL-CAPS line (a section title)."""
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if _RUNNING_HEADER_FOOTER.match(line):
        return False
    if _NUMBERED_HEADING.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    # Short, mostly-uppercase, multi-char line -> a title like "SAFETY INSTRUCTIONS".
    return len(letters) >= 3 and line.upper() == line and any(c.isalpha() for c in line)


def _hard_split(text: str) -> list[str]:
    """Last resort: split an over-long unit on character windows with overlap."""
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step)]


def _atomize(paragraph: str) -> list[str]:
    """Break a paragraph into units no larger than CHUNK_SIZE (by sentence, then by force)."""
    if len(paragraph) <= CHUNK_SIZE:
        return [paragraph]
    units: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        units.extend([sentence] if len(sentence) <= CHUNK_SIZE else _hard_split(sentence))
    return units


def _pack(units: list[str]) -> list[str]:
    """Greedily pack units into chunks, overlapping by the trailing unit when it's short."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        addition = len(unit) + (1 if current else 0)
        if current and length + addition > CHUNK_SIZE:
            chunks.append(" ".join(current))
            if len(current[-1]) < CHUNK_OVERLAP:  # carry a short tail for continuity
                current = [current[-1], unit]
                length = len(current[-1]) + 1 + len(unit)
            else:
                current = [unit]
                length = len(unit)
        else:
            current.append(unit)
            length += addition
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_page(page: int, text: str, start_section: str | None = None) -> list[TextChunk]:
    """Chunk a page's text on paragraph/sentence boundaries, tracking the section heading.

    `start_section` carries the heading in effect at the top of the page (headings span pages).
    Returns chunks tagged with the nearest preceding heading.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    section = start_section
    # Group runs of body paragraphs under the heading that precedes them, so each chunk gets
    # the right section even when a heading appears mid-page.
    segments: list[tuple[str | None, list[str]]] = []
    for paragraph in paragraphs:
        if _is_heading(paragraph):
            section = " ".join(paragraph.split())
            continue
        if segments and segments[-1][0] == section:
            segments[-1][1].append(paragraph)
        else:
            segments.append((section, [paragraph]))

    chunks: list[TextChunk] = []
    for seg_section, seg_paragraphs in segments:
        units: list[str] = []
        for paragraph in seg_paragraphs:
            units.extend(_atomize(" ".join(paragraph.split())))
        for content in _pack(units):
            prefixed = f"[Section: {seg_section}]\n{content}" if seg_section else content
            chunks.append(TextChunk(page=page, content=prefixed, section=seg_section, kind="text"))
    return chunks


def last_heading(text: str) -> str | None:
    """The last heading seen on a page — carried into the next page's chunking."""
    section: str | None = None
    for line in _PARAGRAPH_SPLIT.split(text):
        if _is_heading(line):
            section = " ".join(line.split())
    return section


def build_outline(chunks: list[TextChunk], page_count: int) -> list[dict]:
    """Router catalog for one document: [{id, title, level, page_from, page_to}], derived from the
    section headings seen while chunking. Best effort; an empty outline just means the router
    falls back to document-level routing."""
    entries: list[dict] = []
    for c in chunks:
        if not c.section:
            continue
        if entries and entries[-1]["title"] == c.section:
            entries[-1]["page_to"] = max(entries[-1]["page_to"], c.page)
            continue
        level = c.section.split()[0].count(".") + 1 if _NUMBERED_HEADING.match(c.section) else 1
        entries.append({"id": f"s{len(entries) + 1}", "title": c.section[:90], "level": level,
                        "page_from": c.page, "page_to": c.page})
    return entries[:200]
