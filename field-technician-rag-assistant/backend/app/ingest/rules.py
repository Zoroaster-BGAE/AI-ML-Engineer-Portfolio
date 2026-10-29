"""Folder/filename conventions -> metadata. The folder tree IS the schema (PRD section 5)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.domain.enums import DocType

# Ordered filename rules (case-insensitive); first match wins. Mirrors the PRD's table; client.yaml may
# prepend its own rules.
DEFAULT_RULES: list[tuple[str, str]] = [
    (r"(manual|instructions).*wiring", DocType.mixed),
    (r"wiring|wd_en|\bwd\b", DocType.wiring_diagram),
    (r"flow diagram", DocType.flow_diagram),
    (r"instructions", DocType.instructions),
    (r"manual", DocType.operating_manual),
    (r"parameters", DocType.parameters),
]

_GREEK = dict(zip("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρσςτυφχψω",
                  ["A", "V", "G", "D", "E", "Z", "I", "Th", "I", "K", "L", "M", "N", "X", "O", "P", "R", "S", "T", "Y",
                   "F", "Ch", "Ps", "O", "a", "v", "g", "d", "e", "z", "i", "th", "i", "k", "l", "m", "n", "x", "o",
                   "p", "r", "s", "s", "t", "y", "f", "ch", "ps", "o"], strict=True))


def slugify(text: str) -> str:
    text = "".join(_GREEK.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def classify_doc_type(filename: str, extra_rules: list[tuple[str, str]] | None = None) -> str:
    for pattern, doc_type in [*(extra_rules or []), *DEFAULT_RULES]:
        if re.search(pattern, filename, re.IGNORECASE):
            return str(doc_type)
    return DocType.other.value


def detect_languages(text: str) -> list[str]:
    langs = []
    letters = [c for c in text if c.isalpha()]
    if letters and sum("\u0370" <= c <= "\u03ff" for c in letters) / len(letters) > 0.05:
        langs.append("el")
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if words:
        de = sum(w in {"der", "die", "das", "und", "nicht", "mit", "ist", "für", "wird", "bei"} for w in words)
        en = sum(w in {"the", "and", "of", "to", "is", "with", "for", "not", "check", "must"} for w in words)
        if de > en * 0.5 and de > 5:
            langs.append("de")
        if en > 5:
            langs.append("en")
    return langs or ["en"]


def pretty_component(folder: str) -> str:
    """'PRESS SEP75HD PLUS SL DEPR755103000411908 USED (2019)' -> 'Press SEP75HD Plus SL'. Cosmetic only;
    it is a chip label, never used for routing keys."""
    s = re.sub(r"\(.*?\)", " ", folder)
    words = []
    for w in s.split():
        if len(w) >= 10 and any(c.isdigit() for c in w) and any(c.isalpha() for c in w):
            continue  # long serial-like token
        if w.upper() in {"USED", "OLD"}:
            continue
        words.append(w if any(c.isdigit() for c in w) or len(w) <= 2 else w.capitalize())
    return " ".join(words).strip() or folder


_YEAR = re.compile(r"(?<![A-Za-z0-9])(?:19|20)\d{2}(?![A-Za-z0-9])")
_SERIAL = re.compile(r"(?<![A-Za-z0-9])(?:\d[A-Z]\d{4}|\d{6})(?![A-Za-z0-9])")


def parse_machine_label(label: str) -> dict:
    """model_label / year / serial_ref from a folder like 'FEEDER KLIQ PLUS 3300_3R0348 (2026)'.
    serial_ref is opaque display text (PRD: never used for routing)."""
    year = _YEAR.findall(label)
    m_year = _YEAR.search(label)
    serial = _SERIAL.search(label)
    model = re.sub(r"\(.*?\)", "", label)
    if serial:
        model = model.replace(serial.group(0), "")
    model = re.sub(r"[_\s]+", " ", model).strip(" -_")
    return {"model_label": model or None, "year": int(m_year.group(0)) if year and m_year else None,
            "serial_ref": serial.group(0) if serial else None}


@dataclass
class ClientConfig:
    slug: str
    name: str
    rules: list[tuple[str, str]] = field(default_factory=list)


def load_client_config(client_dir: Path, slug: str | None = None, name: str | None = None) -> ClientConfig:
    cfg: dict = {}
    yml = client_dir / "client.yaml"
    if yml.exists():
        cfg = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    rules = [(r["pattern"], r["doc_type"]) for r in cfg.get("rules", [])]
    return ClientConfig(slug=slug or cfg.get("slug") or slugify(client_dir.name),
                        name=name or cfg.get("name") or client_dir.name, rules=rules)
