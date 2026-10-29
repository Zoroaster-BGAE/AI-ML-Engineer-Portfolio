"""Pre-display safety review (design §3.6). Every synthesized answer is judged before the technician sees it.

Three parts, none trusted alone: the policy in the SYNTHESIZE prompt, deterministic keyword tripwires, and a
reviewer call that approves, amends (adds warnings above the answer) or blocks. A reviewer failure raises and
the turn ends without showing the draft: this check fails closed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.agent.context import TurnContext, msg
from app.agent.prompts import v1
from app.agent.schemas import SafetyVerdict
from app.obs import metrics
from app.retrieval.repository import Hit

_LANG = {"el": "Greek", "en": "English", "de": "German", "other": "the language the technician wrote in"}

# Written without accents: text is lower-cased and stripped of diacritics before matching, so Greek with or
# without tonos, and upper-case panel-style text, match the same pattern.
_TRIPWIRES = [
    r"\bby-?pass(es|ed|ing)?\b",
    r"\bjumper(s|ed|ing)?\b",
    r"\bbridg(e|es|ed|ing)\s+(it\s+)?(out|over)\b",
    r"\bdefeat(s|ed|ing)?\s+(the\s+)?(interlock|safety|switch|lock|guard|sensor)",
    r"\boverrid(e|es|ing|den)\s+(the\s+)?(interlock|safety|door|switch|lock|guard|sensor|thermal|limit)",
    r"\bdisabl(e|es|ed|ing)\s+(the\s+)?(interlock|safety|door\s*lock|limit\s*switch|thermal|guard|e-?stop|emergency)",
    r"\btap(e|ing)\s+(down|over)\s+(the\s+)?(switch|sensor|button)",
    r"\bwhile\s+(it\s+is\s+|it'?s\s+|the\s+machine\s+is\s+)?(running|energi[sz]ed|powered|live)\b",
    r"\bwith\s+(the\s+)?power\s+(on|connected|applied)\b",
    r"παρακα?μψ",                                                       # παράκαμψη / παρακάμψτε
    r"γεφυρω",                                                          # γεφύρωση / γεφυρώστε
    r"απενεργοποι\w*\s+(\w+\s+)?(ασφαλει|μανταλ|κλειδαρια|interlock|θερμικ)",
    r"ενω\s+(το\s+μηχανημα\s+)?(λειτουργει|δουλευει|ειναι\s+(σε\s+λειτουργια|αναμμενο|υπο\s+ταση))",
    r"υπο\s+ταση",
    r"\buberbruck",                                                     # überbrücken
    r"\bumgeh(en|t)\b",
]
_TRIP = re.compile("|".join(f"(?:{p})" for p in _TRIPWIRES), re.IGNORECASE | re.UNICODE)


# A rating comment that talks about danger goes to the top of the lead technician's queue.
_CONCERN = re.compile(r"\b(unsafe|dangerous|danger|hazard\w*|injur\w*|electric\s*shock|shock(ed)?|burn(t|ed)?|"
                      r"safety)\b|επικινδυν|κινδυν|ασφαλει|ηλεκτροπληξ|εγκαυμ|τραυματ|gefahr|gefahrlich|unsicher",
                      re.IGNORECASE | re.UNICODE)


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def tripwires(text: str) -> list[str]:
    """Matched phrases, de-duplicated. Empty means no keyword fired."""
    return sorted({m.group(0) for m in _TRIP.finditer(_fold(text or ""))})


def mentions_safety(text: str | None) -> bool:
    return bool(text) and bool(_CONCERN.search(_fold(text)))


@dataclass
class Review:
    verdict: SafetyVerdict
    strict: bool
    question_hits: list[str]
    draft_hits: list[str]

    @property
    def blocked(self) -> bool:
        return self.verdict.verdict == "block"

    @property
    def amended(self) -> bool:
        return self.verdict.verdict == "amend" and bool(self.verdict.warnings)

    @property
    def needs_attention(self) -> bool:
        """A human should look: anything but a clean approval, or a draft that tripped a keyword."""
        return self.blocked or self.amended or bool(self.draft_hits)


def review(ctx: TurnContext, draft: str, chunks: list[Hit]) -> Review:
    q_hits, d_hits = tripwires(ctx.user_text), tripwires(draft)
    strict = bool(q_hits or d_hits)
    excerpts = "\n\n".join(f"[{i}] {h.doc_title} p.{h.page}\n{h.content}" for i, h in enumerate(chunks, start=1))
    user = (f"Machine: {ctx.machine_label}\nTechnician's question: {ctx.user_text}\n"
            f"Standalone question: {ctx.understanding.standalone_query}\n\n"
            f"MANUAL EXCERPTS\n{excerpts or '(none)'}\n\nDRAFT ANSWER\n{draft}"
            + (f"\n\n{v1.SAFETY_STRICT_NOTE} Matched: {', '.join(q_hits + d_hits)}" if strict else ""))
    system = v1.SAFETY_REVIEW.format(language=_LANG.get(ctx.lang, "English"))
    verdict = ctx.rec.llm_step("safety_review", system, user, SafetyVerdict, effort=ctx.settings.effort_safety,
                               max_tokens=2000, fast=True,
                               note={"strict": strict, "question_hits": q_hits, "draft_hits": d_hits})
    metrics.SAFETY_VERDICTS.labels(verdict.verdict, str(strict).lower()).inc()
    return Review(verdict, strict, q_hits, d_hits)


def render(ctx: TurnContext, r: Review, draft: str) -> str:
    """What the technician sees. Warnings are a block above the steps, never buried inside them."""
    if r.blocked:
        return msg("safety_blocked", ctx.lang)
    if r.amended:
        lines = "\n".join(f"> - {w.strip()}" for w in r.verdict.warnings if w.strip())
        return f"> **{msg('safety_header', ctx.lang)}**\n{lines}\n\n{draft}"
    return draft
