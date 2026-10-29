"""Interview facet selection. CODE picks the facet with the highest information gain over the candidate set;
the LLM only phrases the question (PRD section 6)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.retrieval.repository import Hit

DOC_TYPE_LABEL = {
    "operating_manual": "Operating manual", "instructions": "Instructions", "wiring_diagram": "Wiring diagram",
    "flow_diagram": "Flow diagram", "parameters": "Parameters", "parts": "Parts list", "datasheet": "Datasheet",
    "mixed": "Manual + wiring", "other": "Other document",
}


@dataclass
class Option:
    label: str
    effect: dict          # routing effect: {"component": str} | {"doc_ids": [uuid]} | {"doc_type": str}
    weight: float = 0.0


@dataclass
class Facet:
    name: str             # component | document | doc_type
    options: list[Option] = field(default_factory=list)
    entropy: float = 0.0
    margin: float = 1.0   # top share minus runner-up share


def _facet(name: str, weights: dict[str, tuple[Option, float]]) -> Facet | None:
    if len(weights) < 2:
        return None
    total = sum(w for _, w in weights.values()) or 1.0
    shares = sorted(((w / total, o) for o, w in weights.values()), key=lambda t: t[0], reverse=True)
    for s, o in shares:
        o.weight = s
    entropy = -sum(s * math.log2(s) for s, _ in shares if s > 0)
    return Facet(name, [o for _, o in shares], entropy, shares[0][0] - shares[1][0])


def candidate_facets(hits: list[Hit], *, use_doc_type: bool = True) -> list[Facet]:
    """Distributions of retrieval score mass over each facet's values, best information gain first."""
    comp: dict[str, tuple[Option, float]] = {}
    docs: dict[str, tuple[Option, float]] = {}
    types: dict[str, tuple[Option, float]] = {}
    for h in hits:
        w = h.rrf_score or 1e-6
        if h.component:
            o, acc = comp.get(h.component, (Option(h.component, {"component": h.component}), 0.0))
            comp[h.component] = (o, acc + w)
        o, acc = docs.get(str(h.document_id), (Option(h.doc_title, {"doc_ids": [str(h.document_id)]}), 0.0))
        docs[str(h.document_id)] = (o, acc + w)
        o, acc = types.get(h.doc_type, (Option(DOC_TYPE_LABEL.get(h.doc_type, h.doc_type), {"doc_type": h.doc_type}), 0.0))
        types[h.doc_type] = (o, acc + w)
    out = [f for f in (_facet("component", comp), _variant_facet(hits, docs), _facet("doc_type", types) if use_doc_type else None) if f]
    return sorted(out, key=lambda f: f.entropy, reverse=True)


def _variant_facet(hits: list[Hit], docs: dict) -> Facet | None:
    """'document' facet = sibling variants: several documents sharing (component, doc_type), e.g. the three
    DT140 wiring sets. Only meaningful when such siblings are actually among the hits."""
    groups: dict[tuple, set] = {}
    for h in hits:
        groups.setdefault((h.component, h.doc_type), set()).add(str(h.document_id))
    if not any(len(ids) > 1 for ids in groups.values()):
        return None
    sibling_ids = {i for ids in groups.values() if len(ids) > 1 for i in ids}
    return _facet("document", {k: v for k, v in docs.items() if k in sibling_ids})


def pick_ambiguous(facets: list[Facet], asked: list[str], margin_threshold: float) -> Facet | None:
    """First facet (by information gain) not asked yet whose top candidate does not clearly win."""
    for f in facets:
        if f.name not in asked and f.margin < margin_threshold:
            return f
    return None


def any_askable(facets: list[Facet], asked: list[str]) -> Facet | None:
    return next((f for f in facets if f.name not in asked), None)
