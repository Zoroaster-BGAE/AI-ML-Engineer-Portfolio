"""Pydantic contract for every node's structured output (PRD section 6). The LLM is constrained to these.

Every field is REQUIRED and non-nullable, on purpose. Measured against the live model: optional (defaulted) or
nullable (`X | None`) properties made constrained decoding loop and truncate in most calls, while the same schema
with all-required fields returned valid JSON every time. "Not applicable" is therefore "" / 0 / [], never null,
and there are no Python defaults (the SDK turns a default into "optional").
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DocTypeName = Literal["operating_manual", "instructions", "wiring_diagram", "flow_diagram", "parameters",
                      "parts", "datasheet", "mixed", "other"]


class Slots(BaseModel):
    error_codes: list[str]
    component_hint: str
    symptom: str


class PlanItem(BaseModel):
    document_id: str  # a catalog alias such as "D3"; validated against the scope before use
    section_hints: list[str]
    reason: str


class Understanding(BaseModel):
    """`understand` and `route` share ONE model call (PRD section 12 allows merging them once measurements
    justify it: each sequential structured call costs ~3-5 s). The routing half is still validated by code in
    the separate `route` step, so a bad plan can never widen the scope."""

    intent: Literal["troubleshoot", "procedure", "information", "smalltalk", "other"]
    slots: Slots
    wanted_doc_types: list[DocTypeName]
    standalone_query: str
    language: Literal["el", "en", "de", "other"]
    plan: list[PlanItem]
    rationale: str


class RetrieveAction(BaseModel):
    action: Literal["search", "read_pages", "finish"]
    query: str        # for action=search, else ""
    document_id: str  # catalog alias for action=read_pages, else ""
    page_from: int    # 0 when unused
    page_to: int      # 0 when unused
    rationale: str


class RetrieveDecision(BaseModel):
    actions: list[RetrieveAction]
    rationale: str


class Grade(BaseModel):
    sufficient: bool
    missing: list[str]
    llm_confidence: float
    rationale: str


class Reformulation(BaseModel):
    queries: list[str]
    widen: Literal["none", "siblings", "all_types"]
    rationale: str


class Clarify(BaseModel):
    question: str
    options: list[str]  # only used for the "symptom" facet, else []


class MachineGuess(BaseModel):
    """The Questioner agent's structured output. Aliases are validated against the catalog by code."""

    machine: str
    candidates: list[str]
    question: str
    has_issue: bool
    language: Literal["el", "en", "de", "other"]
    rationale: str


class SafetyVerdict(BaseModel):
    verdict: Literal["approve", "amend", "block"]
    issues: list[str]    # what is wrong, in English, for the reviewers
    warnings: list[str]  # amend only: warnings to show above the answer, in the technician's language
    rationale: str
