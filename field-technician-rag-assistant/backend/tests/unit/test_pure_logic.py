"""Deterministic units: RRF fusion, confidence gate, verbatim guard, facet selection, filename rules."""
from uuid import uuid4

from app.agent import facets as fx
from app.agent.gate import GateSignals, code_pattern, compute_signals, decide
from app.agent.guard import SentenceGuard
from app.agent.schemas import Grade
from app.domain.enums import AgentAction
from app.ingest.rules import classify_doc_type, parse_machine_label, pretty_component, slugify
from app.retrieval.hybrid import rrf_fuse
from app.retrieval.repository import Hit, fts_query_string


def hit(doc, title="D", page=1, content="x", comp=None, typ="operating_manual", vec=None, fts=None, rrf=0.0):
    return Hit(uuid4(), doc, page, None, content, "text", title, typ, comp, vec, fts, rrf)


# ---- RRF --------------------------------------------------------------------------------------------
def test_rrf_rewards_chunks_found_by_both_retrievers():
    d = uuid4()
    both, only_v, only_f = hit(d, "both"), hit(d, "v"), hit(d, "f")
    fused = rrf_fuse([both, only_v], [both, only_f], k=60)
    assert fused[0].doc_title == "both"
    assert fused[0].rrf_score > fused[1].rrf_score
    assert len(fused) == 3


def test_fts_query_is_or_of_word_tokens_only():
    q = fts_query_string("E4 alarm; DROP TABLE x -- (drum)")
    assert " | " in q and ";" not in q and "(" not in q and "-" not in q


# ---- gate -------------------------------------------------------------------------------------------
def _sig(**kw):
    base = dict(n_hits=5, best_distance=0.3, exact_code_hit=False, plan_coverage=1.0, det_ok=True)
    return GateSignals(**{**base, **kw})


def _dec(sig, sufficient=True, **kw):
    args = dict(ambiguous=False, any_facet=False, expand_count=0, ask_turns=0, max_expand=2, max_ask=2) | kw
    return decide(sig, Grade(sufficient=sufficient, missing=[], llm_confidence=.5, rationale="r"), **args)


def test_gate_answers_when_signals_and_grader_agree():
    assert _dec(_sig())[0] == AgentAction.ANSWER


def test_gate_expands_then_escalates_when_nothing_relevant():
    weak = _sig(det_ok=False, best_distance=0.9)
    assert _dec(weak, expand_count=0)[0] == AgentAction.EXPAND
    assert _dec(weak, expand_count=2)[0] == AgentAction.ESCALATE          # a third loop never happens


def test_gate_asks_on_ambiguity_within_turn_budget_only():
    assert _dec(_sig(), ambiguous=True)[0] == AgentAction.ASK
    assert _dec(_sig(), ambiguous=True, ask_turns=2)[0] == AgentAction.ANSWER  # never a 3rd question


def test_gate_partial_answer_after_budget_when_evidence_exists():
    action, _, partial = _dec(_sig(), sufficient=False, expand_count=2)
    assert (action, partial) == (AgentAction.ANSWER, True)


def test_exact_code_beats_a_poor_vector_distance():
    d = uuid4()
    s = compute_signals([hit(d, content="ALARM E 4: motor", vec=0.95)], ["E4"], {d}, max_distance=0.6)
    assert s.exact_code_hit and s.det_ok


def test_code_pattern_does_not_match_inside_longer_tokens():
    p = code_pattern("E4")
    assert p.search("ALARM E4 TRIPPED") and p.search("E-4") and not p.search("BE41")


# ---- verbatim guard -----------------------------------------------------------------------------------
SOURCE = "Isolate the main power supply and wait for the drum to stop completely before opening the inspection door panel."


def _run_guard(text_out: str, n=8):
    g = SentenceGuard([SOURCE], n)
    out = "".join(p for i in range(0, len(text_out), 5) for p in g.feed(text_out[i:i + 5])) + "".join(g.flush())
    return out, g


def test_guard_drops_sentences_copied_from_the_manual():
    out, g = _run_guard(f"Before you start: switch it off [1]. {SOURCE} Then check the belt [2].")
    assert "inspection door panel" not in out and g.dropped == 1
    assert "check the belt" in out and g.cited == [1, 2]


def test_guard_keeps_paraphrases_and_strips_markers():
    out, g = _run_guard("Cut power first and let the drum stop [1].")
    assert "[1]" not in out and "Cut power first" in out and g.dropped == 0


# ---- facets -------------------------------------------------------------------------------------------
def test_component_facet_is_ambiguous_when_scores_are_close_and_clear_when_not():
    d1, d2 = uuid4(), uuid4()
    close = fx.candidate_facets([hit(d1, "P", comp="Press", rrf=.03), hit(d2, "D", comp="Dryers", rrf=.029)])
    assert fx.pick_ambiguous([f for f in close if f.name == "component"], [], 0.25).name == "component"
    clear = fx.candidate_facets([hit(d1, "P", comp="Press", rrf=.05), hit(d2, "D", comp="Dryers", rrf=.002)])
    assert fx.pick_ambiguous([f for f in clear if f.name == "component"], [], 0.25) is None


def test_asked_facets_are_never_asked_twice():
    d1, d2 = uuid4(), uuid4()
    fs = fx.candidate_facets([hit(d1, comp="Press", rrf=.03), hit(d2, comp="Dryers", rrf=.03)])
    assert fx.pick_ambiguous(fs, ["component", "document", "doc_type"], 0.25) is None


def test_variant_siblings_produce_a_document_facet():
    a, b = uuid4(), uuid4()
    fs = fx.candidate_facets([hit(a, "WIRINGS 01", comp="Dryers", typ="wiring_diagram", rrf=.03),
                              hit(b, "WIRINGS 02", comp="Dryers", typ="wiring_diagram", rrf=.03)])
    doc = next(f for f in fs if f.name == "document")
    assert {o.label for o in doc.options} == {"WIRINGS 01", "WIRINGS 02"}


# ---- filename rules (PRD section 5 table, real corpus names) -----------------------------------------
def test_filename_rules_match_the_prd_table():
    cases = {"Manual + WIRINGS_790337 Bottom Up.pdf": "mixed", "KliQ Plus 3300_3R0348_WD_EN.pdf": "wiring_diagram",
             "FLOW DIAGRAM 16x72.pdf": "flow_diagram", "Silverline 3300_5S0046_Instructions.pdf": "instructions",
             "220623 MANUAL.pdf": "operating_manual", "INVERTER PARAMETERS.xlsx": "parameters",
             "Deloading conveyor, horizontal 4000-800.pdf": "other"}
    for name, expected in cases.items():
        assert classify_doc_type(name) == expected, name


def test_slug_and_label_parsing_on_real_folder_names():
    assert slugify("ΠΑΠΑΔΟΠΟΥΛΟΣ - SUNRISE LAUNDRY - ATHENS") == "papadopoylos-sunrise-laundry-athens"
    m = parse_machine_label("FEEDER KLIQ PLUS 3300_3R0348 (2026)")
    assert m["year"] == 2026 and m["serial_ref"] == "3R0348"
    assert pretty_component("PRESS SEP75HD PLUS SL DEPR755103000411908 USED (2019)") == "Press SEP75HD Plus SL"
