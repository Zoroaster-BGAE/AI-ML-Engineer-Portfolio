from prometheus_client import Counter, Histogram

DECISIONS = Counter("agent_decisions_total", "Final agent action per turn", ["action"])
INTERVIEW_TURNS = Counter("interview_turns_total", "Clarifying questions asked", ["facet"])
NODE_SECONDS = Histogram("agent_node_seconds", "Latency per agent node", ["node"],
                         buckets=(.05, .1, .25, .5, 1, 2, 4, 8, 16, 32, 64))
TURN_SECONDS = Histogram("agent_turn_seconds", "End-to-end turn latency", buckets=(1, 2, 4, 8, 16, 32, 64, 128))
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["direction"])
GUARD_DROPS = Counter("verbatim_guard_drops_total", "Sentences dropped by the verbatim guard")
SAFETY_VERDICTS = Counter("safety_verdicts_total", "Pre-display safety review outcomes", ["verdict", "strict"])
