from enum import StrEnum


class MachineKind(StrEnum):
    machine = "machine"
    line = "line"


class DocType(StrEnum):
    operating_manual = "operating_manual"
    instructions = "instructions"
    wiring_diagram = "wiring_diagram"
    flow_diagram = "flow_diagram"
    parameters = "parameters"
    parts = "parts"
    datasheet = "datasheet"
    mixed = "mixed"
    other = "other"


class IngestStatus(StrEnum):
    pending = "pending"
    indexed = "indexed"
    skipped = "skipped"
    failed = "failed"


class AgentAction(StrEnum):
    ANSWER = "ANSWER"
    EXPAND = "EXPAND"
    ASK = "ASK"
    ESCALATE = "ESCALATE"
    ERROR = "ERROR"  # a run that ended in a model or infrastructure failure (not a decision)
