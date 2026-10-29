from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Scope:
    """The technician's locked selection. Built only from a chat_session row, never from a
    request payload or an LLM output; retrieval.repository refuses to run without one."""

    client_id: UUID
    machine_line_id: UUID
