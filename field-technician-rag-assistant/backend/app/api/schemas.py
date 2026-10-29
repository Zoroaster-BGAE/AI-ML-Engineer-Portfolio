"""Response WHITELISTS (FR-14). Nothing here may carry `content`, `storage_uri` or a file path; a contract
test walks every technician-facing schema to prove it."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Orm(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserOut(Orm):
    id: UUID
    email: str
    role: str
    is_active: bool
    access_expires: datetime | None = None
    can_review: bool = False


class ClientOut(Orm):
    id: UUID
    name: str


class MachineOut(BaseModel):
    id: UUID
    label: str
    kind: str
    year: int | None = None
    document_count: int
    component_count: int


class SessionCreate(BaseModel):
    """Both empty (the usual case): the Questioner agent identifies the machine. Both set: a pre-scoped chat."""

    client_id: UUID | None = None
    machine_line_id: UUID | None = None


class SessionRatingOut(BaseModel):
    stars: int
    comment: str | None = None


class SessionOut(BaseModel):
    id: UUID
    title: str | None
    machine_label: str | None = None   # None until the machine is identified
    kind: str | None = None
    rating: SessionRatingOut | None = None
    has_answers: bool = False          # something to rate: the client asks for a rating when leaving
    pinned: bool = False
    created_at: datetime
    updated_at: datetime


class SourceOut(BaseModel):
    n: int | None = None           # the [n] marker in the answer; None on answers from before numbering
    title: str
    doc_type: str | None = None
    page: int
    section: str | None = None
    figure_id: UUID | None = None  # the page is a drawing: show it as it is (GET /api/figures/{id})


class SourceExcerptOut(BaseModel):
    """The exact passage an answer cited (D-A: technicians and admins may read the manuals). Served only for
    sources the answer actually cited, never for arbitrary chunks."""

    n: int
    title: str
    doc_type: str | None = None
    page: int
    section: str | None = None
    excerpts: list[str]
    figure_id: UUID | None = None


class QuestionOptionOut(BaseModel):
    label: str


class QuestionOut(BaseModel):
    question: str
    options: list[QuestionOptionOut]
    allow_free_text: bool = True


class AttachmentOut(BaseModel):
    id: UUID
    kind: str   # image | audio


class FigureRefOut(BaseModel):
    """A drawing shown with an answer (GET /api/figures/{figure_id}). `n` is set when the answer cited it."""

    figure_id: UUID
    title: str
    page: int
    n: int | None = None


class MessageOut(BaseModel):
    id: UUID
    role: str
    text: str
    sources: list[SourceOut] = []
    created_at: datetime
    attachments: list[AttachmentOut] = []
    figures: list[FigureRefOut] = []


class SessionDetailOut(SessionOut):
    client_name: str | None = None
    messages: list[MessageOut]
    pending_question: QuestionOut | None = None


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    option_index: int | None = Field(default=None, ge=0, le=20)
    # Any other field a caller sends (e.g. a foreign client_id / machine_line_id) is ignored by design.
