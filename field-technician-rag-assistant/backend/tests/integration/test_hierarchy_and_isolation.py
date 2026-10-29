"""Hierarchy (composite FKs, constraints) and scope isolation, against real Postgres + pgvector."""
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.domain.models import Chunk, Client, Document, MachineLine
from app.domain.scope import Scope
from app.domain.support import AppUser, ChatSession
from app.retrieval.hybrid import hybrid_search
from app.retrieval.repository import ScopedRepository
from tests.conftest import EMB, add_doc

pytestmark = pytest.mark.db


# ---- hierarchy -----------------------------------------------------------------------------------------
def test_document_cannot_claim_a_client_its_machine_does_not_belong_to(db, world):
    bad = Document(client_id=world["c2"].id, machine_line_id=world["a"].id, title="x", doc_type="other",
                   media_type="application/pdf", storage_uri="/x", sha256="deadbeef")
    db.add(bad)
    with pytest.raises(IntegrityError):
        db.flush()


def test_chunk_cannot_sit_under_the_wrong_machine(db, world):
    db.add(Chunk(document_id=world["doc_a"].id, machine_line_id=world["b"].id, client_id=world["c1"].id, page=1,
                 content="misfiled"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_machine_slug_is_unique_per_client_but_reusable_across_clients(db, world):
    db.add(MachineLine(client_id=world["c1"].id, slug="a", label="dup", kind="machine"))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()
    db.add(MachineLine(client_id=world["c2"].id, slug="a", label="ok on another client", kind="machine"))
    db.flush()


def test_doc_type_and_kind_are_constrained(db, world):
    db.add(MachineLine(client_id=world["c1"].id, slug="k", label="k", kind="spaceship"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_same_file_cannot_be_registered_twice_on_one_machine(db, world):
    d = world["doc_a"]
    db.add(Document(client_id=d.client_id, machine_line_id=d.machine_line_id, title="again", doc_type="other",
                    media_type="application/pdf", storage_uri="/y", sha256=d.sha256))
    with pytest.raises(IntegrityError):
        db.flush()


def test_chat_session_scope_is_immutable_in_the_database(db, world):
    u = AppUser(email="u@t", password_hash="x", role="technician")
    db.add(u)
    db.flush()
    s = ChatSession(user_id=u.id, client_id=world["c1"].id, machine_line_id=world["a"].id)
    db.add(s)
    db.commit()
    with pytest.raises(DBAPIError, match="immutable"):
        db.execute(text("UPDATE chat_session SET machine_line_id = :m WHERE id = :i"),
                   {"m": world["b"].id, "i": s.id})


def test_generated_tsvector_and_pgvector_are_populated(db, world):
    row = db.execute(text("select tsv is not null, embedding is not null from chunk limit 1")).one()
    assert tuple(row) == (True, True)


# ---- scope isolation -----------------------------------------------------------------------------------
def _scope(m):
    return Scope(m.client_id, m.id)


def test_repository_refuses_to_run_without_a_scope(db):
    for bad in (None, {"machine_line_id": "x"}, "machine"):
        with pytest.raises(TypeError):
            ScopedRepository(db, bad)


def test_identical_text_in_machines_a_and_b_never_crosses_over(db, world):
    """The load-bearing test: same words, same embeddings, different machines."""
    repo = ScopedRepository(db, _scope(world["a"]))
    q = "Alarm E4 drum motor thermal switch"
    hits = hybrid_search(repo, EMB, q)
    assert hits, "machine A should find its own chunk"
    assert {h.document_id for h in hits} == {world["doc_a"].id}
    assert not {h.document_id for h in hybrid_search(repo, EMB, q)} & {world["doc_b"].id, world["doc_x"].id}


def test_each_retriever_is_scoped_on_its_own(db, world):
    repo = ScopedRepository(db, _scope(world["b"]))
    v = repo.vector_search(EMB.embed(["drum motor thermal switch"])[0], 20)
    f = repo.fts_search("drum motor thermal switch", 20)
    assert v and f
    assert {h.document_id for h in v + f} == {world["doc_b"].id}


def test_a_foreign_document_id_in_a_plan_returns_nothing(db, world):
    repo = ScopedRepository(db, _scope(world["a"]))
    assert repo.filter_doc_ids([world["doc_b"].id, uuid4(), "not-a-uuid"]) == set()
    hits = repo.vector_search(EMB.embed(["drum motor"])[0], 20, doc_ids=[world["doc_b"].id])
    assert hits == []  # doc_ids narrows within the scope; it can never widen it
    assert repo.read_pages(world["doc_b"].id, 1, 3) == []


def test_a_machine_id_from_another_client_is_not_a_valid_scope_pair(db, world):
    repo = ScopedRepository(db, Scope(world["c2"].id, world["a"].id))  # A belongs to client 1, not 2
    assert repo.documents() == []
    assert hybrid_search(repo, EMB, "drum motor") == []


def test_read_pages_is_scoped_and_ordered(db, world):
    repo = ScopedRepository(db, _scope(world["a"]))
    pages = [h.page for h in repo.read_pages(world["doc_a"].id, 1, 2)]
    assert pages == [1, 2]


def test_unindexed_documents_are_invisible_to_retrieval(db, world):
    add_doc(db, world["a"], "Pending Manual", ["secret unindexed text about widgets"])
    db.execute(text("update document set ingest_status='pending' where title='Pending Manual'"))
    db.commit()
    repo = ScopedRepository(db, _scope(world["a"]))
    assert not any("widgets" in h.content for h in hybrid_search(repo, EMB, "widgets"))
    assert "Pending Manual" not in [d.title for d in repo.documents()]


def test_client_row_is_not_needed_for_isolation_tests_to_be_meaningful(db, world):
    assert db.query(Client).count() == 2 and db.query(Chunk).count() >= 5
