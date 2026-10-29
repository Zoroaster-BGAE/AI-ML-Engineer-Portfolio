"""Feature 4: drawing and scanned pages are rendered at ingest and shown as they are."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy import select

from app.agent.schemas import Grade, Slots, Understanding
from app.domain.models import Chunk, Document, Figure, MachineLine
from app.ingest import parse
from app.ingest.onboard import apply_plan, build_plan
from tests.conftest import EMB, login
from tests.integration.test_api import _post

pytestmark = pytest.mark.db


def scanned_pdf(path, pages=2):
    """A PDF whose pages are nothing but an image, like a scanned wiring set."""
    sheets = [Image.new("RGB", (1200, 850), (255, 255, 255)) for _ in range(pages)]
    for i, s in enumerate(sheets):
        for x in range(100, 1100, 25):
            for y in range(100 + i * 10, 750):
                s.putpixel((x, y), (0, 0, 0))
    sheets[0].save(path, "PDF", save_all=True, append_images=sheets[1:], resolution=100)


def test_dense_vector_pages_and_scanned_pages_are_figures_and_text_pages_are_not():
    drawing = SimpleNamespace(lines=[0] * 250, rects=[0] * 60, images=[], width=842, height=595)
    scanned = SimpleNamespace(lines=[], rects=[], width=842, height=595,
                              images=[{"x0": 0, "x1": 842, "top": 0, "bottom": 595}])
    text_page = SimpleNamespace(lines=[0] * 10, rects=[], images=[], width=595, height=842)
    assert parse._figure_kind(drawing, "K1 K2 X1:3") == "drawing"
    assert parse._figure_kind(scanned, "") == "scanned"
    assert parse._figure_kind(text_page, "Isolate the supply before opening the cabinet.") is None


@pytest.fixture
def scanned_library(db, tmp_path):
    root = tmp_path / "Client Scan"
    (root / "DRYER DT100").mkdir(parents=True)
    scanned_pdf(root / "DRYER DT100" / "WIRINGS DT100_EN.pdf")
    plan = build_plan(root)
    stats = apply_plan(db, plan, EMB, log=lambda *_: None, workers=1)
    db.commit()
    return stats


def test_a_scanned_wiring_set_is_indexed_with_its_sheets_instead_of_failing(db, scanned_library):
    assert scanned_library["indexed"] == 1 and scanned_library["failed"] == 0
    doc = db.scalar(select(Document))
    figs = db.scalars(select(Figure).order_by(Figure.page)).all()
    assert [f.page for f in figs] == [1, 2] and all(f.kind == "scanned" for f in figs)
    assert all(f.storage_key == f"derived/{doc.id}/p{f.page}.png" for f in figs)
    chunks = db.scalars(select(Chunk).order_by(Chunk.page)).all()
    assert [c.content_type for c in chunks] == ["drawing", "drawing"] and "WIRINGS DT100_EN" in chunks[0].content


def test_reindex_renders_figures_for_documents_that_were_already_indexed(db, scanned_library, tmp_path):
    db.query(Figure).delete()
    db.commit()
    stats = apply_plan(db, build_plan(tmp_path / "Client Scan"), EMB, log=lambda *_: None, workers=1)
    assert stats["already"] == 1 and db.scalar(select(Figure)) is None          # plain run: untouched
    stats = apply_plan(db, build_plan(tmp_path / "Client Scan"), EMB, log=lambda *_: None, workers=1, reindex=True)
    db.commit()
    assert stats["indexed"] == 1 and len(db.scalars(select(Figure)).all()) == 2


def test_an_answer_citing_a_drawing_page_shows_the_sheet(api, db, llm, scanned_library):
    ml = db.scalar(select(MachineLine))
    h = login(api, "tech@t")
    sid = api.post("/api/sessions", headers=h, json={"client_id": str(ml.client_id),
                                                    "machine_line_id": str(ml.id)}).json()["id"]
    llm.script["understand"] = [Understanding(intent="information", slots=Slots(error_codes=[], component_hint="",
                                                                                 symptom=""),
                                              wanted_doc_types=["wiring_diagram"], plan=[], language="en",
                                              standalone_query="WIRINGS DT100 wiring diagram drawing sheet",
                                              rationale="r")]
    llm.script["assess_grade"] = [Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="ok")]
    llm.text = "The wiring is on the first sheet [1]."
    done = json.loads(_post(api, h, sid, "Show me the wiring diagram")[-1]["data"])
    fig_id = done["sources"][0]["figure_id"]
    assert fig_id
    r = api.get(f"/api/figures/{fig_id}", headers=h)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.width > 500                                                    # the sheet, not a thumbnail
    excerpt = api.get(f"/api/messages/{done['message_id']}/sources/1", headers=h).json()
    assert excerpt["figure_id"] == fig_id
    assert done["figures"][0] == {"figure_id": fig_id, "title": "WIRINGS DT100_EN", "page": 1, "n": 1}
    detail = api.get(f"/api/sessions/{sid}", headers=h).json()
    assert detail["messages"][-1]["figures"][0]["figure_id"] == fig_id


def test_a_retrieved_drawing_is_shown_even_when_the_answer_does_not_cite_it(api, db, llm, scanned_library):
    ml = db.scalar(select(MachineLine))
    h = login(api, "tech@t")
    sid = api.post("/api/sessions", headers=h, json={"client_id": str(ml.client_id),
                                                    "machine_line_id": str(ml.id)}).json()["id"]
    llm.script["understand"] = [Understanding(intent="information", slots=Slots(error_codes=[], component_hint="",
                                                                                 symptom=""),
                                              wanted_doc_types=["wiring_diagram"], plan=[], language="en",
                                              standalone_query="WIRINGS DT100 drawing sheet", rationale="r")]
    llm.script["assess_grade"] = [Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="ok")]
    llm.text = "The sheets are in the wiring set."                                  # no [n] at all
    events = _post(api, h, sid, "wiring for the dryer")
    done = json.loads(events[-1]["data"])
    answer = "".join(json.loads(e["data"]) for e in events if e["event"] == "token")
    assert "[" not in answer                                                         # nothing cited in the text
    assert sorted(f["page"] for f in done["figures"]) == [1, 2]                      # both sheets still shown


def test_customers_cannot_open_drawings(api, db, scanned_library):
    import tests.conftest as c
    from app.domain.support import AppUser

    fig = db.scalar(select(Figure))
    db.add(AppUser(email="cust@t", role="customer"))
    db.commit()
    c._TEST_GROUPS["cust@t"] = ["customer"]
    try:
        assert api.get(f"/api/figures/{fig.id}", headers={"Authorization": "Bearer test.cust@t"}).status_code == 403
    finally:
        c._TEST_GROUPS.pop("cust@t")
