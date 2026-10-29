"""Feature 2: photos and voice notes with a message."""
import io
import json

import pytest
from PIL import Image
from sqlalchemy import select

from app.agent.schemas import Grade, Slots, Understanding
from app.domain.support import Attachment, ChatMessage
from app.media import images as img
from tests.conftest import login
from tests.integration.test_api import _create_session

pytestmark = pytest.mark.db

WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 64


def png(w=64, h=32, exif=False) -> bytes:
    out = io.BytesIO()
    im = Image.new("RGB", (w, h), (200, 30, 30))
    if exif:
        ex = Image.Exif()
        ex[0x8825] = {2: (35.0, 20.0, 1.0)}  # GPSInfo
        im.save(out, format="JPEG", exif=ex)
    else:
        im.save(out, format="PNG")
    return out.getvalue()


def _script(llm):
    llm.script["understand"] = [Understanding(intent="troubleshoot", slots=Slots(error_codes=["E4"], component_hint="",
                                                                                 symptom=""), wanted_doc_types=[],
                                              standalone_query="alarm E4 drum motor", language="en", plan=[],
                                              rationale="read E4 off the photo")]
    llm.script["assess_grade"] = [Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="ok")]
    llm.text = "Reset the thermal switch after isolating power [1]."


def _events(r) -> list[dict]:
    out, cur = [], {}
    for line in r.iter_lines():
        if line.startswith("event:"):
            cur = {"event": line.split(":", 1)[1].strip()}
        elif line.startswith("data:"):
            cur["data"] = json.loads(line.split(":", 1)[1].strip())
            out.append(cur)
    return out


def test_a_photo_reaches_the_model_and_is_stored_re_encoded(api, world, db, llm):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    _script(llm)
    with api.stream("POST", f"/api/sessions/{sid}/messages", headers=h,
                    data={"text": "what does this alarm mean"},
                    files=[("images", ("panel.png", png(), "image/png"))]) as r:
        assert r.status_code == 200
        ev = _events(r)
    assert ev[-1]["data"]["action"] == "ANSWER"
    assert [c["images"] for c in llm.calls if c["node"] in ("understand", "synthesize")] == [1, 1]
    a = db.scalar(select(Attachment))
    assert a.kind == "image" and a.content_type == "image/jpeg" and a.storage_key.startswith(f"events/{sid}/")
    got = api.get(f"/api/attachments/{a.id}", headers=h)
    assert got.status_code == 200 and got.content[:2] == b"\xff\xd8"
    assert api.get(f"/api/attachments/{a.id}", headers=login(api, "admin@t")).status_code == 200  # reviewer view
    detail = api.get(f"/api/sessions/{sid}", headers=h).json()
    assert detail["messages"][0]["attachments"] == [{"id": str(a.id), "kind": "image"}]


def test_a_photo_without_words_is_accepted(api, world, db, llm):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    _script(llm)
    with api.stream("POST", f"/api/sessions/{sid}/messages", headers=h,
                    files=[("images", ("panel.png", png(), "image/png"))]) as r:
        assert r.status_code == 200
        _events(r)
    assert db.scalar(select(ChatMessage).where(ChatMessage.role == "user")).content == ""
    assert "photo" in next(c for c in llm.calls if c["node"] == "understand")["user"]


def test_not_an_image_is_refused_and_nothing_is_stored(api, world, db):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    r = api.post(f"/api/sessions/{sid}/messages", headers=h, data={"text": "x"},
                 files=[("images", ("evil.png", b"<script>alert(1)</script>", "image/png"))])
    assert r.status_code == 415 and db.scalar(select(Attachment)) is None


def test_too_many_photos_are_refused(api, world):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    r = api.post(f"/api/sessions/{sid}/messages", headers=h, data={"text": "x"},
                 files=[("images", (f"{i}.png", png(), "image/png")) for i in range(5)])
    assert r.status_code == 422


def test_exif_and_gps_are_stripped_and_size_is_bounded():
    c = img.clean(png(4000, 1000, exif=True))
    with Image.open(io.BytesIO(c.data)) as im:
        assert not im.getexif() and max(im.size) == img.MAX_EDGE


def test_a_voice_note_is_transcribed_confirmed_and_linked_once(api, world, db, llm):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    r = api.post(f"/api/sessions/{sid}/audio", headers=h, files={"file": ("note.webm", WEBM, "audio/webm")},
                 data={"language": "en"})
    assert r.status_code == 200 and r.json()["transcript"] == "Alarm E4 on the drum motor."
    audio_id = r.json()["audio_id"]
    _script(llm)
    with api.stream("POST", f"/api/sessions/{sid}/messages", headers=h,
                    data={"text": "Alarm E4 on the drum motor, corrected", "audio_id": audio_id}) as resp:
        assert resp.status_code == 200
        _events(resp)
    a = db.get(Attachment, audio_id)
    msg = db.get(ChatMessage, a.message_id)
    assert a.stt_text == "Alarm E4 on the drum motor." and msg.content == "Alarm E4 on the drum motor, corrected"
    again = api.post(f"/api/sessions/{sid}/messages", headers=h, data={"text": "again", "audio_id": audio_id})
    assert again.status_code == 422


def test_audio_must_really_be_audio(api, world):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    r = api.post(f"/api/sessions/{sid}/audio", headers=h, files={"file": ("x.webm", b"not audio", "audio/webm")})
    assert r.status_code == 415


def test_someone_elses_photo_is_not_found(api, world, db, llm):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    _script(llm)
    with api.stream("POST", f"/api/sessions/{sid}/messages", headers=h, data={"text": "look"},
                    files=[("images", ("p.png", png(), "image/png"))]) as r:
        _events(r)
    a = db.scalar(select(Attachment))
    from app.domain.support import AppUser
    other = AppUser(email="other@t", role="technician")
    db.add(other)
    db.commit()
    import tests.conftest as c
    c._TEST_GROUPS["other@t"] = ["technician"]
    try:
        assert api.get(f"/api/attachments/{a.id}", headers={"Authorization": "Bearer test.other@t"}).status_code == 404
    finally:
        del c._TEST_GROUPS["other@t"]
