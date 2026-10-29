"""Voice notes to text (D-K: AWS Transcribe, English and Greek only).

Batch transcription reading the clip already stored in S3: browsers record WebM/Opus, which batch accepts and
streaming does not, and the clip is stored anyway. A 30-second note typically comes back in 10-30 s. The
transcript is shown to the technician to confirm or correct before it is sent -- a misheard alarm code on a
safety-relevant machine is worse than one extra tap.

Locally the emulator has no Transcribe, so `TRANSCRIBER=fake` returns a fixed sentence.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol
from uuid import uuid4

from app.config import settings

AUDIO_TYPES = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/x-m4a": "m4a",
               "audio/m4a": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav"}
MAX_AUDIO_BYTES = 10 * 1024 * 1024
LANGUAGES = {"el": "el-GR", "en": "en-US"}


class TranscriptionFailed(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language: str


class Transcriber(Protocol):
    def transcribe(self, s3_uri: str, media_format: str, language_hint: str | None) -> Transcript: ...


class FakeTranscriber:
    TEXT = "Alarm E4 on the drum motor."

    def transcribe(self, s3_uri, media_format, language_hint):
        return Transcript(self.TEXT, language_hint or "en")


class AwsTranscriber:
    def __init__(self, timeout_s: float = 90.0):
        import boto3

        self._c = boto3.client("transcribe", region_name=settings.cognito_region)
        self.timeout_s = timeout_s

    def transcribe(self, s3_uri, media_format, language_hint):
        job = f"assistant-{uuid4().hex}"
        kw = {"LanguageCode": LANGUAGES[language_hint]} if language_hint in LANGUAGES else {
            "IdentifyLanguage": True, "LanguageOptions": list(LANGUAGES.values())}
        self._c.start_transcription_job(TranscriptionJobName=job, Media={"MediaFileUri": s3_uri},
                                        MediaFormat=media_format, **kw)
        deadline = time.monotonic() + self.timeout_s
        try:
            while time.monotonic() < deadline:
                j = self._c.get_transcription_job(TranscriptionJobName=job)["TranscriptionJob"]
                status = j["TranscriptionJobStatus"]
                if status == "COMPLETED":
                    with urllib.request.urlopen(j["Transcript"]["TranscriptFileUri"], timeout=30) as r:  # noqa: S310
                        data = json.load(r)
                    text = " ".join(t["transcript"] for t in data["results"]["transcripts"]).strip()
                    code = j.get("LanguageCode") or LANGUAGES.get(language_hint or "", "en-US")
                    return Transcript(text, code.split("-")[0])
                if status == "FAILED":
                    raise TranscriptionFailed(j.get("FailureReason", "transcription failed"))
                time.sleep(1.5)
            raise TranscriptionFailed("transcription timed out")
        finally:
            try:
                self._c.delete_transcription_job(TranscriptionJobName=job)  # the text is ours now; keep nothing there
            except Exception:  # noqa: BLE001, S110 - best effort cleanup
                pass


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    return FakeTranscriber() if settings.transcriber == "fake" else AwsTranscriber()
