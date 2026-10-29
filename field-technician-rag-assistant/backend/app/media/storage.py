"""Where media bytes live. S3 in the deployment (the documents bucket: `events/` for what technicians send,
`derived/` for rendered drawing pages), the moto emulator locally, a directory in tests.

Keys are always generated here from ids, never taken from a request, so a key cannot point outside its prefix.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import settings


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def uri(self, key: str) -> str: ...


class S3Storage:
    def __init__(self, bucket: str):
        import boto3  # the API image has it; imported late so tests without AWS never touch it

        self.bucket = bucket
        self._s3 = boto3.client("s3")

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                            ServerSideEncryption="AES256")

    def get(self, key: str) -> bytes:
        return self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"key escapes the storage root: {key}")
        return p

    def put(self, key: str, data: bytes, content_type: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def uri(self, key: str) -> str:
        return str(self._path(key))


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    if settings.media_store == "local":
        return LocalStorage(settings.media_local_dir)
    return S3Storage(settings.media_bucket)
