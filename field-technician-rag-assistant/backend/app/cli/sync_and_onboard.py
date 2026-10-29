"""Mirror the manuals bucket to local disk, then onboard every client folder in it.

    python -m app.cli.sync_and_onboard s3://example-documents-bucket/manuals [--dry-run] [--only SUBSTR]

This is what runs on a schedule in the deployment. `app.cli.onboard` stays the interactive,
one-client-at-a-time entry point; this wraps it for unattended use.

Why sync to disk instead of reading S3 directly: `build_plan` walks a directory
tree -- iterdir, rglob, is_dir, open -- and the folder layout *is* the schema
(`<client>/<machine>/[<component>/]<file>`, PRD section 5). Reimplementing that
against object keys would mean reimplementing the part most likely to be subtly
wrong, for no benefit. A scheduled task's ephemeral storage is far larger than a
typical corpus.

Safe to run repeatedly. Documents are identified by (machine_line, sha256), so a
daily run over an unchanged corpus reaches the database and changes nothing. That
is what makes scheduling this sane rather than alarming.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from app.db import SessionLocal
from app.ingest.onboard import apply_plan, build_plan, render_plan
from app.llm.embed import get_embedder
from app.obs.logging import get_logger

log = get_logger()


def sync_prefix(bucket: str, prefix: str, dest: Path) -> int:
    """Download every object under `prefix` into `dest`, preserving the key layout.

    Skips a file already present at the same byte size. The ingest step is
    idempotent anyway, so this is a bandwidth optimisation rather than a
    correctness one -- but on a fresh task's empty disk it does nothing, which is
    the normal case.
    """
    import boto3  # imported here so the API container never pays for it at startup

    s3 = boto3.client("s3")
    prefix = prefix.strip("/")
    downloaded = 0

    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Directory placeholder objects: zero bytes, key ends in a slash.
            if key.endswith("/"):
                continue
            # The sync module keeps its bookkeeping alongside the manuals; it is
            # not a document and would be planned as an unassigned file.
            if "/_sync/" in f"/{key}" or key.startswith("_sync/"):
                continue

            relative = key[len(prefix):].lstrip("/")
            target = dest / relative
            if target.exists() and target.stat().st_size == obj["Size"]:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target))
            downloaded += 1

    return downloaded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="s3://bucket/prefix holding <client>/<machine>/... folders")
    ap.add_argument("--dry-run", action="store_true", help="print each client's plan and exit")
    ap.add_argument("--only", help="only machine folders whose name contains this text")
    ap.add_argument("--work-dir", type=Path, help="where to sync (default: a temp dir)")
    ap.add_argument("--reindex", action="store_true",
                    help="re-parse documents that are already indexed (once, after an ingest upgrade)")
    a = ap.parse_args()

    parsed = urlparse(a.source)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise SystemExit(f"source must look like s3://bucket/prefix, got: {a.source}")

    work = a.work_dir or Path(tempfile.mkdtemp(prefix="corpus-"))
    work.mkdir(parents=True, exist_ok=True)

    log.info("sync_start", bucket=parsed.netloc, prefix=parsed.path, dest=str(work))
    downloaded = sync_prefix(parsed.netloc, parsed.path, work)
    log.info("sync_done", downloaded=downloaded)

    clients = sorted(p for p in work.iterdir() if p.is_dir())
    if not clients:
        # Loud, not silent. An empty corpus almost always means the prefix is
        # wrong, and a run that "succeeded" having ingested nothing is the most
        # expensive kind of failure to notice.
        raise SystemExit(f"no client folders under {a.source} -- nothing to onboard")

    log.info("clients_found", count=len(clients), names=[c.name for c in clients])

    failures = 0
    for client_dir in clients:
        try:
            plan = build_plan(client_dir, None, None, a.only)
            print(render_plan(plan))
            if a.dry_run:
                continue
            with SessionLocal() as db:
                stats = apply_plan(db, plan, get_embedder(), reindex=a.reindex)
            log.info("client_onboarded", client=client_dir.name, stats=str(stats))

            # apply_plan catches per-document errors, records them on the row and
            # returns normally -- so without this a run that lost 23 of 45
            # documents exits 0 and looks clean. That is exactly what happened on
            # the first full ingest: a parser worker was OOM-killed, every
            # pending future failed with BrokenProcessPool, and the task went
            # green with half the library missing.
            if stats.get("failed"):
                failures += 1
                log.error("documents_failed", client=client_dir.name, count=stats["failed"])
        except Exception as exc:  # noqa: BLE001 - one bad client should not stop the rest
            failures += 1
            log.error("client_failed", client=client_dir.name, error=str(exc)[:500])

    if failures:
        # Non-zero exit so the scheduled task shows as failed rather than
        # quietly leaving part of the library unsearchable. Failed documents are
        # retried on the next run -- apply_plan re-queues anything not already
        # 'indexed' -- so this is a signal to look, not a reason to intervene.
        sys.exit(
            f"{failures} of {len(clients)} clients had failures -- see the log above. "
            "Query: select title, ingest_note from document where ingest_status='failed';"
        )


if __name__ == "__main__":
    main()
