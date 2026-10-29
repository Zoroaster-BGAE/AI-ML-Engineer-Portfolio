"""The nightly job: conversation retention, trace purge, and the summary batches.

    docker-entrypoint.sh python -m app.cli.nightly [--skip-summaries]

One command so it needs one schedule (infra request #7). Each step runs even if an earlier one failed; the exit
code is non-zero if any failed, so the scheduled task shows red instead of quietly skipping work.
"""
from __future__ import annotations

import argparse
import sys

from app.cli.purge_traces import purge
from app.db import SessionLocal
from app.obs.logging import configure_logging, get_logger
from app.quality import retention, summaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-summaries", action="store_true", help="retention and trace purge only (no Claude calls)")
    a = ap.parse_args()
    configure_logging()
    log = get_logger()
    failed: list[str] = []

    def step(name, fn):
        try:
            with SessionLocal() as db:
                result = fn(db)
            log.info("nightly_step", step=name, result=result)
        except Exception as exc:  # noqa: BLE001 - one failed step must not skip the others
            failed.append(name)
            log.error("nightly_step_failed", step=name, error=str(exc)[:500])

    step("retention", retention.apply)
    step("purge_traces", lambda db: {"runs_deleted": purge(db)})
    if not a.skip_summaries:
        from app.llm.batches import AnthropicBatches

        def run_summaries(db):
            client = AnthropicBatches()
            return {"collected": summaries.collect(db, client), "submitted": summaries.submit(db, client)}

        step("summaries", run_summaries)

    if failed:
        sys.exit(f"nightly: failed steps: {', '.join(failed)}")


if __name__ == "__main__":
    main()
