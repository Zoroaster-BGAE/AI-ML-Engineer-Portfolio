import argparse
from pathlib import Path

from app.db import SessionLocal
from app.ingest.onboard import apply_plan, build_plan, render_plan
from app.llm.embed import get_embedder


def main() -> None:
    ap = argparse.ArgumentParser(description="Onboard a client folder into the library.")
    ap.add_argument("client_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the tree-to-rows plan and exit")
    ap.add_argument("--only", help="only machine folders whose name contains this text")
    ap.add_argument("--slug"), ap.add_argument("--name")
    ap.add_argument("--reindex", action="store_true", help="re-parse documents that are already indexed")
    a = ap.parse_args()
    if not a.client_dir.is_dir():
        raise SystemExit(f"not a folder: {a.client_dir}")
    plan = build_plan(a.client_dir, a.slug, a.name, a.only)
    print(render_plan(plan))
    if a.dry_run:
        return
    with SessionLocal() as db:
        print("stats:", apply_plan(db, plan, get_embedder(), reindex=a.reindex))


if __name__ == "__main__":
    main()
