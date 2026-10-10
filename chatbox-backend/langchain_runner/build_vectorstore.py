import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from template_store import init_db, reindex_templates, seed_templates_from_files
from tenancy import tenant_context, ensure_workspace

load_dotenv()


def build_if_missing(force: bool = False) -> None:
    """Initialize and refresh the Postgres/pgvector template store.

    Templates and embeddings live in Postgres.
    This seeds bundled JSON templates and refreshes embedding fields.
    """
    init_db()
    seeded = seed_templates_from_files(embed=True, overwrite=force)
    if force or seeded:
        indexed = reindex_templates()
        print(f"Postgres template store ready: seeded={seeded}, indexed={indexed}")
    else:
        print("Postgres template store already initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True, help="Supabase Auth user UUID")
    args = parser.parse_args()
    with tenant_context(args.tenant):
        ensure_workspace()
        build_if_missing(force=True)
