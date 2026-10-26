"""Apply supabase/migrations inside the caller's transaction so test fixtures roll back."""
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
RUNTIME_ROLE = "heywrite_app"


def apply_migrations(conn) -> None:
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        conn.execute(path.read_text())
