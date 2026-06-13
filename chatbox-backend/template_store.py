import json
import os
import re
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_ROOT / ".env", override=True)

TEMPLATE_DIR = Path(os.getenv("TEMPLATE_DIR", "./templates"))
EMBEDDING_MODEL = os.getenv("TEMPLATE_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("TEMPLATE_EMBEDDING_DIMENSION", "1536"))

embedding_client = None


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for Postgres template storage")
    return database_url


def get_connection():
    return psycopg.connect(get_database_url(), row_factory=dict_row)


def get_embedding_client():
    global embedding_client
    if embedding_client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required to embed templates")
        embedding_client = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=openai_api_key,
        )
    return embedding_client


def embed_text(text: str) -> list[float]:
    return get_embedding_client().embed_query(text)


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "template"


def template_id_from_filename(path: Path) -> str:
    return path.stem


def is_template_file(path: Path) -> bool:
    return path.suffix == ".json" and not path.name.endswith(".embedding.json")


def infer_structure(content: str) -> list[str]:
    sections: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip().strip("#").strip("*").strip()
        if not line:
            continue
        if line.endswith(":") and len(line) <= 80:
            sections.append(line.rstrip(":").strip())
            continue
        numbered = re.match(r"^\d+\.\s+(.+)$", line)
        if numbered and len(numbered.group(1)) <= 80:
            sections.append(numbered.group(1).strip())
    seen = set()
    unique = []
    for section in sections:
        key = section.lower()
        if key not in seen:
            unique.append(section)
            seen.add(key)
    return unique[:12]


def normalize_template(data: dict[str, Any], template_id: str | None = None) -> dict[str, Any]:
    title = str(data.get("title") or template_id or "Untitled Template").strip()
    content = str(data.get("content") or "").strip()
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    if not isinstance(tags, list):
        tags = []

    category = str(data.get("category") or (tags[0] if tags else "general")).strip().lower()
    description = str(data.get("description") or "").strip()
    if not description:
        description = f"Template for {title.lower()}."

    structure = data.get("structure")
    if not isinstance(structure, list):
        structure = infer_structure(content)

    return {
        "id": str(data.get("id") or template_id or slugify(title)),
        "title": title,
        "category": category,
        "description": description,
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        "style": str(data.get("style") or "Formal"),
        "language": str(data.get("language") or "English"),
        "use_cases": data.get("use_cases") if isinstance(data.get("use_cases"), list) else [],
        "structure": [str(section).strip() for section in structure if str(section).strip()],
        "content": content,
        "enabled": bool(data.get("enabled", True)),
        "version": int(data.get("version") or 1),
    }


def build_embedding_text(template: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Title: {template.get('title', '')}",
            f"Category: {template.get('category', '')}",
            f"Description: {template.get('description', '')}",
            "Tags: " + ", ".join(template.get("tags", [])),
            "Use cases:",
            *[f"- {case}" for case in template.get("use_cases", [])],
        ]
    )


def init_db() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS templates (
                    id text PRIMARY KEY,
                    title text NOT NULL,
                    category text NOT NULL DEFAULT 'general',
                    description text NOT NULL DEFAULT '',
                    tags text[] NOT NULL DEFAULT '{{}}',
                    style text NOT NULL DEFAULT 'Formal',
                    language text NOT NULL DEFAULT 'English',
                    use_cases jsonb NOT NULL DEFAULT '[]'::jsonb,
                    structure jsonb NOT NULL DEFAULT '[]'::jsonb,
                    content text NOT NULL DEFAULT '',
                    enabled boolean NOT NULL DEFAULT true,
                    version integer NOT NULL DEFAULT 1,
                    embedding vector({EMBEDDING_DIMENSION}),
                    embedding_text text,
                    embedding_model text,
                    embedding_updated_at timestamptz,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS templates_enabled_idx ON templates (enabled)")
            cur.execute("CREATE INDEX IF NOT EXISTS templates_category_idx ON templates (category)")
            cur.execute("CREATE INDEX IF NOT EXISTS templates_tags_idx ON templates USING gin (tags)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS templates_embedding_idx
                ON templates USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )


def row_to_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "description": row["description"],
        "tags": row.get("tags") or [],
        "style": row.get("style") or "Formal",
        "language": row.get("language") or "English",
        "use_cases": row.get("use_cases") or [],
        "structure": row.get("structure") or [],
        "content": row.get("content") or "",
        "enabled": bool(row.get("enabled", True)),
        "version": int(row.get("version") or 1),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }


def list_templates() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, category, description, tags, style, language,
                   use_cases, structure, content, enabled, version, created_at, updated_at
            FROM templates
            ORDER BY title
            """
        ).fetchall()
    return [row_to_template(row) for row in rows]


def get_template(template_id: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, title, category, description, tags, style, language,
                   use_cases, structure, content, enabled, version, created_at, updated_at
            FROM templates
            WHERE id = %s
            """,
            (template_id,),
        ).fetchone()
    return row_to_template(row) if row else None


def save_template(template: dict[str, Any], embed: bool = True) -> dict[str, Any]:
    init_db()
    normalized = normalize_template(template)
    normalized["id"] = slugify(normalized["id"] or normalized["title"])

    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO templates (
                id, title, category, description, tags, style, language,
                use_cases, structure, content, enabled, version
            )
            VALUES (
                %(id)s, %(title)s, %(category)s, %(description)s, %(tags)s,
                %(style)s, %(language)s, %(use_cases)s, %(structure)s,
                %(content)s, %(enabled)s, %(version)s
            )
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                description = EXCLUDED.description,
                tags = EXCLUDED.tags,
                style = EXCLUDED.style,
                language = EXCLUDED.language,
                use_cases = EXCLUDED.use_cases,
                structure = EXCLUDED.structure,
                content = EXCLUDED.content,
                enabled = EXCLUDED.enabled,
                version = templates.version + 1,
                updated_at = now()
            RETURNING id, title, category, description, tags, style, language,
                      use_cases, structure, content, enabled, version, created_at, updated_at
            """,
            {
                **normalized,
                "use_cases": Jsonb(normalized["use_cases"]),
                "structure": Jsonb(normalized["structure"]),
            },
        ).fetchone()

    saved = row_to_template(row)
    if embed:
        upsert_template_embedding(saved)
        refreshed = get_template(saved["id"])
        return refreshed or saved
    return saved


def upsert_template_embedding(template: dict[str, Any]) -> None:
    embedding_text = build_embedding_text(template)
    vector = embed_text(embedding_text)
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE templates
            SET embedding = %s::vector,
                embedding_text = %s,
                embedding_model = %s,
                embedding_updated_at = now(),
                updated_at = now()
            WHERE id = %s
            """,
            (vector_literal(vector), embedding_text, EMBEDDING_MODEL, template["id"]),
        )


def search_templates(query: str, limit: int = 3) -> list[tuple[dict[str, Any], float]]:
    init_db()
    query_vector = vector_literal(embed_text(query))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, category, description, tags, style, language,
                   use_cases, structure, content, enabled, version, created_at, updated_at,
                   embedding <=> %s::vector AS distance
            FROM templates
            WHERE enabled = true AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, limit),
        ).fetchall()
    return [(row_to_template(row), float(row["distance"])) for row in rows]


def read_template_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return normalize_template(data, template_id_from_filename(path))


def seed_templates_from_files(embed: bool = True, overwrite: bool = False) -> int:
    init_db()
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for path in sorted(TEMPLATE_DIR.iterdir()):
        if not path.is_file() or not is_template_file(path):
            continue
        template = read_template_file(path)
        if not overwrite and get_template(template["id"]):
            continue
        save_template(template, embed=embed)
        seeded += 1
    return seeded


def reindex_templates() -> int:
    templates = list_templates()
    count = 0
    for template in templates:
        if template.get("enabled", True):
            upsert_template_embedding(template)
            count += 1
    return count
