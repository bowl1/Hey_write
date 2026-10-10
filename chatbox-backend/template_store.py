import hashlib
import json
import math
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from tenancy import tenant_id, tenant_schema
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
VECTOR_WEIGHT = float(os.getenv("TEMPLATE_VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("TEMPLATE_BM25_WEIGHT", "0.3"))
TEMPLATE_CHOICE_SCORE_GAP = float(os.getenv("TEMPLATE_CHOICE_SCORE_GAP", "0.05"))
MIN_MATCH_SCORE = float(os.getenv("TEMPLATE_MIN_MATCH_SCORE", "0.65"))
RETRIEVAL_CANDIDATE_LIMIT = int(os.getenv("TEMPLATE_RETRIEVAL_CANDIDATE_LIMIT", "20"))

embedding_client = None


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for Postgres template storage")
    return database_url


def get_connection():
    schema = tenant_schema(tenant_id())
    conn = psycopg.connect(get_database_url(), row_factory=dict_row)
    try:
        conn.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(sql.Identifier(schema)))
        return conn
    except Exception:
        conn.close()
        raise


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
    return path.suffix == ".json"


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
            "Structure:",
            *[f"- {section}" for section in template.get("structure", [])],
            "Content:",
            template.get("content", ""),
        ]
    )


def build_retrieval_text(template: dict[str, Any]) -> str:
    weighted_parts = [
        template.get("title", ""),
        template.get("title", ""),
        template.get("category", ""),
        template.get("description", ""),
        " ".join(template.get("tags", [])),
        " ".join(template.get("tags", [])),
        " ".join(template.get("use_cases", [])),
        " ".join(template.get("structure", [])),
        template.get("content", ""),
    ]
    return " ".join(part for part in weighted_parts if part)


def tokenize(text: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        terms.extend(segment[index:index + 2] for index in range(len(segment) - 1))
        if len(segment) == 1:
            terms.append(segment)
    return terms


def bm25_scores(query: str, templates: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[str]]]:
    query_terms = tokenize(query)
    if not query_terms or not templates:
        return {}, {}

    docs = {template["id"]: tokenize(build_retrieval_text(template)) for template in templates}
    avgdl = sum(len(tokens) for tokens in docs.values()) / max(len(docs), 1)
    document_frequency: dict[str, int] = {}
    for term in set(query_terms):
        document_frequency[term] = sum(1 for tokens in docs.values() if term in tokens)

    scores: dict[str, float] = {}
    matched_terms: dict[str, list[str]] = {}
    k1 = 1.5
    b = 0.75
    total_docs = len(docs)

    for template_id, tokens in docs.items():
        if not tokens:
            scores[template_id] = 0.0
            matched_terms[template_id] = []
            continue
        term_counts = {term: tokens.count(term) for term in set(query_terms)}
        score = 0.0
        matched = []
        for term, frequency in term_counts.items():
            if frequency <= 0:
                continue
            matched.append(term)
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens) / max(avgdl, 1))
            score += idf * (frequency * (k1 + 1)) / denominator
        scores[template_id] = score
        matched_terms[template_id] = matched
    return scores, matched_terms


def init_db(connection=None) -> None:
    with (nullcontext(connection) if connection is not None else get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
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
                    embedding public.vector({EMBEDDING_DIMENSION}),
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
                ON templates USING ivfflat (embedding public.vector_cosine_ops)
                WITH (lists = 100)
                """
            )


def row_to_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "private",
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
    return sorted([*list_public_templates(), *[row_to_template(row) for row in rows]], key=lambda template: template["title"])


def get_template(template_id: str) -> dict[str, Any] | None:
    if template_id.startswith("default:"):
        return next((template for template in list_public_templates() if template["id"] == template_id), None)
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
            SET embedding = %s::public.vector,
                embedding_text = %s,
                embedding_model = %s,
                embedding_updated_at = now(),
                updated_at = now()
            WHERE id = %s
            """,
            (vector_literal(vector), embedding_text, EMBEDDING_MODEL, template["id"]),
        )


def prioritize_template_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: (
        item["final_score"] >= MIN_MATCH_SCORE,
        item["template"].get("source") == "private" if item["final_score"] >= MIN_MATCH_SCORE else False,
        item["final_score"],
    ), reverse=True)


def hybrid_match_score(vector_score: float, keyword_score: float, has_keywords: bool) -> float:
    if not has_keywords:
        return vector_score
    total = VECTOR_WEIGHT + BM25_WEIGHT
    return (VECTOR_WEIGHT * vector_score + BM25_WEIGHT * keyword_score) / total if total > 0 else vector_score


def search_templates_with_metadata(query: str, limit: int = 3) -> list[dict[str, Any]]:
    init_db()
    query_embedding = embed_text(query)
    query_vector = vector_literal(query_embedding)
    candidate_limit = max(limit, RETRIEVAL_CANDIDATE_LIMIT)
    with get_connection() as conn:
        vector_rows = conn.execute(
            """
            SELECT id, title, category, description, tags, style, language,
                   use_cases, structure, content, enabled, version, created_at, updated_at,
                   embedding OPERATOR(public.<=>) %s::public.vector AS distance
            FROM templates
            WHERE enabled = true AND embedding IS NOT NULL
            ORDER BY embedding OPERATOR(public.<=>) %s::public.vector
            LIMIT %s
            """,
            (query_vector, query_vector, candidate_limit),
        ).fetchall()
        lexical_rows = conn.execute(
            """
            SELECT id, title, category, description, tags, style, language,
                   use_cases, structure, content, enabled, version, created_at, updated_at
            FROM templates
            WHERE enabled = true
            """,
        ).fetchall()

    public_templates = list_public_templates()
    public_distances = {}
    public_vectors = load_public_template_vectors(public_templates)
    for template in public_templates:
        vector = public_vectors[template["id"]]
        if len(vector) != len(query_embedding):
            raise ValueError("Default template embedding dimension does not match query")
        norm = math.sqrt(sum(value * value for value in vector) * sum(value * value for value in query_embedding))
        similarity = sum(a * b for a, b in zip(vector, query_embedding)) / norm if norm else 0.0
        public_distances[template["id"]] = 1.0 - similarity
    all_templates = [*public_templates, *[row_to_template(row) for row in lexical_rows]]
    lexical_scores, matched_terms = bm25_scores(query, all_templates)
    max_bm25 = max(lexical_scores.values(), default=0.0)
    lexical_top_ids = {
        template_id
        for template_id, _score in sorted(
            lexical_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_limit]
    }
    vector_distances = {
        row["id"]: float(row["distance"])
        for row in vector_rows
    }
    vector_distances.update(public_distances)
    vector_top_ids = set(vector_distances)
    templates_by_id = {template["id"]: template for template in all_templates}
    candidate_ids = vector_top_ids | lexical_top_ids

    ranked = []
    for template_id in candidate_ids:
        template = templates_by_id.get(template_id)
        if not template:
            continue
        vector_distance = vector_distances.get(template_id)
        vector_score = max(0.0, 1.0 - vector_distance) if vector_distance is not None else 0.0
        raw_bm25 = lexical_scores.get(template["id"], 0.0)
        normalized_bm25 = raw_bm25 / max_bm25 if max_bm25 else 0.0
        # Missing lexical evidence must not penalize queries in another language.
        final_score = hybrid_match_score(vector_score, normalized_bm25, max_bm25 > 0)
        if template_id in vector_top_ids and template_id in lexical_top_ids:
            source = "hybrid"
        elif template_id in vector_top_ids:
            source = "vector"
        else:
            source = "keyword"
        ranked.append(
            {
                "template": template,
                "distance": 1.0 - final_score,
                "vector_distance": vector_distance,
                "vector_score": vector_score,
                "bm25_score": normalized_bm25,
                "raw_bm25_score": raw_bm25,
                "final_score": final_score,
                "retrieval_source": source,
                "matched_terms": matched_terms.get(template["id"], []),
            }
        )
    ranked = prioritize_template_candidates(ranked)
    return ranked[:limit]


def read_template_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return normalize_template(data, template_id_from_filename(path))


def list_public_templates() -> list[dict[str, Any]]:
    # Only version-controlled bundled files are public. Never read legacy public
    # database tables or a tenant's configurable TEMPLATE_DIR here.
    templates = []
    for path in sorted((BACKEND_ROOT / "templates").glob("*.json")):
        template = read_template_file(path)
        if template["enabled"]:
            templates.append({**template, "id": f"default:{path.stem}", "source": "default"})
    return templates


def load_public_template_vectors(templates: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Persist only trusted bundled content; never cache private templates here."""
    texts = {template["id"]: build_embedding_text(template) for template in templates}
    hashes = {key: hashlib.sha256(text.encode()).hexdigest() for key, text in texts.items()}
    vectors = {}
    with get_connection() as conn:
        # Serialize cold-cache fills across workers and tenants.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('heywrite-public-template-vectors'))")
        conn.execute("""CREATE TABLE IF NOT EXISTS public.heywrite_default_template_vectors (
            template_id text NOT NULL, content_hash text NOT NULL, model text NOT NULL,
            dimension integer NOT NULL, embedding jsonb NOT NULL,
            PRIMARY KEY (template_id, content_hash, model, dimension)
        )""")
        rows = conn.execute("""SELECT template_id, content_hash, embedding
            FROM public.heywrite_default_template_vectors WHERE model = %s AND dimension = %s""",
            (EMBEDDING_MODEL, EMBEDDING_DIMENSION)).fetchall()
        for row in rows:
            if hashes.get(row["template_id"]) == row["content_hash"]:
                vectors[row["template_id"]] = row["embedding"]
        missing = [key for key in texts if key not in vectors]
        if missing:
            batch = get_embedding_client().embed_documents([texts[key] for key in missing])
            if len(batch) != len(missing) or any(len(vector) != EMBEDDING_DIMENSION for vector in batch):
                raise ValueError("Default template embedding batch has invalid dimensions")
            for key, vector in zip(missing, batch):
                conn.execute("""INSERT INTO public.heywrite_default_template_vectors
                    (template_id, content_hash, model, dimension, embedding)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                    (key, hashes[key], EMBEDDING_MODEL, EMBEDDING_DIMENSION, Jsonb(vector)))
                vectors[key] = vector
    return vectors


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
