import json
import os
import shutil
import tempfile

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from template_store import list_templates

load_dotenv()

TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "./templates")
DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_EMBEDDING_DIM = 1536

REWRITE_PROMPT = """
You are generating example user requests.

Write 5 short natural phrases a user would type 
to ask an AI to generate this kind of document.

Rules:
- conversational English
- 3–8 words
- one per line
- no numbering
- no explanations

Template:
Title: {title}
Tags: {tags}
Content:
{preview}

Return ONLY the sentence.
"""

rewrite_llm = None


def get_rewrite_llm():
    global rewrite_llm
    if rewrite_llm is None and DEEPSEEK_API_KEY:
        rewrite_llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
        )
    return rewrite_llm


def _has_vectors(db_dir: str) -> bool:
    if not os.path.isdir(db_dir):
        return False

    try:
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        client = Chroma(
            persist_directory=db_dir,
            embedding_function=embedding,
        )

        if client._collection.count() <= 0:
            return False

        # 新增：维度检查
        sample = client._collection.get(include=["embeddings"], limit=1)
        db_dim = len(sample["embeddings"][0])

        if db_dim != OPENAI_EMBEDDING_DIM:
            print(f"Embedding dimension mismatch: db={db_dim}, expected={OPENAI_EMBEDDING_DIM}")
            return False

        return True

    except Exception:
        return False


def _build_description(data: dict, filename: str) -> str:
    cache_path = os.path.join(
        TEMPLATE_DIR, filename.replace(".json", ".embedding.json")
    )
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            cached_description = (cache_data.get("description") or "").strip()
            if cached_description:
                return cached_description
        except Exception:
            pass

    description = (data.get("description") or "").strip()
    if description:
        return description

    title = (data.get("title") or "").strip()
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags_text = ", ".join([str(t).strip() for t in tags if str(t).strip()])
    language = (data.get("language") or "").strip()
    preview = (data.get("content") or "")[:400]

    generated_description = ""
    active_rewrite_llm = get_rewrite_llm()
    if active_rewrite_llm:
        response = active_rewrite_llm.invoke(
            REWRITE_PROMPT.format(
                title=title,
                tags=tags_text,
                language=language,
                preview=preview,
            )
        )
        generated_description = (getattr(response, "content", "") or "").strip()

    if not generated_description:
        # Deterministic fallback when rewrite LLM is unavailable.
        if title:
            generated_description = f"Write a {title.lower()} with clear purpose, details, and next-step actions."
        else:
            generated_description = "Write a clear business message with purpose, key details, and actionable follow-up."

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"description": generated_description}, f, ensure_ascii=False, indent=2)

    return generated_description


def build_if_missing(force: bool = False) -> None:
    if not force and _has_vectors(DB_DIR):
        print("Vector DB already exists")
        return

    print("Building vector database from templates...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for text-embedding-3-small")

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    docs = []
    for template in list_templates():
        if not template.get("enabled", True):
            continue
        content = template.get("content", "")
        description = _build_description(template, f"{template['id']}.json")
        search_text = "\n".join(
            [
                f"Title: {template.get('title', '')}",
                f"Category: {template.get('category', '')}",
                f"Description: {description}",
                "Tags: " + ", ".join(template.get("tags", [])),
                "Use cases:",
                *[f"- {case}" for case in template.get("use_cases", [])],
            ]
        )
        docs.append(
            Document(
                page_content=search_text,
                metadata={
                    "template_id": template["id"],
                    "template_title": template.get("title", ""),
                    "template_category": template.get("category", ""),
                    "template_content": content,
                    "description": description,
                },
            )
        )

    if not docs:
        raise ValueError(f"No templates found in {TEMPLATE_DIR}")

    parent_dir = os.path.dirname(os.path.abspath(DB_DIR)) or "."
    tmp_dir = tempfile.mkdtemp(prefix="chroma_build_", dir=parent_dir)
    try:
        Chroma.from_documents(docs, embedding, persist_directory=tmp_dir)
        if os.path.isdir(DB_DIR):
            shutil.rmtree(DB_DIR)
        shutil.move(tmp_dir, DB_DIR)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print("Vector DB built successfully")
    
if __name__ == "__main__":
    build_if_missing()
