import json
import os
import shutil

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

load_dotenv()

TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "./templates")
DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

REWRITE_PROMPT = """
You are generating a semantic intent description for vector search.

Convert the template into a single sentence describing:
WHAT the user wants to do.

Rules:
- Describe user intention, not the document name
- Start with a verb
- 10~20 words
- No formatting words like "template" or "document"
- Must be concrete and actionable

Template Info:
Title: {title}
Tags: {tags}
Language: {language}
Content Preview:
{preview}

Return ONLY the sentence.
"""

rewrite_llm = None
if DEEPSEEK_API_KEY:
    rewrite_llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
    )


def _has_vectors(db_dir: str) -> bool:
    if not os.path.isdir(db_dir):
        return False
    if len(os.listdir(db_dir)) == 0:
        return False
    try:
        client = Chroma(
            persist_directory=db_dir,
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        )
        if client._collection.count() <= 0:
            return False
        # Router schema requires template_content in metadata.
        sample = client._collection.get(include=["metadatas"], limit=1)
        metas = sample.get("metadatas") or []
        if not metas:
            return False
        first_meta = metas[0] or {}
        return "template_content" in first_meta
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
    if rewrite_llm:
        response = rewrite_llm.invoke(
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


def build_if_missing() -> None:
    if _has_vectors(DB_DIR):
        print("Vector DB already exists")
        return

    print("Building vector database from templates...")
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for text-embedding-3-small")

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    docs = []
    for filename in os.listdir(TEMPLATE_DIR):
        if not filename.endswith(".json") or filename.endswith(".embedding.json"):
            continue
        path = os.path.join(TEMPLATE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            description = _build_description(data, filename)
            content = data.get("content", "") or json.dumps(data, ensure_ascii=False)
            docs.append(
                Document(
                    page_content=description,
                    metadata={
                        "source": filename,
                        "template_content": content,
                        "description": description,
                    },
                )
            )

    if not docs:
        raise ValueError(f"No templates found in {TEMPLATE_DIR}")

    Chroma.from_documents(docs, embedding, persist_directory=DB_DIR)
    print("Vector DB built successfully")
