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
if DEEPSEEK_API_KEY:
    rewrite_llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
    )


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
        query_dim = len(embedding.embed_query("dimension check"))

        if db_dim != query_dim:
            print(f"Embedding dimension mismatch: db={db_dim}, query={query_dim}")
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

            # 1. 取模板正文
            content = data.get("content", "") or json.dumps(data, ensure_ascii=False)

            # 2. 生成用户query描述（多行）
            description = _build_description(data, filename)

            queries = [q.strip() for q in description.split("\n") if q.strip()]

            # 3. 每个query写入一个向量
            for q in queries:
                docs.append(
                    Document(
                        page_content=q,
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
    
if __name__ == "__main__":
    build_if_missing()
