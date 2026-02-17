import json
import os

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "./templates")
DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")


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
        return client._collection.count() > 0
    except Exception:
        return False


def build_if_missing() -> None:
    if _has_vectors(DB_DIR):
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
    for filename in os.listdir(TEMPLATE_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(TEMPLATE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            content = data.get("content", "") or json.dumps(data, ensure_ascii=False)
            docs.append(Document(page_content=content, metadata={"source": filename}))

    if not docs:
        raise ValueError(f"No templates found in {TEMPLATE_DIR}")

    Chroma.from_documents(docs, embedding, persist_directory=DB_DIR)
    print("Vector DB built successfully")
