from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import json
import shutil

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is required for text-embedding-3-small")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key,
)

docs = []
template_dir = os.getenv("TEMPLATE_DIR", "./templates")
persist_directory = os.getenv("CHROMA_DB_DIR", "./chroma_db")

if os.path.isdir(persist_directory):
    shutil.rmtree(persist_directory)

for filename in os.listdir(template_dir):
    if not filename.endswith(".json") or filename.endswith(".embedding.json"):
        continue
    with open(os.path.join(template_dir, filename), "r", encoding="utf-8") as f:
        data = json.load(f)
        description = (data.get("description") or "").strip()
        if not description:
            title = (data.get("title") or "").strip()
            category = (data.get("category") or "").strip()
            language = (data.get("language") or "").strip()
            parts = [p for p in [title, category, language] if p]
            if parts:
                description = "Template for " + ", ".join(parts)
            else:
                description = (
                    f"Template intent for {filename.replace('.json', '').replace('_', ' ')}"
                )
        content = data.get("content", "") or json.dumps(data, ensure_ascii=False)
        docs.append(
            Document(
                page_content=description,
                metadata={
                    "filename": filename,
                    "template_content": content,
                    "description": description,
                },
            )
        )

Chroma.from_documents(
    documents=docs, embedding=embedding, persist_directory=persist_directory
)
