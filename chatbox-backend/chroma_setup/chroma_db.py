
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os, json

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is required for text-embedding-3-small")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key,
)

docs = []

TEMPLATE_DIR = "./templates"  

for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith(".json"):
        with open(os.path.join(TEMPLATE_DIR, filename)) as f:
            data = json.load(f)
            content = data.get("content", "") or json.dumps(data, ensure_ascii=False)
            docs.append(Document(page_content=content, metadata={"filename": filename}))

# 存入 Chroma
Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="./chroma_db")
