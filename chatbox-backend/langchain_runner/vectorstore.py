import logging
import re
from io import BytesIO
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "intfloat/e5-small-v2"
PERSIST_DIRECTORY = "./chroma_db"

embedding = None
vectorstore = None
retriever = None


def get_embedding():
    global embedding
    if embedding is None:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embedding


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=get_embedding(),
        )
    return vectorstore


def get_retriever():
    global retriever
    if retriever is None:
        retriever = get_vectorstore().as_retriever(search_kwargs={"k": 4})
    return retriever


def add_pdf_to_vectorstore(file_bytes: bytes, filename: str) -> int:
    """Parse PDF bytes, split to chunks with page metadata, and store in Chroma."""
    reader = PdfReader(BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    texts: List[str] = []
    metadatas: List[dict] = []
    chunk_counter = 0
    for idx, page_text in enumerate(pages_text, start=1):
        if not page_text.strip():
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        for para_idx, paragraph in enumerate(paragraphs, start=1):
            chunks = splitter.split_text(paragraph)
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append(
                    {
                        "page": idx,
                        "source": filename,
                        "paragraph": para_idx,
                        "chunk_order": chunk_counter,
                    }
                )
                chunk_counter += 1

    if not texts:
        return 0

    store = get_vectorstore()
    store.add_texts(texts=texts, metadatas=metadatas)
    store.persist()
    logger.info(f"[PDF ingest] {filename} -> pages: {len(pages_text)}, chunks: {len(texts)}")
    return len(texts)


def delete_pdfs(file_names: List[str]) -> int:
    """Delete vectors for given file names and persist."""
    if not file_names:
        return 0
    deleted = 0
    for name in file_names:
        try:
            get_vectorstore().delete(where={"source": name})
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete vectors for {name}: {e}", exc_info=True)
    get_vectorstore().persist()
    return deleted
