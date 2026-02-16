import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ROOT_ENV)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

try:
    persist_directory = "./chroma_db"
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    logger.info("Successfully loaded Chroma vector library")
except Exception as e:
    logger.error(f"Failed to load Chroma vector library: {e}")
    raise

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say: "I cannot find the answer in the provided documents."

Respond in the requested language: {language}
Use the requested tone/style: {style}

Keep the answer concise and cite page numbers inline when relevant, e.g., (page 3).

Question: {question}

Context:
{context}
"""
)

summary_map_prompt = PromptTemplate.from_template(
    """
You are creating a brief summary of a PDF chunk.
Respond in the requested language: {language}. Use the requested tone/style: {style}.

Chunk:
{chunk}

Write a concise 1-2 sentence summary.
"""
)

summary_reduce_prompt = PromptTemplate.from_template(
    """
You are creating an overall summary from chunk summaries.
Respond in the requested language: {language}. Use the requested tone/style: {style}.

Chunk summaries:
{summaries}

Write a concise overall summary (3-5 sentences) highlighting the main idea and key points.
"""
)

try:
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3, api_key=DEEPSEEK_API_KEY)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    summary_map_chain = LLMChain(llm=llm, prompt=summary_map_prompt)
    summary_reduce_chain = LLMChain(llm=llm, prompt=summary_reduce_prompt)
    logger.info("DeepSeek model initialization successful")
except Exception as e:
    logger.error(f"DeepSeek model initialization failed: {e}")
    raise

FIRST_CHUNKS = 3
MIDDLE_CHUNKS = 2
LAST_CHUNKS = 2
MAX_SUMMARY_PER_FILE = 8


def _safe_int(meta: dict, key: str, default: int) -> int:
    try:
        return int(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _pick_chunk_indices(count: int) -> List[int]:
    """Return ordered indices for first/middle/last slices with de-duping."""
    if count <= 0:
        return []

    candidates: List[int] = []
    candidates.extend(range(min(FIRST_CHUNKS, count)))

    mid_indices: List[int] = []
    if count > 0:
        mid_center = count // 2
        if MIDDLE_CHUNKS >= 2 and count >= 2:
            mid_indices = [max(0, mid_center - 1), mid_center]
        else:
            mid_indices = [mid_center]
    candidates.extend(mid_indices)

    last_start = max(count - LAST_CHUNKS, 0)
    candidates.extend(range(last_start, count))

    seen = set()
    ordered_unique = []
    for idx in candidates:
        if 0 <= idx < count and idx not in seen:
            ordered_unique.append(idx)
            seen.add(idx)

    ordered_unique.sort()
    return ordered_unique[:MAX_SUMMARY_PER_FILE]


def _select_summary_docs(docs: List[Document]) -> List[Document]:
    """Pick representative chunks per source (first/middle/last)."""
    grouped = {}
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        grouped.setdefault(source, []).append((idx, doc))

    selected: List[Document] = []
    for _, items in grouped.items():
        def sort_key(item):
            insertion_idx, document = item
            meta = document.metadata or {}
            chunk_order = meta.get("chunk_order")
            if chunk_order is not None:
                try:
                    return (0, int(chunk_order))
                except (TypeError, ValueError):
                    return (0, chunk_order)
            page = _safe_int(meta, "page", 0)
            paragraph = _safe_int(meta, "paragraph", _safe_int(meta, "line", 0))
            return (1, page, paragraph, insertion_idx)

        ordered_docs = [doc for _, doc in sorted(items, key=sort_key)]
        indices = _pick_chunk_indices(len(ordered_docs))
        selected.extend(ordered_docs[i] for i in indices)

    return selected

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

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    vectorstore.persist()
    logger.info(f"[PDF ingest] {filename} -> pages: {len(pages_text)}, chunks: {len(texts)}")
    return len(texts)

def delete_pdfs(file_names: List[str]) -> int:
    """Delete vectors for given file names and persist."""
    if not file_names:
        return 0
    deleted = 0
    for name in file_names:
        try:
            vectorstore.delete(where={"source": name})
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete vectors for {name}: {e}", exc_info=True)
    vectorstore.persist()
    return deleted


def summarize_docs(docs: List, language: str, style: str) -> str:
    if not docs:
        return "I cannot find any documents to summarize."

    docs = _select_summary_docs(docs)
    logger.info(f"[Summary] selected {len(docs)} chunks for summarization")

    chunk_summaries = []
    for doc in docs:
        summary = summary_map_chain.predict(
            chunk=doc.page_content, language=language, style=style
        ).strip()
        chunk_summaries.append(summary)

    combined = "\n".join(chunk_summaries)
    overall = summary_reduce_chain.predict(
        summaries=combined, language=language, style=style
    ).strip()
    return overall


def summarize_all_docs(language: str, style: str) -> str:
    raw = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
    ]
    return summarize_docs(docs, language, style)


def answer_with_context(question: str, language: str, style: str) -> Tuple[str, List[dict]]:
    if not question.strip():
        return "please provide a valid question", []

    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "I cannot find the answer in the provided documents.", []
    context = "\n\n".join(doc.page_content for doc in docs)
    logger.info(f"[QA] retrieved {len(docs)} docs for question: {question[:80]}")
    result = llm_chain.predict(
        question=question, context=context, language=language, style=style
    ).strip()
    # 🔒 Guardrail：如果没有引用 page，拒答
    if "(page" not in result.lower():
        return "I cannot find the answer in the provided documents.", []
    
    sources = []
    for idx, doc in enumerate(docs, start=1):
        para = doc.metadata.get("paragraph") or doc.metadata.get("line") or idx
        sources.append(
            {
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", ""),
                "paragraph": para,
            }
        )
    return result, sources
