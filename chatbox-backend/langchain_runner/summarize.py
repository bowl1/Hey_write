import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


FIRST_CHUNKS = 3
MIDDLE_CHUNKS = 2
LAST_CHUNKS = 2
MAX_SUMMARY_PER_FILE = 8


def safe_int(meta: dict, key: str, default: int) -> int:
    try:
        return int(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def pick_chunk_indices(count: int) -> List[int]:
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


def select_summary_docs(docs: List[Document]) -> List[Document]:
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
            page = safe_int(meta, "page", 0)
            paragraph = safe_int(meta, "paragraph", safe_int(meta, "line", 0))
            return (1, page, paragraph, insertion_idx)

        ordered_docs = [doc for _, doc in sorted(items, key=sort_key)]
        indices = pick_chunk_indices(len(ordered_docs))
        selected.extend(ordered_docs[i] for i in indices)

    return selected


def summarize_docs(docs: List[Document], language: str, style: str) -> str:
    if not docs:
        return "I cannot find any documents to summarize."

    docs = select_summary_docs(docs)
    logger.info(f"[Summary] selected {len(docs)} chunks for summarization")

    chunk_summaries = []
    for doc in docs:
        from .chains import get_summary_map_chain

        summary = get_summary_map_chain().invoke(
            {"chunk": doc.page_content, "language": language, "style": style}
        ).strip()
        chunk_summaries.append(summary)

    combined = "\n".join(chunk_summaries)
    from .chains import get_summary_reduce_chain

    return get_summary_reduce_chain().invoke(
        {"summaries": combined, "language": language, "style": style}
    ).strip()


def summarize_all_docs(language: str, style: str) -> str:
    from .vectorstore import get_vectorstore

    raw = get_vectorstore().get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
    ]
    return summarize_docs(docs, language, style)
