from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_deepseek import ChatDeepSeek
import os
import logging
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 加载环境变量
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

# PromptTemplate
prompt = PromptTemplate.from_template(
    """
You are a professional writing assistant.

Your task is to revise the **previous version** of a document based on the user’s intent.
You **must retain** the structure and content of the previous version, and make **minimal, localized edits** only where necessary to reflect the intent, style, and language preferences.

Do **not** rewrite the entire document.
Do **not** change parts unrelated to the new intent.
Do **not** invent any new content beyond what is clearly implied by the user intent.

Use only plain text for formatting. Separate sections using blank lines if needed.

---

### User Guidance:

If the user's intent is to modify or add to a specific section (e.g., “Comments”, “Summary”, “Tasks”), then:

- If that section exists in the previous version, update it accordingly.
- If that section does **not** exist, create it in a logically appropriate position in the document.
- Do not remove or restructure unrelated sections.

If the user's intent has **no meaningful relation** to the reference template or previous version,
respond with the following text **exactly**, and do not include any document body or change summary:

"Did not find any template matched, please try the wild mode."

---

User Intent:
{intent}

Preferred Style: {style}
Language: {language}

Reference Template:
{context}

Previous Version:
{previous}


At the end of your response, provide a summary of changes made in this format:

Changes:
- [brief description of change 1]
- [brief description of change 2]
(If no changes, write "No changes made.")
"""
)

embedding = None
vectorstore = None
llm = None
llm_chain = None


def get_embedding():
    global embedding
    if embedding is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for text-embedding-3-small")
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
        )
    return embedding


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        persist_directory = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=get_embedding(),
        )
        logger.info("成功加载 Chroma 向量库")
    return vectorstore


def get_llm():
    global llm
    if llm is None:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for deepseek-chat")
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            api_key=deepseek_api_key,
        )
        logger.info("DeepSeek 模型初始化成功")
    return llm


def get_llm_chain():
    global llm_chain
    if llm_chain is None:
        llm_chain = LLMChain(llm=get_llm(), prompt=prompt)
    return llm_chain

# 主函数
def generate_with_template(
    intent: str, style: str, language: str, history: list[dict] = None
) -> dict:
    try:
        if not intent.strip():
            return {"reply": "please provide a valid intent", "template_meta": None}

        logger.info(f"searching intent: {intent}")
        results = get_vectorstore().similarity_search_with_score(intent, k=3)

        if not results:
            logger.warning("did not find any template matched, please try the wild mode")
            return {
                "reply": "did not find any template matched, please try the wild mode",
                "template_meta": {
                    "used_template": False,
                    "reason": "No templates were returned by retrieval.",
                },
            }

        top_doc, top_dist = results[0]
        logger.info(f"[template-match] dist={top_dist:.4f}")

        if top_dist > SIMILARITY_THRESHOLD:
            logger.warning("distance too far -> wild mode")
            return {
                "reply": "did not find any template matched, please try the wild mode",
                "template_meta": {
                    "used_template": False,
                    "match_score": top_dist,
                    "selected_template": top_doc.metadata.get("template_title", ""),
                    "reason": f"Best template distance {top_dist:.2f} exceeded threshold {SIMILARITY_THRESHOLD:.2f}.",
                },
            }

        docs = [doc for doc, _ in results]
        logger.info(f"found {len(docs)} templates")
        context = "\n\n".join(
            [
                doc.metadata.get("template_content", doc.page_content)
                for doc in docs
            ]
        )

        # 提取最近一条 assistant
        previous = ""
        if history:
            for msg in reversed(history):
                if msg.role == "assistant":
                    previous = msg.content
                    break

        result = get_llm_chain().run(
            {
                "intent": intent,
                "style": style or "formal",
                "language": language or "English",
                "context": context,
                "previous": previous,
            }
        )

        logger.info("generate content successfully")
        return {
            "reply": result.strip(),
            "template_meta": {
                "used_template": True,
                "match_score": top_dist,
                "selected_template": top_doc.metadata.get("template_title", ""),
                "selected_template_id": top_doc.metadata.get("template_id", ""),
                "reason": f"Selected because it was the closest template match with distance {top_dist:.2f}.",
            },
        }

    except Exception as e:
        logger.error(f"failed to generate content: {e}", exc_info=True)
        return {"reply": "failed to generate content :" + str(e), "template_meta": None}
