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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required for text-embedding-3-small")

# 初始化嵌入模型
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)


# 初始化 Chroma 向量库（假设已经预先插入模板内容）
try:
    persist_directory = "./chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embedding
    )
    logger.info("成功加载 Chroma 向量库")
except Exception as e:
    logger.error(f" 加载 Chroma 向量库失败: {e}")
    raise

SIMILARITY_THRESHOLD = 0.38

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

# 初始化 DeepSeek 模型
try:
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7, api_key=DEEPSEEK_API_KEY)
    logger.info("DeepSeek 模型初始化成功")
except Exception as e:
    logger.error(f" DeepSeek 模型初始化失败: {e}")
    raise

# 构造 LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# 主函数
def generate_with_template(
    intent: str, style: str, language: str, history: list[dict] = None
) -> str:
    try:
        if not intent.strip():
            return "please provide a valid intent"

        logger.info(f"searching intent: {intent}")
        results = vectorstore.similarity_search_with_score(intent, k=3)

        if not results:
            logger.warning("did not find any template matched, please try the wild mode")
            return "did not find any template matched, please try the wild mode"

        top_doc, top_dist = results[0]
        logger.info(f"[template-match] dist={top_dist:.4f}")

        if top_dist > SIMILARITY_THRESHOLD:
            logger.warning("distance too far -> wild mode")
            return "did not find any template matched, please try the wild mode"

        docs = [doc for doc, _ in results]
        logger.info(f"found {len(docs)} templates")
        context = "\n\n".join([doc.page_content for doc in docs])

        # 提取最近一条 assistant
        previous = ""
        if history:
            for msg in reversed(history):
                if msg.role == "assistant":
                    previous = msg.content
                    break

        result = llm_chain.run(
            {
                "intent": intent,
                "style": style or "formal",
                "language": language or "English",
                "context": context,
                "previous": previous,
            }
        )

        logger.info("generate content successfully")
        return result.strip()

    except Exception as e:
        logger.error(f"failed to generate content: {e}", exc_info=True)
        return "failed to generate content :" + str(e)
