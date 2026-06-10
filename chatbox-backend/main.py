from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import logging
from template_store import list_templates, get_template, save_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class RequestBody(BaseModel):
    intent: str
    style: str = "Formal"
    language: str = "English"
    history: list[ChatMessage] = []  # 对话历史


class TemplateCreateRequest(BaseModel):
    title: str
    category: str = "general"
    description: str = ""
    tags: list[str] = []
    style: str = "Formal"
    language: str = "English"
    use_cases: list[str] = []
    content: str


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


@app.post("/write")
async def generate_text(body: RequestBody):
    system_prompt = """
    You are a professional writing assistant for workplace communication.

    Your job is to generate practical, polished, and natural-sounding text based on the user's intent, preferred style, and language. Follow these rules strictly:

    Ensure the tone matches the requested style (e.g., formal, friendly, concise).
    Avoid unnecessary elaboration or repetition.
    Include all key details mentioned by the user.
    Do **not** format using Markdown symbols like ##, **, -, or *.

    At the end of the generated text, provide a **Changes** section summarizing the main points or edits made based on the user's intent.

    (If it's a first draft, list the main elements included.)
    """
    # 构造本轮的 user prompt
    current_user_prompt = f"intent:{body.intent}\nstyle:{body.style}\nlanguage:{body.language}\nPlease generate a suitable text based on the information provided."

    # 构造完整 message 列表（含历史）
    messages = [{"role": "system", "content": system_prompt}]
    for msg in body.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": current_user_prompt})
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.5,
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Something went wrong: {str(e)}"

    return {"reply": reply}


@app.post("/write_with_template")
def write_with_template(body: RequestBody):
    try:
        from langchain_runner.build_vectorstore import build_if_missing
        from langchain_runner.rag_chain import generate_with_template

        build_if_missing()
        result = generate_with_template(
            body.intent, body.style, body.language, body.history
        )
    except Exception as e:
        logger.error("Template generation failed during initialization or generation", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template generation unavailable: {e}")

    return result


@app.get("/templates")
def templates():
    return {"templates": list_templates()}


@app.get("/templates/{template_id}")
def template_detail(template_id: str):
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@app.post("/templates")
def create_template(body: TemplateCreateRequest):
    template_data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    template = save_template(template_data)
    index_status = {"ok": True, "error": None}
    try:
        from langchain_runner.build_vectorstore import build_if_missing

        build_if_missing(force=True)
    except Exception as e:
        logger.error("Template saved but reindex failed", exc_info=True)
        index_status = {"ok": False, "error": str(e)}
    return {"template": template, "index_status": index_status}
