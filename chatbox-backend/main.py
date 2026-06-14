from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from agent import AgentRunRequest, run_agent
from template_store import list_templates, get_template, save_template, upsert_template_embedding

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


class TemplateCreateRequest(BaseModel):
    title: str
    category: str = "general"
    description: str = ""
    tags: list[str] = []
    style: str = "Formal"
    language: str = "English"
    use_cases: list[str] = []
    content: str


@app.post("/agent/run")
def agent_run(body: AgentRunRequest):
    try:
        return run_agent(body)
    except Exception as e:
        logger.error("Agent run failed", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {e}")


@app.get("/templates")
def templates():
    try:
        return {"templates": list_templates()}
    except Exception as e:
        logger.error("Failed to list templates", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")


@app.get("/templates/{template_id}")
def template_detail(template_id: str):
    try:
        template = get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get template", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")


@app.post("/templates")
def create_template(body: TemplateCreateRequest):
    try:
        template_data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        template = save_template(template_data, embed=False)
        index_status = {"ok": True, "error": None}
        try:
            upsert_template_embedding(template)
        except Exception as e:
            logger.error("Template saved but reindex failed", exc_info=True)
            index_status = {"ok": False, "error": str(e)}
        return {"template": template, "index_status": index_status}
    except Exception as e:
        logger.error("Failed to create template", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")
