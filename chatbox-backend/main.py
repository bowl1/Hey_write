from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
from agent import AgentRunRequest, run_agent
from agent.store import list_session_summaries, load_session_snapshot
from template_store import list_templates, list_public_templates, get_template, save_template, upsert_template_embedding
from observability import shutdown_observability
from tenancy import authenticate, tenant_context, tenant_id, ensure_workspace
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        shutdown_observability()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def tenant_middleware(request, call_next):
    if request.method == "OPTIONS" or request.url.path == "/health" or (request.method == "GET" and request.url.path == "/public/templates"):
        return await call_next(request)
    try:
        user = await authenticate(request.headers.get("authorization"))
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    with tenant_context(user["id"]):
        request.state.user = user
        try:
            await run_in_threadpool(ensure_workspace)
        except Exception:
            logger.error("Workspace access check failed", exc_info=True)
            return JSONResponse({"detail": "Workspace unavailable"}, status_code=503)
        return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/workspace")
def workspace():
    return {"tenant_id": tenant_id()}


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


@app.get("/agent/sessions")
def agent_sessions():
    try:
        return {"sessions": list_session_summaries()}
    except Exception as e:
        logger.error("Failed to list agent sessions", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Agent sessions unavailable: {e}")


@app.get("/agent/session/{session_id}")
def agent_session(session_id: str):
    try:
        snapshot = load_session_snapshot(session_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Session not found")
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load agent session", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Agent session unavailable: {e}")


@app.get("/templates")
def templates():
    try:
        return {"templates": list_templates()}
    except Exception as e:
        logger.error("Failed to list templates", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")


@app.get("/public/templates")
def public_templates():
    return {"templates": list_public_templates()}


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
