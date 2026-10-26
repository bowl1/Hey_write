from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
from agent import AgentRunRequest, AgentRunResponse, run_agent
from agent.schemas import SessionListResponse, SessionSnapshot
from agent.store import list_session_summaries, load_session_snapshot
from schemas import (
    HealthResponse, IndexStatus, Template, TemplateCreateRequest, TemplateCreateResponse,
    TemplateListResponse, WorkspaceResponse,
)
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
    with tenant_context(str(user.id)):
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
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/workspace")
def workspace() -> WorkspaceResponse:
    return WorkspaceResponse(tenant_id=tenant_id())


@app.post("/agent/run")
def agent_run(body: AgentRunRequest) -> AgentRunResponse:
    try:
        return run_agent(body)
    except Exception as e:
        logger.error("Agent run failed", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {e}")


@app.get("/agent/sessions")
def agent_sessions() -> SessionListResponse:
    try:
        return SessionListResponse(sessions=list_session_summaries())
    except Exception as e:
        logger.error("Failed to list agent sessions", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Agent sessions unavailable: {e}")


@app.get("/agent/session/{session_id}")
def agent_session(session_id: str) -> SessionSnapshot:
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
def templates() -> TemplateListResponse:
    try:
        return TemplateListResponse.model_validate({"templates": list_templates()})
    except Exception as e:
        logger.error("Failed to list templates", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")


@app.get("/public/templates")
def public_templates() -> TemplateListResponse:
    return TemplateListResponse.model_validate({"templates": list_public_templates()})


@app.get("/templates/{template_id}")
def template_detail(template_id: str) -> Template:
    try:
        template = get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return Template.model_validate(template)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get template", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")


@app.post("/templates")
def create_template(body: TemplateCreateRequest) -> TemplateCreateResponse:
    try:
        template = save_template(body.model_dump(), embed=False)
        index_status = IndexStatus(ok=True)
        try:
            upsert_template_embedding(template)
        except Exception as e:
            logger.error("Template saved but reindex failed", exc_info=True)
            index_status = IndexStatus(ok=False, error=str(e))
        return TemplateCreateResponse(template=Template.model_validate(template), index_status=index_status)
    except Exception as e:
        logger.error("Failed to create template", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Template store unavailable: {e}")
