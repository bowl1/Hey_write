from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
from agent import AgentRunRequest, AgentRunResponse, run_agent
from agent.schemas import SessionListResponse, SessionSnapshot, MemoryProfileResponse, MemorySettingsRequest
from agent.memory_store import load_profile, configure_profile
from agent.long_term_memory import enabled as memory_feature_enabled
from agent.store import list_session_summaries, load_session_snapshot, delete_session
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


@app.delete("/agent/session/{session_id}", status_code=204)
def delete_agent_session(session_id: str) -> Response:
    try:
        if not delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception:
        logger.error("Failed to delete agent session", exc_info=True)
        raise HTTPException(status_code=503, detail="Session could not be deleted. Please try again.")


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


def memory_response(profile: dict) -> MemoryProfileResponse:
    return MemoryProfileResponse(**profile, available=memory_feature_enabled("AGENT_MEMORY_ENABLED"))


@app.get("/agent/memory", response_model=MemoryProfileResponse)
def personal_memory():
    return memory_response(load_profile())


@app.patch("/agent/memory", response_model=MemoryProfileResponse)
def memory_settings(body: MemorySettingsRequest):
    return memory_response(configure_profile(enabled=body.enabled))


@app.delete("/agent/memory", response_model=MemoryProfileResponse)
def clear_personal_memory():
    return memory_response(configure_profile(clear=True))
