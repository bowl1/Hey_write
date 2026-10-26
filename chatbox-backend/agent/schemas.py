from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentRunRequest(BaseModel):
    session_id: str | None = None
    action: Literal["new_task", "continue_editing", "wild"]
    intent: str
    style: str = "Formal"
    language: str = "English"
    history: list[AgentMessage] = []
    current_draft: str = ""
    selected_template_id: str | None = None
    active_template_id: str | None = None


class AgentRunState(BaseModel):
    session_id: str
    active_template_id: str | None = None
    current_draft: str = ""
    draft_accepted: bool = False
    blocked_reason: str | None = None


class AgentRunResponse(BaseModel):
    reply: str
    template_meta: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    state: AgentRunState
    trace: list[dict[str, Any]]


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: datetime | None = None


class AgentSession(BaseModel):
    session_id: str
    active_template_id: str | None = None
    current_draft: str = ""
    style: str = "Formal"
    language: str = "English"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentRunRecord(BaseModel):
    run_id: str
    action: str
    intent: str
    selected_template_id: str | None = None
    output_state: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    reply: str = ""
    created_at: datetime | None = None


class SessionSnapshot(BaseModel):
    session: AgentSession
    messages: list[SessionMessage]
    last_run: AgentRunRecord | None = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    active_template_id: str | None = None
    current_draft: str = ""
    style: str = "Formal"
    language: str = "English"
    message_count: int
    updated_at: datetime | None = None
    messages: list[SessionMessage]


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
