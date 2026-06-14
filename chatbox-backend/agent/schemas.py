from typing import Literal

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
    active_template_id: str | None = None


class AgentRunResponse(BaseModel):
    reply: str
    template_meta: dict | None = None
    evaluation: dict | None = None
    state: dict
    trace: list[dict]
