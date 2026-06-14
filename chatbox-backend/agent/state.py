from typing import Any, Literal, TypedDict

AgentAction = Literal["new_task", "continue_editing", "wild"]


class WritingAgentState(TypedDict, total=False):
    session_id: str
    run_id: str
    action: AgentAction
    intent: str
    style: str
    language: str
    history: list[dict[str, str]]
    active_template_id: str | None
    active_template: dict[str, Any] | None
    current_draft: str
    previous_draft: str
    retrieved_templates: list[dict[str, Any]]
    template_meta: dict[str, Any] | None
    next_tool: str | None
    steps: int
    max_steps: int
    last_tool: str | None
    observations: list[dict[str, Any]]
    evaluation: dict[str, Any]
    retry_count: int
    max_retries: int
    evaluation_feedback: str
    reply: str
    trace: list[dict[str, Any]]
    error: str | None
