from __future__ import annotations

from uuid import uuid4

from langgraph.graph import END, StateGraph

from .nodes import (
    evaluator_node,
    finalize_node,
    generate_draft_node,
    generate_wild_node,
    load_active_template_node,
    load_session_node,
    observe_node,
    planner_node,
    persist_state_node,
    retrieve_template_node,
    revise_with_feedback_node,
    revise_draft_node,
    route_after_evaluation,
    route_next_tool,
)
from .schemas import AgentRunRequest
from .state import WritingAgentState


def build_graph():
    graph = StateGraph(WritingAgentState)
    graph.add_node("load_session", load_session_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieve_template", retrieve_template_node)
    graph.add_node("generate_draft", generate_draft_node)
    graph.add_node("load_active_template", load_active_template_node)
    graph.add_node("revise_draft", revise_draft_node)
    graph.add_node("generate_wild", generate_wild_node)
    graph.add_node("observe", observe_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("revise_with_feedback", revise_with_feedback_node)
    graph.add_node("persist_state", persist_state_node)

    graph.set_entry_point("load_session")
    graph.add_edge("load_session", "planner")
    graph.add_conditional_edges(
        "planner",
        route_next_tool,
        {
            "retrieve_template": "retrieve_template",
            "load_active_template": "load_active_template",
            "generate_draft": "generate_draft",
            "revise_draft": "revise_draft",
            "generate_wild": "generate_wild",
            "finish": "finalize",
        },
    )
    graph.add_edge("retrieve_template", "observe")
    graph.add_edge("generate_draft", "observe")
    graph.add_edge("load_active_template", "observe")
    graph.add_edge("revise_draft", "observe")
    graph.add_edge("generate_wild", "observe")
    graph.add_edge("observe", "planner")
    graph.add_edge("finalize", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "revise_with_feedback": "revise_with_feedback",
            "persist_state": "persist_state",
        },
    )
    graph.add_edge("revise_with_feedback", "evaluator")
    graph.add_edge("persist_state", END)
    return graph.compile()


writing_graph = build_graph()


def run_agent(request: AgentRunRequest) -> dict:
    session_id = request.session_id or str(uuid4())
    initial_state: WritingAgentState = {
        "session_id": session_id,
        "run_id": str(uuid4()),
        "action": request.action,
        "intent": request.intent,
        "style": request.style,
        "language": request.language,
        "history": [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in request.history
        ],
        "current_draft": request.current_draft,
        "previous_draft": request.current_draft,
        "active_template_id": request.active_template_id,
        "steps": 0,
        "max_steps": 5,
        "retry_count": 0,
        "max_retries": 1,
        "observations": [],
        "trace": [
            {
                "node": "start",
                "action": request.action,
                "session_id": session_id,
            }
        ],
    }
    final_state = writing_graph.invoke(initial_state)
    template_meta = final_state.get("template_meta") or {}
    active_template_id = (
        template_meta.get("selected_template_id")
        or final_state.get("active_template_id")
    )
    return {
        "reply": final_state.get("reply", ""),
        "template_meta": template_meta,
        "evaluation": final_state.get("evaluation") or {},
        "state": {
            "session_id": session_id,
            "active_template_id": active_template_id,
        },
        "trace": final_state.get("trace") or [],
    }
