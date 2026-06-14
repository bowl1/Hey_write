from __future__ import annotations

from langchain_runner.rag_chain import get_llm_chain
from template_store import get_template, search_templates_with_metadata

from .evaluator import evaluate_agent_output
from .state import WritingAgentState
from .store import load_session, persist_agent_state


def append_trace(state: WritingAgentState, node: str, **data) -> list[dict]:
    return [*(state.get("trace") or []), {"node": node, **data}]


def load_session_node(state: WritingAgentState) -> WritingAgentState:
    session = load_session(state["session_id"])
    updates: WritingAgentState = {
        "trace": append_trace(
            state,
            "load_session",
            found=bool(session),
            session_id=state["session_id"],
        )
    }
    if session:
        updates["active_template_id"] = state.get("active_template_id") or session.get("active_template_id")
        updates["current_draft"] = state.get("current_draft") or session.get("current_draft") or ""
    return updates


def planner_node(state: WritingAgentState) -> WritingAgentState:
    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", 5)
    action = state.get("action")
    next_tool = "finish"

    if state.get("reply") or steps >= max_steps:
        next_tool = "finish"
    elif action == "new_task":
        next_tool = "generate_draft" if state.get("retrieved_templates") else "retrieve_template"
    elif action == "continue_editing":
        if state.get("active_template_id") and not state.get("active_template"):
            next_tool = "load_active_template"
        else:
            next_tool = "revise_draft"
    elif action == "wild":
        next_tool = "generate_wild"

    return {
        "next_tool": next_tool,
        "trace": append_trace(
            state,
            "planner",
            decision=next_tool,
            step=steps,
            max_steps=max_steps,
        ),
    }


def route_next_tool(state: WritingAgentState) -> str:
    return state.get("next_tool") or "finish"


def retrieve_template_node(state: WritingAgentState) -> WritingAgentState:
    results = search_templates_with_metadata(state.get("intent", ""), limit=3)
    trace_data = [
        {
            "template_id": result["template"].get("id"),
            "title": result["template"].get("title"),
            "distance": result["distance"],
            "vector_distance": result["vector_distance"],
            "bm25_score": result["bm25_score"],
            "final_score": result["final_score"],
            "retrieval_source": result.get("retrieval_source"),
            "matched_terms": result["matched_terms"],
        }
        for result in results
    ]
    updates: WritingAgentState = {
        "retrieved_templates": results,
        "trace": append_trace(state, "retrieve_template", results=trace_data),
    }
    if results:
        updates["active_template"] = results[0]["template"]
        updates["active_template_id"] = results[0]["template"].get("id")
    return updates


def generate_draft_node(state: WritingAgentState) -> WritingAgentState:
    retrieved = state.get("retrieved_templates") or []
    if not retrieved:
        return {
            "reply": "did not find any template matched, please try the wild mode",
            "template_meta": {
                "used_template": False,
                "reason": "No templates were returned by retrieval.",
            },
            "trace": append_trace(state, "generate_draft", status="no_template"),
        }

    top = retrieved[0]
    templates = [result["template"] for result in retrieved]
    context = "\n\n".join(template.get("content", "") for template in templates)
    result = get_llm_chain().run(
        {
            "intent": state.get("intent", ""),
            "style": state.get("style") or "Formal",
            "language": state.get("language") or "English",
            "context": context,
            "previous": "",
        }
    )
    template = top["template"]
    return {
        "reply": result.strip(),
        "template_meta": {
            "used_template": True,
            "selected_template": template.get("title", ""),
            "selected_template_id": template.get("id", ""),
            "match_score": top["distance"],
            "vector_distance": top["vector_distance"],
            "bm25_score": top["bm25_score"],
            "final_score": top["final_score"],
            "retrieval_source": top.get("retrieval_source"),
            "matched_terms": top["matched_terms"],
            "reason": (
                "Selected by hybrid retrieval from merged pgvector semantic "
                "and BM25 keyword candidates."
            ),
        },
        "trace": append_trace(
            state,
            "generate_draft",
            status="ok",
            selected_template_id=template.get("id"),
            retrieval_source=top.get("retrieval_source"),
        ),
    }


def load_active_template_node(state: WritingAgentState) -> WritingAgentState:
    active_template_id = state.get("active_template_id")
    active_template = get_template(active_template_id) if active_template_id else None
    return {
        "active_template": active_template,
        "trace": append_trace(
            state,
            "load_active_template",
            found=bool(active_template),
            template_id=active_template_id,
        ),
    }


def revise_draft_node(state: WritingAgentState) -> WritingAgentState:
    current_draft = state.get("current_draft") or state.get("previous_draft") or ""
    if not current_draft.strip():
        return {
            "reply": "please generate a draft before continuing editing",
            "template_meta": {
                "used_template": False,
                "reason": "Continue Editing requires an existing draft.",
            },
            "trace": append_trace(state, "revise_draft", status="missing_current_draft"),
        }

    active_template = state.get("active_template")
    result = get_llm_chain().run(
        {
            "intent": state.get("intent", ""),
            "style": state.get("style") or "Formal",
            "language": state.get("language") or "English",
            "context": active_template.get("content", "") if active_template else "",
            "previous": current_draft,
        }
    )
    return {
        "reply": result.strip(),
        "template_meta": {
            "used_template": bool(active_template),
            "selected_template": active_template.get("title", "") if active_template else "",
            "selected_template_id": active_template.get("id", "") if active_template else "",
            "reason": (
                "Continued editing the current draft using the active template."
                if active_template
                else "Continued editing the current draft without reselecting a template."
            ),
        },
        "trace": append_trace(
            state,
            "revise_draft",
            status="ok",
            used_active_template=bool(active_template),
        ),
    }


def generate_wild_node(state: WritingAgentState) -> WritingAgentState:
    result = get_llm_chain().run(
        {
            "intent": state.get("intent", ""),
            "style": state.get("style") or "Formal",
            "language": state.get("language") or "English",
            "context": "",
            "previous": state.get("current_draft") or "",
        }
    )
    return {
        "reply": result.strip(),
        "template_meta": {
            "used_template": False,
            "reason": "Generated without template retrieval because the user selected wild mode.",
        },
        "active_template_id": None,
        "trace": append_trace(state, "generate_wild", status="ok"),
    }


def observe_node(state: WritingAgentState) -> WritingAgentState:
    last_tool = state.get("next_tool")
    observations = [
        *(state.get("observations") or []),
        {
            "tool": last_tool,
            "has_reply": bool(state.get("reply")),
            "has_template": bool(state.get("active_template_id")),
        },
    ]
    return {
        "steps": state.get("steps", 0) + 1,
        "last_tool": last_tool,
        "observations": observations,
        "trace": append_trace(
            state,
            "observe",
            tool=last_tool,
            has_reply=bool(state.get("reply")),
        ),
    }


def finalize_node(state: WritingAgentState) -> WritingAgentState:
    if state.get("reply"):
        return {
            "trace": append_trace(
                state,
                "finalize",
                status="ok",
                steps=state.get("steps", 0),
            )
        }

    return {
        "reply": "I could not complete the request within the agent step limit.",
        "template_meta": {
            "used_template": False,
            "reason": "Agent stopped before producing a final draft.",
        },
        "trace": append_trace(
            state,
            "finalize",
            status="step_limit_or_no_reply",
            steps=state.get("steps", 0),
        ),
    }


def evaluator_node(state: WritingAgentState) -> WritingAgentState:
    evaluation = evaluate_agent_output(state)
    feedback = ""
    if not evaluation["passed"]:
        feedback = (
            "Evaluator failed these checks: "
            + ", ".join(evaluation["issues"])
            + ". "
            + evaluation.get("reason", "")
        )
    return {
        "evaluation": evaluation,
        "evaluation_feedback": feedback,
        "trace": append_trace(
            state,
            "evaluator",
            status="passed" if evaluation["passed"] else "failed",
            failed_checks=evaluation["issues"],
            source=evaluation.get("source"),
        ),
    }


def route_after_evaluation(state: WritingAgentState) -> str:
    evaluation = state.get("evaluation") or {}
    if (
        not evaluation.get("passed", False)
        and state.get("reply")
        and state.get("retry_count", 0) < state.get("max_retries", 1)
    ):
        return "revise_with_feedback"
    return "persist_state"


def revise_with_feedback_node(state: WritingAgentState) -> WritingAgentState:
    feedback = state.get("evaluation_feedback") or "Fix the evaluator failures."
    active_template = state.get("active_template")
    context = active_template.get("content", "") if active_template else ""
    result = get_llm_chain().run(
        {
            "intent": (
                f"{state.get('intent', '')}\n\n"
                f"Evaluator feedback: {feedback}\n"
                "Revise the current draft only enough to satisfy the evaluator. "
                "Keep the same template behavior and include a Changes section."
            ),
            "style": state.get("style") or "Formal",
            "language": state.get("language") or "English",
            "context": context,
            "previous": state.get("reply") or state.get("current_draft") or "",
        }
    )
    return {
        "reply": result.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": append_trace(
            state,
            "revise_with_feedback",
            status="ok",
            feedback=feedback,
        ),
    }


def persist_state_node(state: WritingAgentState) -> WritingAgentState:
    persist_agent_state(state)
    return {
        "trace": append_trace(
            state,
            "persist_state",
            status="ok",
            session_id=state.get("session_id"),
        )
    }
