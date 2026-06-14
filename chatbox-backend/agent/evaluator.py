from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_runner.rag_chain import get_llm


EVALUATOR_PROMPT = PromptTemplate.from_template(
    """
You are an evaluator for a template-grounded writing agent.

Return only valid JSON. Do not include markdown.

Evaluate whether the final draft satisfies the user's request.

Checks:
- preserves_original_structure: true if a revision keeps the previous draft's section structure and does not rewrite unrelated parts.
- completed_user_request: true if the final draft actually follows the user intent.
- used_correct_template: true if the template behavior is correct for the selected action.
- contains_changes: true if the response contains a clear Changes section.
- no_unrelated_fabrication: true if the draft does not invent unrelated details.

Action semantics:
- new_task should use a selected template when a template is available.
- continue_editing should revise the current draft and should not select a new template.
- wild should not use a template.

User intent:
{intent}

Action:
{action}

Previous draft:
{previous_draft}

Selected template:
{template_summary}

Final response:
{reply}

Return JSON with this exact shape:
{{
  "checks": {{
    "preserves_original_structure": true,
    "completed_user_request": true,
    "used_correct_template": true,
    "contains_changes": true,
    "no_unrelated_fabrication": true
  }},
  "reason": "one concise sentence"
}}
"""
)

evaluator_chain = None


def _draft_body(reply: str) -> str:
    match = re.search(r"(?:^|\n)Changes:\s*", reply or "", flags=re.IGNORECASE)
    if not match:
        return (reply or "").strip()
    return reply[: match.start()].strip()


def _trace_nodes(state: dict[str, Any]) -> list[str]:
    return [item.get("node", "") for item in state.get("trace") or []]


def _required_terms(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d+", text or "")
    terms = {
        token.lower()
        for token in tokens
        if token.isdigit() or (len(token) > 1 and token[0].isupper())
    }
    for match in re.finditer(
        r"\b(?:name|named|called)\s+([A-Za-z][A-Za-z0-9_-]{1,})\b",
        text or "",
        flags=re.IGNORECASE,
    ):
        terms.add(match.group(1).lower())
    for match in re.finditer(
        r"\b(?:add|include|mention)\s+([A-Za-z][A-Za-z0-9_-]{1,})\b",
        text or "",
        flags=re.IGNORECASE,
    ):
        candidate = match.group(1).lower()
        if candidate not in {"name", "date", "section", "paragraph", "sentence"}:
            terms.add(candidate)
    return terms


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def get_evaluator_chain():
    global evaluator_chain
    if evaluator_chain is None:
        evaluator_chain = LLMChain(llm=get_llm(), prompt=EVALUATOR_PROMPT)
    return evaluator_chain


def deterministic_evaluate(state: dict[str, Any]) -> dict[str, Any]:
    reply = state.get("reply") or ""
    draft = _draft_body(reply)
    previous = state.get("previous_draft") or ""
    template_meta = state.get("template_meta") or {}
    action = state.get("action")
    nodes = _trace_nodes(state)

    contains_changes = bool(re.search(r"(?:^|\n)Changes:\s*", reply, flags=re.IGNORECASE))

    if previous.strip() and action == "continue_editing":
        previous_headings = re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9 ._-]{1,60}:)\s*$", previous)
        draft_headings = re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9 ._-]{1,60}:)\s*$", draft)
        preserves_original_structure = (
            previous_headings == draft_headings
            if previous_headings
            else abs(len(draft.splitlines()) - len(previous.splitlines())) <= max(2, len(previous.splitlines()))
        )
    else:
        preserves_original_structure = True

    required_terms = _required_terms(state.get("intent") or "")
    response_lower = f"{draft}\n{reply}".lower()
    missing_required_terms = sorted(
        term for term in required_terms if term not in response_lower
    )
    completed_user_request = bool(draft.strip()) and not missing_required_terms

    if action == "wild":
        used_correct_template = not template_meta.get("used_template") and "retrieve_template" not in nodes
    elif action == "continue_editing":
        used_correct_template = "retrieve_template" not in nodes and (
            not state.get("active_template_id")
            or template_meta.get("selected_template_id") in {state.get("active_template_id"), "", None}
        )
    else:
        used_correct_template = bool(template_meta.get("used_template")) and bool(
            template_meta.get("selected_template_id")
        )

    unrelated_markers = ["as an ai", "lorem ipsum", "i cannot browse", "fictional details"]
    no_unrelated_fabrication = bool(draft.strip()) and not any(
        marker in draft.lower() for marker in unrelated_markers
    )
    if previous.strip() and action == "continue_editing":
        no_unrelated_fabrication = no_unrelated_fabrication and len(draft) <= max(
            600,
            len(previous) * 3,
        )

    checks = {
        "preserves_original_structure": preserves_original_structure,
        "completed_user_request": completed_user_request,
        "used_correct_template": used_correct_template,
        "contains_changes": contains_changes,
        "no_unrelated_fabrication": no_unrelated_fabrication,
    }
    return {
        "checks": checks,
        "hard_failures": [
            name
            for name, passed in checks.items()
            if not passed
        ],
        "missing_required_terms": missing_required_terms,
        "reason": "Deterministic evaluator checked hard constraints.",
    }


def llm_evaluate(state: dict[str, Any]) -> dict[str, Any] | None:
    if os.getenv("AGENT_LLM_EVALUATOR_ENABLED", "true").lower() == "false":
        return None

    template_meta = state.get("template_meta") or {}
    template_summary = "No template selected."
    if template_meta.get("used_template"):
        template_summary = (
            f"{template_meta.get('selected_template', '')} "
            f"({template_meta.get('selected_template_id', '')})"
        ).strip()

    try:
        result = get_evaluator_chain().run(
            {
                "intent": state.get("intent", ""),
                "action": state.get("action", ""),
                "previous_draft": state.get("previous_draft", ""),
                "template_summary": template_summary,
                "reply": state.get("reply", ""),
            }
        )
        parsed = _extract_json(result)
    except Exception as exc:
        return {
            "error": str(exc),
            "checks": {},
            "reason": "LLM evaluator unavailable; used deterministic fallback.",
        }

    checks = parsed.get("checks") if isinstance(parsed, dict) else {}
    if not isinstance(checks, dict):
        return None
    normalized_checks = {
        "preserves_original_structure": bool(checks.get("preserves_original_structure")),
        "completed_user_request": bool(checks.get("completed_user_request")),
        "used_correct_template": bool(checks.get("used_correct_template")),
        "contains_changes": bool(checks.get("contains_changes")),
        "no_unrelated_fabrication": bool(checks.get("no_unrelated_fabrication")),
    }
    return {
        "checks": normalized_checks,
        "reason": str(parsed.get("reason") or "LLM evaluator completed."),
    }


def evaluate_agent_output(state: dict[str, Any]) -> dict[str, Any]:
    deterministic = deterministic_evaluate(state)
    llm_result = llm_evaluate(state)
    deterministic_checks = deterministic["checks"]
    checks = dict(deterministic_checks)
    source = "deterministic"
    reason = deterministic["reason"]
    llm_error = None

    if llm_result and not llm_result.get("error") and llm_result.get("checks"):
        source = "llm_with_deterministic_guards"
        reason = llm_result.get("reason") or reason
        llm_checks = llm_result["checks"]
        checks["preserves_original_structure"] = (
            deterministic_checks["preserves_original_structure"]
            and llm_checks["preserves_original_structure"]
        )
        checks["completed_user_request"] = (
            deterministic_checks["completed_user_request"]
            and llm_checks["completed_user_request"]
        )
        checks["used_correct_template"] = deterministic_checks["used_correct_template"]
        checks["contains_changes"] = deterministic_checks["contains_changes"]
        checks["no_unrelated_fabrication"] = (
            deterministic_checks["no_unrelated_fabrication"]
            and llm_checks["no_unrelated_fabrication"]
        )
    elif llm_result and llm_result.get("error"):
        llm_error = llm_result["error"]

    issues = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not issues,
        "checks": checks,
        "issues": issues,
        "reason": reason,
        "source": source,
        "missing_required_terms": deterministic.get("missing_required_terms", []),
        "llm_error": llm_error,
    }
