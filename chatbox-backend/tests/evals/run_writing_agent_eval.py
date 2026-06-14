import json
from pathlib import Path

from fastapi.testclient import TestClient

import main


CASES_PATH = Path(__file__).with_name("writing_agent_cases.jsonl")


def load_cases():
    """Load JSONL eval cases.

    Each line is one independent scenario. The optional `description` field is
    documentation only; assertions are driven by the `expected` object.
    """
    with CASES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def trace_nodes(response: dict) -> list[str]:
    return [item.get("node", "") for item in response.get("trace", [])]


def evaluate_case(client: TestClient, case: dict) -> dict:
    # Use a deterministic session id per case so state persistence can be
    # tested without cases accidentally sharing a browser session.
    payload = {
        "session_id": f"eval-{case['id']}",
        "action": case["action"],
        "intent": case["intent"],
        "style": "Formal",
        "language": "English",
        "current_draft": case.get("current_draft", ""),
        "active_template_id": case.get("active_template_id"),
    }
    response = client.post("/agent/run", json=payload)
    result = {
        "id": case["id"],
        "status_code": response.status_code,
        "passed": response.status_code == 200,
        "failures": [],
    }
    if response.status_code != 200:
        result["failures"].append(response.text)
        return result

    data = response.json()
    nodes = trace_nodes(data)
    expected = case["expected"]

    # Trace checks verify the agent took the intended path through LangGraph.
    # This catches regressions such as bypassing retrieval, skipping evaluation,
    # or failing to persist the run.
    for node in expected.get("must_have_trace_nodes", []):
        if node not in nodes:
            result["passed"] = False
            result["failures"].append(f"missing trace node: {node}")

    # Continue Editing and Wild Mode should not retrieve a new template. This
    # protects the product behavior where editing continues from the active
    # draft instead of unexpectedly starting a new task.
    if expected.get("must_not_retrieve_template") and "retrieve_template" in nodes:
        result["passed"] = False
        result["failures"].append("unexpected template retrieval")

    # Template hint checks validate retrieval quality at a simple behavioral
    # level. The exact score may vary, but a meeting request should not select
    # a project report template.
    template_hint = expected.get("template_hint")
    selected_template = (data.get("template_meta") or {}).get("selected_template", "")
    if template_hint and template_hint.lower() not in selected_template.lower():
        result["passed"] = False
        result["failures"].append(f"template hint not matched: {selected_template}")

    # Evaluator checks are the quality gate. They ensure the final answer is not
    # only generated, but also follows the user request, uses the right template
    # behavior, includes Changes, and avoids obvious unrelated fabrication.
    evaluation = data.get("evaluation") or {}
    checks = evaluation.get("checks") or {}
    required_checks = [
        "preserves_original_structure",
        "completed_user_request",
        "used_correct_template",
        "contains_changes",
        "no_unrelated_fabrication",
    ]
    for check_name in required_checks:
        if checks.get(check_name) is not True:
            result["passed"] = False
            result["failures"].append(f"evaluator check failed: {check_name}")
    return result


def main_eval():
    client = TestClient(main.app)
    results = [evaluate_case(client, case) for case in load_cases()]
    print(json.dumps(results, indent=2))
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main_eval()
