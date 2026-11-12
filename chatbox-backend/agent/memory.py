"""Bounded conversation context shared by generation and evaluation."""

import json


def conversation_context(state: dict) -> str:
    messages = []
    remaining = 12000
    for message in reversed((state.get("history") or [])[-20:]):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "")[:4000]
        content = content[:remaining]
        if content:
            messages.append({"role": role, "content": content})
            remaining -= len(content)
        if remaining <= 0:
            break
    # Facts and summaries are contextual data, never system-role messages.
    facts = [{"key": fact["key"], "value": str(fact["value"])[:300]}
             for fact in (state.get("long_term_memory") or [])[-20:]]
    return json.dumps({
        "long_term_memory": facts,
        "session_summary": (state.get("session_summary") or "")[:4000],
        "recent_messages": list(reversed(messages)),
    }, ensure_ascii=False)
