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
    return json.dumps(list(reversed(messages)), ensure_ascii=False)
