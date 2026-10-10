"""Optional Langfuse tracing for the writing graph."""

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)
_client = None


@lru_cache(maxsize=1)
def get_langfuse_client():
    global _client
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in {"false", "0", "no"}:
        return None
    if not all(os.getenv(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse tracing disabled: project keys are not configured")
        return None
    try:
        from langfuse import get_client

        _client = get_client()
        return _client
    except Exception:
        logger.warning("Langfuse initialization failed; continuing without tracing")
        return None


def agent_run_config(state: dict) -> dict:
    config = {
        "run_name": "heywrite-agent",
        "metadata": {
            "run_id": state["run_id"],
            "action": state["action"],
            "style": state["style"],
            "language": state["language"],
            "tenant_id": state.get("tenant_id"),
            "langfuse_user_id": state.get("tenant_id"),
            "langfuse_session_id": f"{state['tenant_id']}:{state['session_id']}" if state.get("tenant_id") else state["session_id"],
            "langfuse_tags": ["heywrite", state["action"]],
        },
    }
    if get_langfuse_client() is not None:
        try:
            from langfuse.langchain import CallbackHandler

            # One handler per invocation keeps concurrent runs independent.
            config["callbacks"] = [CallbackHandler()]
        except Exception:
            logger.warning("Langfuse callback setup failed; continuing without tracing")
    return config


def shutdown_observability() -> None:
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            logger.warning("Langfuse shutdown failed")
