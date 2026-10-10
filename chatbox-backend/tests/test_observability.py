import os
import unittest
from unittest.mock import Mock, patch

from langchain.chains import LLMChain
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import observability
from agent.graph import run_agent
from agent.schemas import AgentRunRequest
from tenancy import tenant_context


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        observability.get_langfuse_client.cache_clear()
        self.tenant = tenant_context("00000000-0000-0000-0000-000000000001")
        self.tenant.__enter__()

    def tearDown(self):
        self.tenant.__exit__(None, None, None)
        observability._client = None
        observability.get_langfuse_client.cache_clear()

    def test_missing_keys_and_explicit_disable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(observability.get_langfuse_client())
        observability.get_langfuse_client.cache_clear()
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
            "LANGFUSE_TRACING_ENABLED": "false",
        }, clear=True):
            self.assertIsNone(observability.get_langfuse_client())

    def test_graph_and_nested_llm_tracing(self):
        exporter = InMemorySpanExporter()
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-offline-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-offline-test",
            "LANGFUSE_TRACING_ENABLED": "true",
        }):
            client = Langfuse(span_exporter=exporter)
            chain = LLMChain(
                llm=FakeListLLM(responses=["Hello team.\n\nChanges:\n- Added greeting."]),
                prompt=PromptTemplate.from_template("{intent} {style} {language} {context} {previous}"),
            )
            try:
                with patch("agent.nodes.load_session", return_value=None), \
                     patch("agent.nodes.persist_agent_state", return_value={"draft_accepted": True}), \
                     patch("agent.nodes.get_llm_chain", return_value=chain), \
                     patch.dict(os.environ, {"AGENT_LLM_EVALUATOR_ENABLED": "false"}):
                    result = run_agent(AgentRunRequest(
                        session_id="test-langfuse-session", action="wild", intent="write a greeting"
                    ))
                client.flush()
                spans = exporter.get_finished_spans()
                names = {span.name for span in spans}
                self.assertIn("heywrite-agent", names)
                self.assertIn("generate_wild", names)
                self.assertIn("evaluator", names)
                self.assertIn("FakeListLLM", names)
                root = next(span for span in spans if span.name == "heywrite-agent")
                self.assertTrue(all(span.context.trace_id == root.context.trace_id for span in spans))
                self.assertTrue(any("test-langfuse-session" in str(dict(span.attributes)) for span in spans))
                self.assertTrue(result["evaluation"]["passed"])
            finally:
                client.shutdown()

    def test_callback_failure_keeps_agent_config_usable(self):
        state = {"run_id": "test-run", "session_id": "test-session", "action": "wild",
                 "style": "Formal", "language": "English"}
        with patch("observability.get_langfuse_client", return_value=Mock()), \
             patch("langfuse.langchain.CallbackHandler", side_effect=RuntimeError("unavailable")):
            config = observability.agent_run_config(state)
        self.assertNotIn("callbacks", config)
        self.assertEqual(config["metadata"]["run_id"], "test-run")

    def test_shutdown_drains_client(self):
        client = Mock()
        with patch.object(observability, "_client", client):
            observability.shutdown_observability()
        client.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
