import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from tenancy import AuthenticatedUser


class MemoryAPITests(unittest.TestCase):
    def test_memory_read_and_mutations_require_authentication(self):
        with TestClient(app) as client:
            for method in ("get", "patch", "delete"):
                self.assertEqual(getattr(client, method)("/agent/memory").status_code, 401)

    def test_settings_clear_and_response_contract(self):
        async def auth(_):
            return AuthenticatedUser(id="00000000-0000-0000-0000-000000000001")
        profile = {"enabled": True, "facts": [], "revision": 4}
        with patch("main.authenticate", side_effect=auth), patch("main.ensure_workspace"), \
             patch("main.load_profile", return_value=profile), \
             patch("main.configure_profile", return_value=profile) as configure, \
             patch.dict("os.environ", {"AGENT_MEMORY_ENABLED": "true"}), TestClient(app) as client:
            self.assertEqual(client.get("/agent/memory").json(), {"enabled": True, "facts": [], "available": True})
            self.assertEqual(client.patch("/agent/memory", json={"enabled": False}).status_code, 200)
            configure.assert_called_with(enabled=False)
            self.assertEqual(client.delete("/agent/memory").status_code, 200)
            configure.assert_called_with(clear=True)
            self.assertEqual(client.patch("/agent/memory", json={"enabled": "false"}).status_code, 422)
            self.assertEqual(client.patch("/agent/memory", json={"enabled": True, "tenant_id": "forged"}).status_code, 422)
