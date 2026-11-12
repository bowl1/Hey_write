import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from tenancy import AuthenticatedUser


class SessionDeleteAPITests(unittest.TestCase):
    def test_deletion_requires_authentication(self):
        with TestClient(app) as client, patch("main.delete_session") as delete:
            self.assertEqual(client.delete("/agent/session/private").status_code, 401)
            delete.assert_not_called()

    def test_delete_response_missing_session_and_database_failure(self):
        async def auth(_):
            return AuthenticatedUser(id="00000000-0000-0000-0000-000000000001")
        with patch("main.authenticate", side_effect=auth), patch("main.ensure_workspace"), \
             patch("main.delete_session") as delete, TestClient(app) as client:
            delete.return_value = True
            response = client.delete("/agent/session/example")
            self.assertEqual(response.status_code, 204)
            self.assertEqual(response.content, b"")
            delete.assert_called_with("example")
            delete.return_value = False
            self.assertEqual(client.delete("/agent/session/missing").status_code, 404)
            delete.side_effect = RuntimeError("private database details")
            response = client.delete("/agent/session/example")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private database details", response.text)
