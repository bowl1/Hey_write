import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from template_store import get_template, list_public_templates


class PublicTemplatesTests(unittest.TestCase):
    def test_guests_read_only_bundled_templates_without_auth_or_database(self):
        with TestClient(app) as client, patch("main.authenticate") as auth, patch("main.ensure_workspace") as provision:
            response = client.get("/public/templates")
        self.assertEqual(response.status_code, 200)
        templates = response.json()["templates"]
        self.assertEqual(len(templates), 15)
        self.assertTrue(all(template["id"].startswith("default:") for template in templates))
        self.assertTrue(all(template["source"] == "default" for template in templates))
        self.assertTrue(all(template["content"] for template in templates))
        auth.assert_not_called()
        provision.assert_not_called()

    def test_default_template_can_be_loaded_for_continued_editing(self):
        template = list_public_templates()[0]
        with patch("template_store.get_connection") as connect:
            self.assertEqual(get_template(template["id"])["content"], template["content"])
            self.assertIsNone(get_template("default:../../private"))
        connect.assert_not_called()

    def test_public_route_does_not_enable_writes_or_private_reads(self):
        with TestClient(app) as client:
            self.assertEqual(client.post("/public/templates", json={}).status_code, 401)
            self.assertEqual(client.get("/templates").status_code, 401)
            self.assertEqual(client.get("/agent/sessions").status_code, 401)


if __name__ == "__main__":
    unittest.main()
