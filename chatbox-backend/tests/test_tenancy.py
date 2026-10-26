import asyncio
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from tenancy import AuthenticatedUser, authenticate, tenant_context, tenant_id, tenant_schema

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"


class TenancyTests(unittest.TestCase):
    def test_context_is_required_and_resets(self):
        with self.assertRaises(RuntimeError):
            tenant_id()
        with tenant_context(A):
            self.assertEqual(tenant_id(), A)
            with tenant_context(B):
                self.assertEqual(tenant_id(), B)
            self.assertEqual(tenant_id(), A)
        with self.assertRaises(RuntimeError):
            tenant_id()
        with self.assertRaises(ValueError):
            tenant_schema('public; DROP TABLE messages')

    def test_rejects_unauthenticated_api_access(self):
        with TestClient(app) as client, patch("main.ensure_workspace") as provision:
            for path in ("/workspace", "/templates", "/agent/sessions", "/agent/session/known-id"):
                self.assertEqual(client.get(path).status_code, 401)
            self.assertEqual(client.post("/agent/run", json={"action": "wild", "intent": "test"}).status_code, 401)
            self.assertEqual(client.post("/templates", json={"title": "test", "content": "test"}).status_code, 401)
            self.assertEqual(client.get("/health").status_code, 200)
            provision.assert_not_called()

    def test_supabase_validates_user_instead_of_client_tenant(self):
        seen = []
        def respond(request):
            seen.append(request)
            return httpx.Response(200, json={"id": A, "email": "a@example.com", "email_confirmed_at": "2026-01-01"})
        auth_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.dict(os.environ, {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "public-test-key"}), \
             patch("tenancy.httpx.AsyncClient", return_value=auth_client):
            user = asyncio.run(authenticate("Bearer access-token"))
        self.assertEqual(str(user.id), A)
        self.assertEqual(str(seen[0].url), "https://example.supabase.co/auth/v1/user")
        self.assertEqual(seen[0].headers["authorization"], "Bearer access-token")

    def test_expired_token_and_auth_service_errors(self):
        for status, expected in ((401, 401), (400, 401), (500, 503)):
            auth_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(status)))
            with patch.dict(os.environ, {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_PUBLISHABLE_KEY": "public-test-key"}), \
                 patch("tenancy.httpx.AsyncClient", return_value=auth_client):
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(authenticate("Bearer expired-token"))
                self.assertEqual(caught.exception.status_code, expected)

    def test_simultaneous_requests_keep_account_context(self):
        async def verify(header):
            return AuthenticatedUser(id=A if header == "Bearer a" else B)
        def send(token):
            with TestClient(app) as client:
                return client.get("/workspace", headers={"Authorization": f"Bearer {token}", "X-Tenant-ID": B if token == "a" else A}).json()
        with patch("main.authenticate", side_effect=verify), patch("main.ensure_workspace"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(send, ["a", "b"]))
        self.assertEqual(results, [{"tenant_id": A}, {"tenant_id": B}])


if __name__ == "__main__":
    unittest.main()
