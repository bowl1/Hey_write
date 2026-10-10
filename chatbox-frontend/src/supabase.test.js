jest.mock("@supabase/supabase-js", () => ({ createClient: jest.fn() }));

test("API requests use the current refreshed access token", async () => {
  process.env.REACT_APP_SUPABASE_URL = "https://example.supabase.co";
  process.env.REACT_APP_SUPABASE_PUBLISHABLE_KEY = "public-test-key";
  jest.resetModules();
  const { createClient } = require("@supabase/supabase-js");
  const getSession = jest.fn().mockResolvedValue({ data: { session: { access_token: "refreshed-token" } } });
  createClient.mockReturnValue({ auth: { getSession } });
  const { apiFetch } = require("./supabase");
  global.fetch = jest.fn().mockResolvedValue({ ok: true });
  await apiFetch("/agent/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  const options = global.fetch.mock.calls[0][1];
  expect(options.headers.get("Authorization")).toBe("Bearer refreshed-token");
  expect(options.headers.get("Content-Type")).toBe("application/json");
  getSession.mockResolvedValue({ data: { session: null } });
  await expect(apiFetch("/templates")).rejects.toThrow("Please sign in again.");
  expect(global.fetch).toHaveBeenCalledTimes(1);
});
