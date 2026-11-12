import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Home } from "./App";

jest.mock("./supabase", () => ({ apiFetch: (...args) => fetch(...args) }));

beforeEach(() => {
  window.localStorage.clear();
  window.localStorage.setItem("heywrite.session_id:test-user", "current");
});

function setup({ fail = false, missing = false } = {}) {
  let sessions = [
    { session_id: "current", title: "Current note", messages: [], message_count: 1 },
    { session_id: "older", title: "Older note", messages: [], message_count: 1 },
  ];
  global.fetch = jest.fn(async (url, options = {}) => {
    if (options.method === "DELETE") {
      if (fail) return { ok: false, status: 503 };
      sessions = sessions.filter((s) => !url.endsWith(`/${s.session_id}`));
      return { ok: !missing, status: missing ? 404 : 204 };
    }
    const data = url.endsWith("/agent/memory") ? {
      enabled: true, available: true,
      facts: sessions.map((s) => ({ key: s.session_id, value: `Memory from ${s.session_id}`, evidence: "Shared preference", source_session_id: s.session_id })),
    } : url.endsWith("/agent/sessions") ? { sessions } :
      url.endsWith("/agent/session/current") ? {
        session: { session_id: "current", current_draft: "Saved draft" },
        messages: [{ role: "user", content: "Current note" }],
      } : { templates: [] };
    return { ok: true, json: async () => data };
  });
  render(<Home userId="test-user" />);
}

test("each session can be deleted after confirmation without changing the current draft", async () => {
  setup();
  const item = await screen.findByTestId("session-item-older");
  await screen.findByText("Saved draft");
  fireEvent.click(within(item).getByRole("button", { name: "Delete session: Older note" }));
  expect(global.fetch.mock.calls.filter(([, options]) => options?.method === "DELETE")).toHaveLength(0);
  fireEvent.click(within(item).getByRole("button", { name: "Cancel" }));
  expect(screen.getByTestId("session-item-older")).toBeInTheDocument();
  fireEvent.click(within(item).getByRole("button", { name: "Delete session: Older note" }));
  fireEvent.click(within(item).getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(screen.queryByTestId("session-item-older")).not.toBeInTheDocument());
  expect(global.fetch).toHaveBeenCalledWith("/api/agent/session/older", { method: "DELETE" });
  expect(screen.getByText("Saved draft")).toBeInTheDocument();
  expect(window.localStorage.getItem("heywrite.session_id:test-user")).toBe("current");
});

test("deleting the current session clears its draft, history and saved session id", async () => {
  setup();
  await screen.findByText("Saved draft");
  const item = await screen.findByTestId("session-item-current");
  fireEvent.click(within(item).getByRole("button", { name: "Delete session: Current note" }));
  fireEvent.click(within(item).getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(screen.queryByTestId("session-item-current")).not.toBeInTheDocument());
  expect(screen.queryByText("Saved draft")).not.toBeInTheDocument();
  expect(window.localStorage.getItem("heywrite.session_id:test-user")).not.toBe("current");
  expect(screen.getByTestId("session-item-older")).toBeInTheDocument();
});

test("a failed deletion retains the conversation and offers another attempt", async () => {
  setup({ fail: true });
  await screen.findByText("Saved draft");
  const item = await screen.findByTestId("session-item-current");
  fireEvent.click(within(item).getByRole("button", { name: "Delete session: Current note" }));
  fireEvent.click(within(item).getByRole("button", { name: "Confirm delete" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not delete");
  expect(screen.getByText("Saved draft")).toBeInTheDocument();
  expect(window.localStorage.getItem("heywrite.session_id:test-user")).toBe("current");
  expect(within(item).getByRole("button", { name: "Confirm delete" })).toBeEnabled();
});

test("a conversation already deleted elsewhere is removed locally", async () => {
  setup({ missing: true });
  await screen.findByText("Saved draft");
  const item = await screen.findByTestId("session-item-older");
  fireEvent.click(within(item).getByRole("button", { name: "Delete session: Older note" }));
  fireEvent.click(within(item).getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(screen.queryByTestId("session-item-older")).not.toBeInTheDocument());
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

test("deleting another session refreshes its memories even when the current draft is unchanged", async () => {
  setup();
  await screen.findByText("Saved draft");
  fireEvent.click(screen.getByRole("button", { name: /Personal Memory/i }));
  expect(await screen.findByText("Memory from older")).toBeInTheDocument();
  const item = screen.getByTestId("session-item-older");
  fireEvent.click(within(item).getByRole("button", { name: "Delete session: Older note" }));
  expect(within(item).getByText(/personal memories saved from it/)).toBeInTheDocument();
  fireEvent.click(within(item).getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(screen.queryByText("Memory from older")).not.toBeInTheDocument());
  expect(screen.getByText("Memory from current")).toBeInTheDocument();
  expect(screen.getByText("Saved draft")).toBeInTheDocument();
});
