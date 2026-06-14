import { render, screen, fireEvent } from "@testing-library/react";
import Home from "./App";

test("does not submit when input is empty", () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ templates: [] }),
  });

  render(<Home />);
  const generateButton = screen.getByText("Generate with Template");
  fireEvent.click(generateButton);

  expect(global.fetch).not.toHaveBeenCalledWith(
    expect.stringContaining("/agent/run"),
    expect.anything()
  );
});

test("shows response after clicking generate button", async () => {
  global.fetch = jest.fn((url) => {
    const requestUrl = String(url);
    return Promise.resolve({
      ok: true,
      json: async () =>
        requestUrl.includes("/templates")
          ? { templates: [] }
          : {
              reply: "This is a test reply\n\nChanges:\n- Added a clearer opening",
              template_meta: { used_template: false },
              evaluation: {
                passed: true,
                checks: {
                  preserves_original_structure: true,
                  completed_user_request: true,
                  used_correct_template: true,
                  contains_changes: true,
                  no_unrelated_fabrication: true,
                },
                issues: [],
              },
              state: { session_id: "session-1", active_template_id: null },
              trace: [{ node: "start", action: "new_task" }],
            },
    });
  });

  render(<Home />);

  const textarea = screen.getByPlaceholderText(/summarize today's product sync/i);
  fireEvent.change(textarea, {
    target: { value: "Write an invitation email" },
  });

  const button = screen.getByText(/generate with template/i);
  fireEvent.click(button);

  const replyElement = await screen.findByTestId("main-reply");
  expect(replyElement).toHaveTextContent("This is a test reply");
  expect(replyElement).not.toHaveTextContent("Added a clearer opening");
  expect(await screen.findByTestId("changes-summary")).toHaveTextContent(
    "Added a clearer opening"
  );
  expect(await screen.findByTestId("agent-trace")).toHaveTextContent("start");
  expect(await screen.findByTestId("agent-evaluation")).toHaveTextContent(
    "Evaluator passed"
  );
});

test("continues editing the current draft without retrieving a new template", async () => {
  global.fetch = jest.fn((url, options = {}) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/templates")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ templates: [] }),
      });
    }
    if (requestUrl.includes("/agent/run")) {
      const body = JSON.parse(options.body || "{}");
      if (body.action === "new_task") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            reply: "Dear team,\nPlease join the meeting.",
            template_meta: {
              used_template: true,
              selected_template_id: "meeting_invitation",
              selected_template: "Meeting Invitation",
            },
            state: {
              session_id: "session-1",
              active_template_id: "meeting_invitation",
            },
            trace: [{ node: "retrieve_template", results: [] }],
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          reply: "Dear Anna,\nPlease join the meeting.",
          template_meta: {
            used_template: true,
            selected_template_id: "meeting_invitation",
          },
          state: {
            session_id: "session-1",
            active_template_id: "meeting_invitation",
          },
          trace: [{ node: "revise_draft", used_active_template: true }],
        }),
      });
    }
    return Promise.reject(new Error(`Unexpected URL: ${requestUrl}`));
  });

  render(<Home />);

  const textarea = screen.getByPlaceholderText(/summarize today's product sync/i);
  fireEvent.change(textarea, {
    target: { value: "Write an invitation email" },
  });
  fireEvent.click(screen.getByText(/generate with template/i));
  expect(await screen.findByTestId("main-reply")).toHaveTextContent(
    "Dear team"
  );

  fireEvent.change(textarea, {
    target: { value: "add name Anna" },
  });
  fireEvent.click(screen.getByText(/continue editing/i));

  expect(await screen.findByTestId("main-reply")).toHaveTextContent(
    "Dear Anna"
  );
  const lastCallOptions =
    global.fetch.mock.calls[global.fetch.mock.calls.length - 1][1];
  expect(global.fetch).toHaveBeenLastCalledWith(
    expect.stringContaining("/agent/run"),
    expect.anything()
  );
  expect(lastCallOptions.body).toContain('"action":"continue_editing"');
  expect(lastCallOptions.body).toContain(
    '"active_template_id":"meeting_invitation"'
  );
});
