import { render, screen, fireEvent } from "@testing-library/react";
import Home from "./App";

beforeEach(() => {
  window.localStorage.clear();
  jest.clearAllMocks();
  Object.assign(navigator, {
    clipboard: {
      writeText: jest.fn(),
    },
  });
});

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
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ sessions: [] }),
      });
    }
    return Promise.resolve({
      ok: true,
      json: async () =>
        requestUrl.includes("/templates")
          ? { templates: [] }
          : {
              reply:
                "**This** is a *test* reply\n\nSecond paragraph stays separate.\nStill same paragraph.\n\n| Owner | Task |\n| --- | --- |\n| Anna | Send sample data |\n\nClosing paragraph.\n\nChanges:\n* Added a clearer opening",
              template_meta: {
                used_template: true,
                selected_template: "Meeting Invitation",
                match_score: 0.82,
              },
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
  expect(replyElement.querySelectorAll("p")).toHaveLength(3);
  expect(replyElement.querySelector("table")).toBeInTheDocument();
  expect(replyElement).toHaveTextContent("Owner");
  expect(replyElement).toHaveTextContent("Anna");
  expect(replyElement).not.toHaveTextContent("**");
  expect(replyElement).not.toHaveTextContent("*test*");
  expect(replyElement).not.toHaveTextContent("| Owner |");
  expect(replyElement).not.toHaveTextContent("Added a clearer opening");
  expect(await screen.findByTestId("changes-summary")).toHaveTextContent(
    "Added a clearer opening"
  );
  expect(screen.getByText(/match score 82%/i)).toBeInTheDocument();
  fireEvent.click(screen.getByText("Copy"));
  expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
    "This is a test reply\n\nSecond paragraph stays separate.\nStill same paragraph.\n\nOwner\tTask\nAnna\tSend sample data\n\nClosing paragraph."
  );
  expect(await screen.findByTestId("agent-trace")).toHaveTextContent("start");
  expect(await screen.findByTestId("agent-evaluation")).toHaveTextContent(
    "Evaluator passed"
  );
  expect(window.localStorage.getItem("heywrite.session_id")).toBe("session-1");
});

test("does not show a draft when no template match clears the threshold", async () => {
  global.fetch = jest.fn((url) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ sessions: [] }),
      });
    }
    if (requestUrl.includes("/templates")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ templates: [] }),
      });
    }
    return Promise.resolve({
      ok: true,
      json: async () => ({
        reply: "",
        template_meta: {
          used_template: false,
          requires_template: true,
          selected_template: "Meeting Invitation",
          match_score: 0.64,
          min_match_score: 0.65,
          reason:
            "Best template match score was below the minimum threshold of 0.65.",
        },
        evaluation: {
          passed: false,
          checks: {
            completed_user_request: false,
            used_correct_template: false,
          },
          issues: ["completed_user_request", "used_correct_template"],
        },
        state: {
          session_id: "session-no-match",
          active_template_id: null,
          blocked_reason: "no_template_match",
        },
        trace: [{ node: "generate_draft", status: "below_match_threshold" }],
      }),
    });
  });

  render(<Home />);

  fireEvent.change(screen.getByPlaceholderText(/summarize today's product sync/i), {
    target: { value: "Write an unrelated template request" },
  });
  fireEvent.click(screen.getByText(/generate with template/i));

  expect(await screen.findByText(/no draft generated/i)).toBeInTheDocument();
  expect(screen.getByText(/best match was 64%, below the 65% minimum/i)).toBeInTheDocument();
  expect(screen.queryByTestId("main-reply")).not.toBeInTheDocument();
  expect(screen.queryByText("Copy")).not.toBeInTheDocument();
});

test("restores the previous session from local storage", async () => {
  window.localStorage.setItem("heywrite.session_id", "session-restore");
  global.fetch = jest.fn((url) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          sessions: [
            {
              session_id: "session-restore",
              title: "Write invitation",
              message_count: 2,
              updated_at: "2026-06-15T10:00:00Z",
              messages: [
                { role: "user", content: "**Write** invitation" },
                { role: "assistant", content: "Restored *draft* body" },
              ],
            },
          ],
        }),
      });
    }
    if (requestUrl.includes("/templates")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ templates: [] }),
      });
    }
    if (requestUrl.includes("/agent/session/session-restore")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          session: {
            session_id: "session-restore",
            active_template_id: "meeting_invitation",
            current_draft: "Restored draft body",
            style: "Formal",
            language: "English",
          },
          messages: [
            { role: "user", content: "**Write** invitation" },
            { role: "assistant", content: "Restored *draft* body" },
          ],
          last_run: {
            output_state: {
              template_meta: {
                used_template: true,
                selected_template: "Meeting Invitation",
                selected_template_id: "meeting_invitation",
              },
              evaluation: {
                passed: true,
                checks: { contains_changes: true },
              },
            },
            trace: [{ node: "evaluator", status: "passed" }],
            reply: "Restored draft body",
          },
        }),
      });
    }
    return Promise.reject(new Error(`Unexpected URL: ${requestUrl}`));
  });

  render(<Home />);

  expect(await screen.findByTestId("main-reply")).toHaveTextContent(
    "Restored draft body"
  );
  expect(await screen.findByTestId("agent-trace")).toHaveTextContent(
    "evaluator"
  );
  expect(await screen.findByTestId("session-list")).toHaveTextContent(
    "Write invitation"
  );
  expect(await screen.findByTestId("session-list")).not.toHaveTextContent("**");
  expect(await screen.findByTestId("session-list")).not.toHaveTextContent(
    "*draft*"
  );
  expect(screen.getAllByText(/Write invitation/i).length).toBeGreaterThan(0);
});

test("reset conversation starts a new current session without clearing saved sessions", async () => {
  window.localStorage.setItem("heywrite.session_id", "session-old");
  const agentRunBodies = [];

  global.fetch = jest.fn((url, options = {}) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          sessions: [
            {
              session_id: "session-old",
              title: "Old saved session",
              message_count: 2,
              updated_at: "2026-06-15T10:00:00Z",
              messages: [
                { role: "user", content: "Old request" },
                { role: "assistant", content: "Old draft" },
              ],
            },
          ],
        }),
      });
    }
    if (requestUrl.includes("/templates")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ templates: [] }),
      });
    }
    if (requestUrl.includes("/agent/session/session-old")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          session: {
            session_id: "session-old",
            current_draft: "Old draft",
            style: "Formal",
            language: "English",
          },
          messages: [
            { role: "user", content: "Old request" },
            { role: "assistant", content: "Old draft" },
          ],
          last_run: { output_state: {}, trace: [], reply: "Old draft" },
        }),
      });
    }
    if (requestUrl.includes("/agent/run")) {
      const body = JSON.parse(options.body || "{}");
      agentRunBodies.push(body);
      return Promise.resolve({
        ok: true,
        json: async () => ({
          reply: "New draft",
          state: { session_id: body.session_id, active_template_id: null },
          trace: [{ node: "start", action: "new_task" }],
        }),
      });
    }
    return Promise.reject(new Error(`Unexpected URL: ${requestUrl}`));
  });

  render(<Home />);

  expect(await screen.findByTestId("main-reply")).toHaveTextContent("Old draft");
  expect(await screen.findByTestId("session-list")).toHaveTextContent(
    "Old saved session"
  );

  fireEvent.click(screen.getByText(/reset conversation/i));
  expect(screen.queryByTestId("main-reply")).not.toBeInTheDocument();
  expect(screen.getByTestId("session-list")).toHaveTextContent(
    "Old saved session"
  );

  const nextSessionId = window.localStorage.getItem("heywrite.session_id");
  expect(nextSessionId).toBeTruthy();
  expect(nextSessionId).not.toBe("session-old");

  fireEvent.change(screen.getByPlaceholderText(/summarize today's product sync/i), {
    target: { value: "Write a new update" },
  });
  fireEvent.click(screen.getByText(/generate with template/i));

  expect(await screen.findByTestId("main-reply")).toHaveTextContent("New draft");
  expect(agentRunBodies[0].session_id).toBe(nextSessionId);
});

test("cleans markdown markers in template library and template modal", async () => {
  global.fetch = jest.fn((url) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ sessions: [] }),
      });
    }
    if (requestUrl.includes("/templates")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          templates: [
            {
              id: "customer_followup",
              title: "**Customer** Follow-up",
              category: "email",
              description: "A *structured* follow-up email",
              tags: ["**customer**", "follow-up"],
              language: "English",
              structure: [],
              content:
                "Subject: **Follow-up**\n\n| Field | Value |\n| --- | --- |\n| Name | *Customer* |",
              enabled: true,
            },
          ],
        }),
      });
    }
    return Promise.reject(new Error(`Unexpected URL: ${requestUrl}`));
  });

  render(<Home />);

  fireEvent.click(screen.getByText(/template library/i));
  const templateRow = await screen.findByText("Customer Follow-up");
  expect(templateRow).not.toHaveTextContent("**");
  fireEvent.click(templateRow);

  expect(await screen.findByText("A structured follow-up email")).toBeInTheDocument();
  expect(screen.getByText("customer")).toBeInTheDocument();
  expect(screen.getByText(/Subject: Follow-up/i)).toBeInTheDocument();
  expect(screen.getByText("Field")).toBeInTheDocument();
  expect(screen.getByText("Value")).toBeInTheDocument();
  expect(screen.queryByText(/\*\*/)).not.toBeInTheDocument();
  expect(screen.queryByText(/\*Customer\*/)).not.toBeInTheDocument();
});

test("resizes conversation history with the drag handle", () => {
  global.fetch = jest.fn((url) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ sessions: [] }),
      });
    }
    return Promise.resolve({
      ok: true,
      json: async () => ({ templates: [] }),
    });
  });

  render(<Home />);

  const handle = screen.getByLabelText("Resize conversation history");
  fireEvent.mouseDown(handle, { clientX: 900 });
  fireEvent.mouseMove(window, { clientX: 820 });
  fireEvent.mouseUp(window);

  const storedWidth = Number(window.localStorage.getItem("heywrite.history_width"));
  expect(storedWidth).toBeGreaterThanOrEqual(240);
  expect(storedWidth).toBeLessThanOrEqual(520);
  expect(storedWidth).not.toBe(300);
});

test("continues editing the current draft without retrieving a new template", async () => {
  global.fetch = jest.fn((url, options = {}) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/agent/sessions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ sessions: [] }),
      });
    }
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
  const agentRunCalls = global.fetch.mock.calls.filter(([url]) =>
    String(url).includes("/agent/run")
  );
  const lastAgentRunOptions = agentRunCalls[agentRunCalls.length - 1][1];
  expect(lastAgentRunOptions.body).toContain('"action":"continue_editing"');
  expect(lastAgentRunOptions.body).toContain(
    '"active_template_id":"meeting_invitation"'
  );
});
