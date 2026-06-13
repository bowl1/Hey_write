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
    expect.stringContaining("/write_with_template"),
    expect.anything()
  );
});

test("shows response after clicking generate button", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      reply: "This is a test reply\n\nChanges:\n- Added a clearer opening",
    }),
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
});

test("continues editing the current draft without retrieving a new template", async () => {
  global.fetch = jest.fn((url) => {
    const requestUrl = String(url);
    if (requestUrl.includes("/templates")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ templates: [] }),
      });
    }
    if (requestUrl.includes("/write_with_template")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          reply: "Dear team,\nPlease join the meeting.",
          template_meta: {
            used_template: true,
            selected_template_id: "meeting_invitation",
            selected_template: "Meeting Invitation",
          },
        }),
      });
    }
    if (requestUrl.includes("/continue_editing")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          reply: "Dear Anna,\nPlease join the meeting.",
          template_meta: {
            used_template: true,
            selected_template_id: "meeting_invitation",
          },
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
  expect(global.fetch).toHaveBeenLastCalledWith(
    expect.stringContaining("/continue_editing"),
    expect.objectContaining({
      body: expect.stringContaining('"active_template_id":"meeting_invitation"'),
    })
  );
});
