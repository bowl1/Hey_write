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
      reply: "This is a test reply",
    }),
  });

  render(<Home />);

  const textarea = screen.getByPlaceholderText(
    /describe what you want to write/i
  );
  fireEvent.change(textarea, {
    target: { value: "Write an invitation email" },
  });

  const button = screen.getByText(/generate with template/i);
  fireEvent.click(button);

  const replyElement = await screen.findByTestId("main-reply");
  expect(replyElement).toHaveTextContent("This is a test reply");
});
