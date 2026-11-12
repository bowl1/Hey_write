import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import MemoryPanel from "./MemoryPanel";
import { apiFetch } from "../supabase";

jest.mock("../supabase", () => ({ apiFetch: jest.fn() }));
const profile = { enabled: true, available: true, facts: [{ key: "name", value: "Anna", evidence: "My name is Anna", source_session_id: "s1" }] };
const response = (body) => ({ ok: true, json: async () => body });

beforeEach(() => jest.clearAllMocks());

test("loads memory on demand, shows evidence, and can pause it", async () => {
  apiFetch.mockResolvedValueOnce(response(profile)).mockResolvedValueOnce(response({ ...profile, enabled: false }));
  render(<MemoryPanel baseUrl="" refreshKey="" />);
  expect(apiFetch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: /Personal Memory/ }));
  expect(await screen.findByText("Anna")).toBeInTheDocument();
  expect(screen.getByText("My name is Anna")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("checkbox"));
  expect(await screen.findByText(/Paused/)).toBeInTheDocument();
  expect(apiFetch).toHaveBeenLastCalledWith("/agent/memory", expect.objectContaining({ method: "PATCH", body: '{"enabled":false}' }));
});

test("clearing needs an explicit confirmation and displays empty memory", async () => {
  apiFetch.mockResolvedValueOnce(response(profile)).mockResolvedValueOnce(response({ ...profile, facts: [] }));
  render(<MemoryPanel baseUrl="" refreshKey="" />);
  fireEvent.click(screen.getByRole("button", { name: /Personal Memory/ }));
  await screen.findByText("Anna");
  fireEvent.click(screen.getByRole("button", { name: "Clear memories" }));
  expect(apiFetch).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Confirm clear" }));
  expect(await screen.findByText("No saved memories yet.")).toBeInTheDocument();
  expect(apiFetch).toHaveBeenLastCalledWith("/agent/memory", expect.objectContaining({ method: "DELETE" }));
});

test("failed requests show a retry without pretending that memory was cleared", async () => {
  apiFetch.mockResolvedValueOnce(response(profile)).mockResolvedValueOnce({ ok: false });
  render(<MemoryPanel baseUrl="" refreshKey="" />);
  fireEvent.click(screen.getByRole("button", { name: /Personal Memory/ }));
  await screen.findByText("Anna");
  fireEvent.click(screen.getByRole("button", { name: "Clear memories" }));
  fireEvent.click(screen.getByRole("button", { name: "Confirm clear" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Memory is unavailable");
  expect(screen.getByText("Anna")).toBeInTheDocument();
});

test("refreshes visible memory after a new draft", async () => {
  apiFetch.mockResolvedValue(response(profile));
  const { rerender } = render(<MemoryPanel baseUrl="" refreshKey="" />);
  fireEvent.click(screen.getByRole("button", { name: /Personal Memory/ }));
  await screen.findByText("Anna");
  rerender(<MemoryPanel baseUrl="" refreshKey="new draft" />);
  await waitFor(() => expect(apiFetch).toHaveBeenCalledTimes(2));
});
