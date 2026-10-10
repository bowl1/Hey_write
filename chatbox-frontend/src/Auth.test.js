import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import Auth from "./Auth";
import { supabase } from "./supabase";

jest.mock("./supabase", () => ({
  supabase: { auth: {
    getSession: jest.fn(), onAuthStateChange: jest.fn(),
    signInWithPassword: jest.fn(), signUp: jest.fn(), signOut: jest.fn(),
  } },
}));

let authChanged;
const Workspace = ({ userId, onSignOut }) => <div>
  <span>Workspace {userId}</span><button onClick={onSignOut}>Sign out</button>
</div>;

beforeEach(() => {
  jest.clearAllMocks();
  supabase.auth.getSession.mockResolvedValue({ data: { session: null } });
  supabase.auth.onAuthStateChange.mockImplementation(callback => {
    authChanged = callback;
    return { data: { subscription: { unsubscribe: jest.fn() } } };
  });
});

test("signs in with email and password and opens the account workspace", async () => {
  supabase.auth.signInWithPassword.mockResolvedValue({ error: null });
  render(<Auth Workspace={Workspace} />);
  await screen.findByLabelText("Email");
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
  await waitFor(() => expect(supabase.auth.signInWithPassword).toHaveBeenCalledWith({ email: "a@example.com", password: "password123" }));
  act(() => authChanged("SIGNED_IN", { user: { id: "account-a", email: "a@example.com" } }));
  expect(screen.getByText("Workspace account-a")).toBeInTheDocument();
  act(() => authChanged("SIGNED_OUT", null));
  expect(screen.queryByText("Workspace account-a")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Email")).toBeInTheDocument();
});

test("registration asks the user to confirm their email", async () => {
  supabase.auth.signUp.mockResolvedValue({ error: null });
  render(<Auth Workspace={Workspace} />);
  fireEvent.click(await screen.findByRole("button", { name: "Create an account" }));
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "b@example.com" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
  fireEvent.click(screen.getByRole("button", { name: "Create account" }));
  expect(await screen.findByText("Check your email to confirm your account, then sign in.")).toBeInTheDocument();
  expect(supabase.auth.signUp).toHaveBeenCalledWith(expect.objectContaining({ email: "b@example.com", password: "password123" }));
  expect(screen.queryByText(/Workspace /)).not.toBeInTheDocument();
});

test("shows login errors without opening a workspace", async () => {
  supabase.auth.signInWithPassword.mockResolvedValue({ error: new Error("Invalid login credentials") });
  render(<Auth Workspace={Workspace} />);
  await screen.findByLabelText("Email");
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
  expect(await screen.findByText("Invalid login credentials")).toBeInTheDocument();
});
