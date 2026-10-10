import React, { useEffect, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "./supabase";
import { AuthDialog, AuthForm, AuthField, AuthFooter, AuthSwitchButton, AuthNotice, Title, Subtitle, TemplateInput, GenerateButton, ModalCloseButton } from "./App-style";

export type PendingGeneration = { action: "new_task" | "continue_editing" | "wild"; intent: string; style: string; language: string };
export type WorkspaceProps = {
  userId?: string; email?: string; onSignOut?: () => void;
  onRequireAuth?: (request?: PendingGeneration) => void;
  pendingGeneration?: PendingGeneration | null; onPendingHandled?: () => void;
};

export default function Auth({ Workspace }: { Workspace: React.ComponentType<WorkspaceProps> }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [signUp, setSignUp] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [open, setOpen] = useState(false);
  const [pendingGeneration, setPendingGeneration] = useState<PendingGeneration | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (open && !session) dialog.current?.showModal();
    else dialog.current?.close();
  }, [open, session]);

  useEffect(() => {
    if (!supabase) { setReady(true); return; }
    let active = true;
    let changed = false;
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, next) => {
      changed = true;
      if (active) { setSession(next); setReady(true); if (next) setOpen(false); }
    });
    supabase.auth.getSession().then(({ data, error }) => {
      if (active && !changed) { setSession(data.session); setReady(true); if (data.session) setOpen(false); }
      if (active && error) setMessage("Unable to restore your session. Please sign in again.");
    }).catch(() => {
      if (active) { setReady(true); setMessage("Unable to restore your session."); }
    });
    return () => { active = false; subscription.unsubscribe(); };
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!supabase) return;
    setBusy(true); setMessage("");
    try {
      const { error } = signUp
        ? await supabase.auth.signUp({ email, password, options: { emailRedirectTo: window.location.origin } })
        : await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      setPassword("");
      if (signUp) setMessage("Check your email to confirm your account, then sign in.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to sign in. Please try again.");
    } finally { setBusy(false); }
  }

  async function signOut() {
    const { error } = await supabase!.auth.signOut();
    if (error) setMessage("Unable to sign out. Please try again.");
  }

  const closeDialog = () => { setOpen(false); setPendingGeneration(null); setPassword(""); setMessage(""); };
  return <>
    <Workspace key={session?.user.id || "guest"} userId={session?.user.id} email={session?.user.email} onSignOut={signOut}
      onRequireAuth={request => { setPendingGeneration(request || null); setOpen(true); }}
      pendingGeneration={session ? pendingGeneration : null} onPendingHandled={() => setPendingGeneration(null)} />
    {session && message && <p role="alert">{message}</p>}
    <AuthDialog ref={dialog} aria-label="Sign in to generate" onCancel={event => { event.preventDefault(); if (!busy) closeDialog(); }}>
    <ModalCloseButton type="button" aria-label="Close sign-in" disabled={busy} onClick={closeDialog}>×</ModalCloseButton>
    <Title>Hey Write!</Title>
    <Subtitle>{signUp ? "Create your private writing workspace." : pendingGeneration ? "Sign in to start writing. Your request is saved." : "Sign in to your writing workspace."}</Subtitle>
    {!ready && <p role="status">Checking your session…</p>}
    {!supabase ? <p role="alert">Sign-in is not configured. Please contact the workspace administrator.</p> : <>
      <AuthForm onSubmit={submit}>
        <AuthField>
          <label htmlFor="auth-email">Email</label>
          <TemplateInput id="auth-email" type="email" autoComplete="email" required value={email} onChange={e => setEmail(e.target.value)} />
        </AuthField>
        <AuthField>
          <label htmlFor="auth-password">Password</label>
          <TemplateInput id="auth-password" type="password" autoComplete={signUp ? "new-password" : "current-password"} minLength={6} required value={password} onChange={e => setPassword(e.target.value)} />
        </AuthField>
        <GenerateButton disabled={busy} type="submit">{busy ? "Please wait…" : signUp ? "Create account" : "Sign in"}</GenerateButton>
      </AuthForm>
      <AuthFooter>
        <span>{signUp ? "Already have an account?" : "New to Hey Write?"}</span>
        <AuthSwitchButton type="button" disabled={busy} onClick={() => { setSignUp(!signUp); setMessage(""); }}>
          {signUp ? "Sign in" : "Create an account"}
          <span aria-hidden="true">→</span>
        </AuthSwitchButton>
      </AuthFooter>
    </>}
    {message && <AuthNotice role="status">{message}</AuthNotice>}
  </AuthDialog></>;
}
