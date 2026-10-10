import React, { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "./supabase";
import { ChatBox, Title, Subtitle, TemplateForm, TemplateInput, GenerateButton } from "./App-style";

export default function Auth({ Workspace }: { Workspace: React.ComponentType<{ userId: string; email?: string; onSignOut?: () => void }> }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [signUp, setSignUp] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!supabase) { setReady(true); return; }
    let active = true;
    let changed = false;
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, next) => {
      changed = true;
      if (active) { setSession(next); setReady(true); }
    });
    supabase.auth.getSession().then(({ data, error }) => {
      if (active && !changed) { setSession(data.session); setReady(true); }
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

  if (!ready) return <p role="status">Loading your workspace…</p>;
  if (session) return <>
    {message && <p role="alert">{message}</p>}
    <Workspace key={session.user.id} userId={session.user.id} email={session.user.email} onSignOut={signOut} />
  </>;
  return <ChatBox style={{ maxWidth: 440, margin: "64px auto" }}>
    <Title>Hey Write!</Title>
    <Subtitle>{signUp ? "Create your private writing workspace." : "Sign in to your writing workspace."}</Subtitle>
    {!supabase ? <p role="alert">Sign-in is not configured. Please contact the workspace administrator.</p> : <>
      <TemplateForm onSubmit={submit}>
        <label htmlFor="auth-email">Email</label>
        <TemplateInput id="auth-email" type="email" autoComplete="email" required value={email} onChange={e => setEmail(e.target.value)} />
        <label htmlFor="auth-password">Password</label>
        <TemplateInput id="auth-password" type="password" autoComplete={signUp ? "new-password" : "current-password"} minLength={6} required value={password} onChange={e => setPassword(e.target.value)} />
        <GenerateButton disabled={busy} type="submit">{busy ? "Please wait…" : signUp ? "Create account" : "Sign in"}</GenerateButton>
      </TemplateForm>
      <button type="button" disabled={busy} onClick={() => { setSignUp(!signUp); setMessage(""); }}>{signUp ? "Already have an account? Sign in" : "Create an account"}</button>
    </>}
    {message && <p role="status">{message}</p>}
  </ChatBox>;
}
