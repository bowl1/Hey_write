import React, { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../supabase";
import { Panel, DropdownHeader, SessionOpenButton } from "../App-style";

type MemoryProfile = {
  enabled: boolean;
  available: boolean;
  facts: { key: string; value: string; evidence: string; source_session_id: string }[];
};

export default function MemoryPanel({ baseUrl, refreshKey }: { baseUrl: string; refreshKey: string }) {
  const [open, setOpen] = useState(false);
  const [profile, setProfile] = useState<MemoryProfile | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [confirmClear, setConfirmClear] = useState(false);
  const request = useRef<AbortController | null>(null);

  const update = useCallback(async (method = "GET", body?: { enabled: boolean }) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    try {
      const result = await apiFetch(`${baseUrl}/agent/memory`, {
        method, signal: controller.signal,
        ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
      });
      if (!result.ok) throw new Error("Memory could not be updated. Please try again.");
      const data = await result.json();
      if (!controller.signal.aborted) {
        setProfile(data);
        setConfirmClear(false);
      }
    } catch (err) {
      if (!controller.signal.aborted) setError("Memory is unavailable. Your drafts are unaffected. Try again.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }, [baseUrl]);

  useEffect(() => {
    if (open) update();
    return () => request.current?.abort();
  }, [open, refreshKey, update]);

  return (
    <Panel>
      <DropdownHeader type="button" aria-expanded={open} aria-controls="personal-memory" onClick={() => setOpen(!open)}>
        <h2>Personal Memory</h2><span aria-hidden="true">{open ? "▲" : "▼"}</span>
      </DropdownHeader>
      {open && <div id="personal-memory" aria-busy={busy}>
        <p>Remember your preferences and reusable details across conversations. Your latest instructions take priority.</p>
        {busy && <p role="status">Updating memory…</p>}
        {error && <p role="alert">{error} <button type="button" onClick={() => update()}>Retry</button></p>}
        {profile && <>
          {!profile.available && <p>Personal memory is disabled on this server.</p>}
          <label>
            <input type="checkbox" checked={profile.enabled} disabled={busy || !profile.available}
              onChange={(event) => update("PATCH", { enabled: event.target.checked })} />
            Use and save personal memory
          </label>
          {!profile.enabled && <p>Paused. Saved facts will not be used or updated until you turn memory on again.</p>}
          {profile.facts.length === 0 ? <p>No saved memories yet.</p> : <ul style={{ paddingLeft: "1.25rem", overflowWrap: "anywhere" }}>
            {profile.facts.map((fact) => <li key={fact.key} style={{ margin: "0.75rem 0" }}>
              {fact.value}
              <details><summary>What you shared</summary><p>{fact.evidence}</p></details>
            </li>)}
          </ul>}
          {confirmClear ? <div>
            <p>Clear all personal memories? Your conversations and their summaries will remain.</p>
            <SessionOpenButton type="button" disabled={busy} onClick={() => update("DELETE")}>Confirm clear</SessionOpenButton>{" "}
            <SessionOpenButton type="button" disabled={busy} onClick={() => setConfirmClear(false)}>Cancel</SessionOpenButton>
          </div> : <SessionOpenButton type="button" disabled={busy || profile.facts.length === 0} onClick={() => setConfirmClear(true)}>Clear memories</SessionOpenButton>}
          <p>Older messages are summarized automatically within each conversation.</p>
        </>}
      </div>}
    </Panel>
  );
}
