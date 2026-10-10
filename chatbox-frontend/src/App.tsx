import React, { useEffect, useRef, useState } from "react";
import { apiFetch } from "./supabase";
import Auth from "./Auth";
import type { WorkspaceProps } from "./Auth";
import {
  PageContainer,
  ChatBox,
  AccountBar,
  AccountEmail,
  AccountButton,
  Title,
  Subtitle,
  IntentTextarea,
  Selectors,
  ResponseControls,
  UndoButton,
  ResetButton,
  GenerateButtons,
  GenerateButton,
  ResponseBox,
  ResponseTitle,
  CopyButton,
  ResponseText,
  ResponseTextHeader,
  ChangesBox,
  AgentTraceBox,
  HistoryPanel,
  HistoryResizeHandle,
  Message,
  SessionList,
  SessionItem,
  SessionRow,
  SessionMeta,
  SessionMessages,
  SessionOpenButton,
  SideStack,
  Panel,
  TemplateList,
  TemplateItem,
  TemplateRow,
  TemplateTagRow,
  TemplateTag,
  TemplateForm,
  TemplateInput,
  TemplateTextarea,
  TemplateMetaBox,
  DropdownHeader,
  TemplateModalBackdrop,
  TemplateModal,
  TemplateModalHeader,
  TemplateModalBody,
  TemplateDocument,
  ModalCloseButton,
} from "./App-style";

import {
  toneOptions,
  languageOptions,
  CustomSelect,
} from "./components/option";

type Template = {
  source?: "default" | "private";
  id: string;
  title: string;
  category: string;
  description: string;
  tags: string[];
  language: string;
  structure: string[];
  content: string;
  enabled: boolean;
};

type TemplateChoice = { id: string; title: string; source?: string; score: number };

type TemplateMeta = {
  requires_selection?: boolean;
  candidates?: TemplateChoice[];
  used_template?: boolean;
  requires_template?: boolean;
  selected_template?: string;
  selected_template_id?: string;
  match_score?: number;
  min_match_score?: number;
  vector_distance?: number;
  bm25_score?: number;
  final_score?: number;
  matched_terms?: string[];
  reason?: string;
};

type AgentEvaluation = {
  passed?: boolean;
  checks?: Record<string, boolean>;
  issues?: string[];
  reason?: string;
  source?: string;
};

type AgentAction = "new_task" | "continue_editing" | "wild";

type AgentTraceItem = {
  node: string;
  [key: string]: any;
};

type AgentMessage = {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
};

type AgentSessionSummary = {
  session_id: string;
  title: string;
  active_template_id?: string | null;
  current_draft?: string;
  style?: string;
  language?: string;
  message_count?: number;
  updated_at?: string | null;
  messages?: AgentMessage[];
};

const emptyTemplateForm = {
  title: "",
  description: "",
  tags: "",
  content: "",
};

const HISTORY_WIDTH_STORAGE_KEY = "heywrite.history_width";

const rejectedDraftNotice = (draft: string) => draft.trim()
  ? "This attempt did not pass evaluation. Your last saved draft has been kept."
  : "This attempt did not pass evaluation. No draft was saved.";

const createSessionId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
const MIN_HISTORY_WIDTH = 240;
const MAX_HISTORY_WIDTH = 520;

const clampHistoryWidth = (value: number) =>
  Math.min(MAX_HISTORY_WIDTH, Math.max(MIN_HISTORY_WIDTH, value));

const splitResponseSections = (text: string) => {
  if (!text) {
    return { draft: "", changes: "" };
  }
  const match = text.match(/(?:^|\n)Changes:\s*/i);
  if (!match || match.index === undefined) {
    return { draft: text, changes: "" };
  }

  return {
    draft: text.slice(0, match.index).trim(),
    changes: text.slice(match.index + match[0].length).trim(),
  };
};

const cleanCopyText = (text: string) =>
  (text || "")
    .replace(/\r\n/g, "\n")
    .replace(/```[\s\S]*?```/g, (block) =>
      block.replace(/^```[a-zA-Z0-9_-]*\n?/, "").replace(/```$/, "")
    )
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/\*\*([^*\n]+)\*\*/g, "$1")
    .replace(/__([^_\n]+)__/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/_([^_\n]+)_/g, "$1")
    .replace(/^\s*\*\s+/gm, "- ")
    .replace(/^\s*[-*]{3,}\s*$/gm, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

const isMarkdownTableSeparator = (line: string) =>
  /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);

const isMarkdownTableRow = (line: string) => line.includes("|");

const splitMarkdownTableRow = (line: string) =>
  line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cleanCopyText(cell.trim()));

const markdownTablesToTsv = (text: string) => {
  const lines = (text || "").replace(/\r\n/g, "\n").split("\n");
  const output: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const nextLine = lines[index + 1] || "";
    if (isMarkdownTableRow(line) && isMarkdownTableSeparator(nextLine)) {
      output.push(splitMarkdownTableRow(line).join("\t"));
      index += 2;
      while (index < lines.length && isMarkdownTableRow(lines[index])) {
        output.push(splitMarkdownTableRow(lines[index]).join("\t"));
        index += 1;
      }
      index -= 1;
      continue;
    }
    output.push(line);
  }

  return output.join("\n");
};

const cleanDisplayText = (text: string) => cleanCopyText(markdownTablesToTsv(text));

const renderFormattedText = (text: string) => {
  const lines = (text || "").replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let textBuffer: string[] = [];

  const flushText = () => {
    const value = cleanCopyText(textBuffer.join("\n"));
    if (value) {
      value.split(/\n{2,}/).forEach((paragraph) => {
        const cleanedParagraph = paragraph.trim();
        if (cleanedParagraph) {
          blocks.push(
            <p key={`text-${blocks.length}`}>{cleanedParagraph}</p>
          );
        }
      });
    }
    textBuffer = [];
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const nextLine = lines[index + 1] || "";

    if (isMarkdownTableRow(line) && isMarkdownTableSeparator(nextLine)) {
      flushText();
      const headers = splitMarkdownTableRow(line);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && isMarkdownTableRow(lines[index])) {
        rows.push(splitMarkdownTableRow(lines[index]));
        index += 1;
      }
      index -= 1;
      blocks.push(
        <table key={`table-${blocks.length}`}>
          <thead>
            <tr>
              {headers.map((header, cellIndex) => (
                <th key={`${header}-${cellIndex}`}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {headers.map((_header, cellIndex) => (
                  <td key={cellIndex}>{row[cellIndex] || ""}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
      continue;
    }

    textBuffer.push(line);
  }

  flushText();
  return blocks;
};

const previewInline = (text: string): React.ReactNode[] =>
  text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]\n]+\])/g).map((part, index) => {
    if (part.startsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*")) return <em key={index}>{part.slice(1, -1)}</em>;
    if (part.startsWith("[")) return <mark key={index}>{part}</mark>;
    return part;
  });

const renderTemplatePreview = (text: string): React.ReactNode[] => {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    const key = blocks.length;
    if (isMarkdownTableRow(line) && isMarkdownTableSeparator(lines[i + 1] || "")) {
      const table: string[] = [line, lines[++i]];
      while (isMarkdownTableRow(lines[i + 1] || "")) table.push(lines[++i]);
      const headers = splitMarkdownTableRow(table[0]);
      blocks.push(<div className="preview-table" key={key}><table>
        <thead><tr>{headers.map((cell, index) => <th key={index}>{previewInline(cell)}</th>)}</tr></thead>
        <tbody>{table.slice(2).map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, cellIndex) => <td key={cellIndex}>{previewInline(splitMarkdownTableRow(row)[cellIndex] || "")}</td>)}</tr>)}</tbody>
      </table></div>);
    } else if (/^#{1,6}\s/.test(line)) {
      const title = previewInline(line.replace(/^#{1,6}\s+/, ""));
      blocks.push(line.startsWith("# ") ? <h2 key={key}>{title}</h2> : <h3 key={key}>{title}</h3>);
    } else if (/^([-*_])\1{2,}$/.test(line)) {
      blocks.push(<hr key={key} />);
    } else if (/^(?:[-*] |\d+\. )/.test(line)) {
      const ordered = /^\d+\. /.test(line);
      const pattern = ordered ? /^\d+\. / : /^[-*] /;
      const items: React.ReactNode[] = [];
      do { items.push(<li key={items.length}>{previewInline(lines[i].trim().replace(pattern, ""))}</li>); i += 1; }
      while (i < lines.length && pattern.test(lines[i].trim()));
      i -= 1;
      blocks.push(ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>);
    } else {
      const paragraph = [line];
      while (i + 1 < lines.length && lines[i + 1].trim() && !/^(?:#{1,6}\s|[-*] |\d+\. |\||([-*_])\1{2,}$)/.test(lines[i + 1].trim())) paragraph.push(lines[++i]);
      blocks.push(<p key={key}>{previewInline(paragraph.join("\n"))}</p>);
    }
  }
  return blocks;
};

export const Home: React.FC<WorkspaceProps> = ({ userId, email, onSignOut, onRequireAuth, pendingGeneration, onPendingHandled }) => {
  const SESSION_STORAGE_KEY = `heywrite.session_id:${userId}`;
  const [intent, setIntent] = useState<string>(pendingGeneration?.intent || "");
  const [style, setStyle] = useState<string>(pendingGeneration?.style || "Formal");
  const [language, setLanguage] = useState<string>(pendingGeneration?.language || "English");
  const [chosenTemplate, setChosenTemplate] = useState<{ id: string; title: string } | null>(
    pendingGeneration?.selectedTemplateId ? { id: pendingGeneration.selectedTemplateId, title: pendingGeneration.selectedTemplateTitle || "Selected template" } : null
  );
  const pendingStarted = useRef(false);
  const [response, setResponse] = useState<string>("");
  const [lastResponse, setLastResponse] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [history, setHistory] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);
  const [sessions, setSessions] = useState<AgentSessionSummary[]>([]);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(
    null
  );
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateStatus, setTemplateStatus] = useState<"idle" | "loading" | "loaded" | "error">("idle");
  const [templateMeta, setTemplateMeta] = useState<TemplateMeta | null>(null);
  const [draftNotice, setDraftNotice] = useState("");
  const [agentEvaluation, setAgentEvaluation] =
    useState<AgentEvaluation | null>(null);
  const [historyWidth, setHistoryWidth] = useState<number>(() => {
    const storedWidth = Number(
      window.localStorage.getItem(HISTORY_WIDTH_STORAGE_KEY)
    );
    return Number.isFinite(storedWidth)
      ? clampHistoryWidth(storedWidth)
      : 300;
  });
  const [templateForm, setTemplateForm] = useState(emptyTemplateForm);
  const [templateMessage, setTemplateMessage] = useState<string>("");
  const [templateLibraryOpen, setTemplateLibraryOpen] =
    useState<boolean>(false);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(
    null
  );
  const previewRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!selectedTemplate) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    previewRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedTemplate(null);
      if (event.key === "Tab") {
        const buttons = previewRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)");
        if (!buttons?.length) return;
        const first = buttons[0];
        const last = buttons[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [selectedTemplate]);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activeTemplateId, setActiveTemplateId] = useState<string | null>(null);
  const [agentTrace, setAgentTrace] = useState<AgentTraceItem[]>([]);

  const BASE_URL =
    process.env.NODE_ENV === "development"
      ? "http://localhost:8000"
      : process.env.REACT_APP_API_URL || "";
  const responseSections = splitResponseSections(response);
  const cleanDraft = cleanDisplayText(responseSections.draft || response);
  const cleanChanges = cleanDisplayText(responseSections.changes);
  const hasAgentResult = Boolean(response || templateMeta || agentTrace.length);

  const summarizeTrace = (item: AgentTraceItem) => {
    if (item.node === "start") return item.action;
    if (item.node === "retrieve_template")
      return `${item.results?.length || 0} candidates`;
    if (item.node === "generate_draft")
      return item.selected_template_id || item.status;
    if (item.node === "load_active_template")
      return item.template_id || "none";
    if (item.node === "revise_draft")
      return item.used_active_template ? "active template" : item.status;
    if (item.node === "planner") return item.decision;
    if (item.node === "evaluator")
      return item.failed_checks?.length
        ? item.failed_checks.join(", ")
        : item.status;
    if (item.node === "revise_with_feedback") return item.status;
    return item.status || item.decision || "";
  };

  const labelEvaluationCheck = (name: string) =>
    name
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());

  const formatSessionDate = (value?: string | null) => {
    if (!value) return "No date";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "No date";
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  };

  const loadTemplates = async (force = false) => {
    if (!force && process.env.NODE_ENV === "test") return;
    if (!BASE_URL && process.env.NODE_ENV === "production") { setTemplateStatus("error"); return; }
    setTemplateStatus("loading");
    try {
      const res = userId ? await apiFetch(`${BASE_URL}/templates`) : await fetch(`${BASE_URL}/public/templates`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setTemplates(data.templates || []);
      setTemplateStatus("loaded");
    } catch (error) {
      setTemplateStatus("error");
      console.error("Failed to load templates:", error);
    }
  };

  const applySessionSnapshot = (data: any, fallbackSessionId: string) => {
    const session = data.session || {};
    const lastRun = data.last_run || {};
    const outputState = lastRun.output_state || {};
    const nextSessionId = session.session_id || fallbackSessionId;

    setSessionId(nextSessionId);
    setExpandedSessionId(nextSessionId);
    setActiveTemplateId(session.active_template_id || null);
    const savedDraft = session.current_draft ?? outputState.current_draft ?? "";
    setResponse(savedDraft);
    setLastResponse(savedDraft);
    setDraftNotice(outputState.draft_accepted === false ? rejectedDraftNotice(savedDraft) : "");
    setStyle(session.style || "Formal");
    setLanguage(session.language || "English");
    setHistory(data.messages || []);
    setTemplateMeta(outputState.template_meta || null);
    setAgentEvaluation(outputState.evaluation || null);
    setAgentTrace(lastRun.trace || []);
    window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
  };

  const loadSessions = async () => {
    if (!userId) return;
    if (!BASE_URL && process.env.NODE_ENV === "production") return;
    try {
      const res = await apiFetch(`${BASE_URL}/agent/sessions`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error("Failed to load sessions:", error);
    }
  };

  const restoreSessionById = async (targetSessionId: string) => {
    if (!userId) return;
    if (!targetSessionId) return;
    if (!BASE_URL && process.env.NODE_ENV === "production") return;

    try {
      const res = await apiFetch(`${BASE_URL}/agent/session/${encodeURIComponent(targetSessionId)}`);
      if (res.status === 404) {
        if (targetSessionId === window.localStorage.getItem(SESSION_STORAGE_KEY)) {
          window.localStorage.removeItem(SESSION_STORAGE_KEY);
        }
        return;
      }
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      applySessionSnapshot(await res.json(), targetSessionId);
    } catch (error) {
      console.error("Failed to restore session:", error);
    }
  };

  const restoreStoredSession = async () => {
    const storedSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!storedSessionId) return;
    await restoreSessionById(storedSessionId);
  };

  useEffect(() => {
    loadTemplates();
    if (!userId) return;
    const storedSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (process.env.NODE_ENV !== "test" || storedSessionId) {
      loadSessions();
    }
    if (!pendingGeneration) restoreStoredSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resetConversation = () => {
    const nextSessionId = createSessionId();
    if (userId) window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
    setSessionId(nextSessionId);
    setActiveTemplateId(null);
    setResponse("");
    setLastResponse("");
    setHistory([]);
    setTemplateMeta(null);
    setAgentEvaluation(null);
    setAgentTrace([]);
    setDraftNotice("");
  };

  const runAgent = async (action: AgentAction, explicitTemplate?: { id: string; title: string }) => {
    const selection = explicitTemplate || chosenTemplate;
    if (!intent.trim()) return;
    if (action === "continue_editing" && !response.trim()) return;
    if (!userId) {
      onRequireAuth?.({ action, intent, style, language, selectedTemplateId: selection?.id, selectedTemplateTitle: selection?.title });
      return;
    }
    const previousResponse = response;
    const currentDraft = responseSections.draft || response;
    setLoading(true);
    setLastResponse(response);
    setResponse("");
    setTemplateMeta(null);
    setAgentEvaluation(null);
    setAgentTrace([]);

    try {
      const res = await apiFetch(`${BASE_URL}/agent/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          action,
          selected_template_id: action === "new_task" ? selection?.id : null,
          intent,
          style,
          language,
          history,
          current_draft: action === "continue_editing" ? currentDraft : "",
          active_template_id:
            action === "continue_editing" ? activeTemplateId : null,
        }),
      });

      if (!res.ok)
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      const data = await res.json();
      const accepted = data.state?.draft_accepted !== false;
      const savedDraft = data.state?.current_draft ?? previousResponse;
      setResponse(accepted ? data.reply || "" : savedDraft);
      setDraftNotice(accepted || data.template_meta?.requires_selection ? "" : rejectedDraftNotice(savedDraft));
      setTemplateMeta(data.template_meta || null);
      setAgentEvaluation(data.evaluation || null);
      setAgentTrace(data.trace || []);
      const nextSessionId = data.state?.session_id || sessionId;
      setSessionId(nextSessionId);
      if (nextSessionId) {
        window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
      }
      setActiveTemplateId(data.state?.active_template_id || null);
      setHistory((prev) => [
        ...prev,
        { role: "user", content: intent },
        ...(accepted ? [{ role: "assistant" as const, content: data.reply || "" }] : []),
      ]);
      if (process.env.NODE_ENV !== "test") {
        loadSessions();
      }
    } catch (error: any) {
      console.error("Agent run failed:", error);
      alert(`Generation failed: ${error.message}`);
      setResponse(action === "continue_editing" ? previousResponse : "");
    } finally {
      setLoading(false);
    }
  };

  const handleUndo = () => {
    setResponse(lastResponse);
    const newHistory = [...history];
    for (let i = newHistory.length - 1; i >= 0; i--) {
      if (newHistory[i].role === "assistant") {
        newHistory[i].content = lastResponse;
        break;
      }
    }
    setHistory(newHistory);
  };

  useEffect(() => {
    if (!userId || !pendingGeneration || pendingStarted.current) return;
    pendingStarted.current = true;
    runAgent(pendingGeneration.action);
    onPendingHandled?.();
    // The queued request initializes this account's newly mounted workspace.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  const handleCopy = () => {
    if (cleanDraft) {
      navigator.clipboard.writeText(cleanDraft);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleCreateTemplate = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    if (!templateForm.title.trim() || !templateForm.content.trim()) {
      setTemplateMessage("Template title and content are required.");
      return;
    }
    if (!userId) { onRequireAuth?.(); return; }

    setTemplateMessage("Saving template...");
    try {
      const res = await apiFetch(`${BASE_URL}/templates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: templateForm.title,
          description: templateForm.description,
          tags: templateForm.tags
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean),
          style,
          language,
          content: templateForm.content,
        }),
      });

      if (!res.ok)
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);

      const data = await res.json();
      setTemplateForm(emptyTemplateForm);
      setTemplateMessage(
        data.index_status?.ok
          ? "Template saved and indexed."
          : `Template saved. Index update failed: ${
              data.index_status?.error || "unknown error"
            }`
      );
      await loadTemplates();
    } catch (error: any) {
      console.error("Template save failed:", error);
      setTemplateMessage(`Template save failed: ${error.message}`);
    }
  };

  const handleHistoryResizeStart = (
    event: React.MouseEvent<HTMLDivElement>
  ) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = historyWidth;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const nextWidth = clampHistoryWidth(startWidth - (moveEvent.clientX - startX));
      setHistoryWidth(nextWidth);
      window.localStorage.setItem(
        HISTORY_WIDTH_STORAGE_KEY,
        String(nextWidth)
      );
    };

    const handleMouseUp = () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  };

  const currentSessionFallback: AgentSessionSummary | null =
    sessionId && history.length > 0
      ? {
          session_id: sessionId,
          title:
            history.find((message) => message.role === "user")?.content ||
            "Current session",
          active_template_id: activeTemplateId,
          current_draft: response,
          style,
          language,
          message_count: history.length,
          messages: history,
        }
      : null;
  const visibleSessions =
    currentSessionFallback &&
    !sessions.some((session) => session.session_id === sessionId)
      ? [currentSessionFallback, ...sessions]
      : sessions;
  const sessionMessages = (session: AgentSessionSummary) =>
    session.session_id === sessionId ? history : session.messages || [];

  return (
    <PageContainer className="page-home" historyWidth={historyWidth}>
        <AccountBar>
          {userId && <AccountEmail title={email}>{email}</AccountEmail>}
          <AccountButton type="button" onClick={userId ? onSignOut : () => onRequireAuth?.()}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
              <circle cx="12" cy="8" r="3.5" />
              <path d="M5 20v-1a7 7 0 0 1 14 0v1" />
            </svg>
            {userId ? "Sign out" : "Sign in"}
          </AccountButton>
        </AccountBar>
      <SideStack>
        <Panel>
          <DropdownHeader
            type="button"
            onClick={() => {
              setTemplateLibraryOpen((open) => !open);
              if (!templateLibraryOpen) {
                loadTemplates(true);
              }
            }}
            aria-expanded={templateLibraryOpen}
          >
            <h2>Template Library</h2>
            <span>
              {templates.length} {templateLibraryOpen ? "▲" : "▼"}
            </span>
          </DropdownHeader>
          {templateLibraryOpen && (
            <TemplateList role="region" aria-label="Available templates" tabIndex={0}>
              {templates.length === 0 && (
                <TemplateItem>
                  <h3>{templateStatus === "error" ? "Unable to load templates" : templateStatus === "loading" ? "Loading templates…" : "No templates available"}</h3>
                  <p>{templateStatus === "error" ? "Please try opening the library again." : templateStatus === "loading" ? "Your templates will appear shortly." : "Create a template to get started."}</p>
                </TemplateItem>
              )}
              {templates.map((template) => (
                <TemplateItem key={template.id}>
                  <TemplateRow
                    type="button"
                    onClick={() => setSelectedTemplate(template)}
                  >
                    <h3>{cleanCopyText(template.title)}</h3>
                    {template.source === "default" && <TemplateTag>Default</TemplateTag>}
                  </TemplateRow>
                </TemplateItem>
              ))}
            </TemplateList>
          )}
        </Panel>

        <Panel>
          <h2>Create Template</h2>
          <TemplateForm onSubmit={handleCreateTemplate}>
            <TemplateInput
              placeholder="Template title"
              value={templateForm.title}
              onChange={(e) =>
                setTemplateForm((prev) => ({ ...prev, title: e.target.value }))
              }
            />
            <TemplateInput
              placeholder="Description"
              value={templateForm.description}
              onChange={(e) =>
                setTemplateForm((prev) => ({
                  ...prev,
                  description: e.target.value,
                }))
              }
            />
            <TemplateInput
              placeholder="Tags, separated by commas"
              value={templateForm.tags}
              onChange={(e) =>
                setTemplateForm((prev) => ({ ...prev, tags: e.target.value }))
              }
            />
            <TemplateTextarea
              placeholder="Paste the template body here"
              value={templateForm.content}
              onChange={(e) =>
                setTemplateForm((prev) => ({
                  ...prev,
                  content: e.target.value,
                }))
              }
            />
            <GenerateButton type="submit">Save Template</GenerateButton>
            {templateMessage && <p>{templateMessage}</p>}
          </TemplateForm>
        </Panel>
      </SideStack>

      <ChatBox>

        <Title>Hey Write!</Title>
        <Subtitle>Write in your style — save templates, generate with them anytime.</Subtitle>

        {chosenTemplate && <TemplateMetaBox>
          <p>Using template: <strong>{chosenTemplate.title}</strong></p>
          <ModalCloseButton type="button" onClick={() => setChosenTemplate(null)}>Use automatic matching</ModalCloseButton>
        </TemplateMetaBox>}
        <IntentTextarea
          rows={5}
          placeholder="e.g., Summarize today's product sync meeting, or draft this week's report for Project Alpha"
          value={intent}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
            setIntent(e.target.value)
          }
        />

        <Selectors>
          <CustomSelect
            label="Choose Tone"
            options={toneOptions}
            value={toneOptions.find((option) => option.value === style) || null}
            onChange={(option) => option && setStyle(option.value)}
          />

          <CustomSelect
            label="Language"
            options={languageOptions}
            value={
              languageOptions.find((option) => option.value === language) ||
              null
            }
            onChange={(option) => option && setLanguage(option.value)}
          />
        </Selectors>

        <ResponseControls>
          {lastResponse && (
            <UndoButton onClick={handleUndo}>
              ⬅️ Back to Last Version
            </UndoButton>
          )}
          <ResetButton onClick={resetConversation}>
            🔄 Reset Conversation
          </ResetButton>
        </ResponseControls>

        <GenerateButtons>
          <GenerateButton
            onClick={() => runAgent("new_task")}
            disabled={loading}
          >
            Generate with Template
          </GenerateButton>
          <GenerateButton
            onClick={() => runAgent("continue_editing")}
            disabled={loading || !response}
          >
            Continue Editing
          </GenerateButton>
          <GenerateButton onClick={() => runAgent("wild")} disabled={loading}>
            {loading
              ? "Writing... It takes around 1 minute"
              : "✨ Generate something wild"}
          </GenerateButton>
        </GenerateButtons>

        {draftNotice && <p role="status">{draftNotice}</p>}
        {hasAgentResult && (
          <ResponseBox>
            {response && <ResponseTitle>Your Draft</ResponseTitle>}
            {templateMeta && (
              <TemplateMetaBox>
                <strong>{templateMeta.requires_selection ? "Choose a template" : "Template source"}</strong>
                {templateMeta.requires_selection ? <>
                  <p>{templateMeta.reason}</p>
                  {(templateMeta.candidates || []).map(candidate => <TemplateItem key={candidate.id}>
                    <TemplateRow type="button" disabled={loading} onClick={() => {
                      setChosenTemplate({ id: candidate.id, title: candidate.title });
                      runAgent("new_task", candidate);
                    }}>
                      <h3>{candidate.title}</h3>
                      <span>{candidate.source === "private" ? "Private" : "Default"}</span>
                    </TemplateRow>
                  </TemplateItem>)}
                </> : templateMeta.used_template ? (
                  <>
                    Used {templateMeta.selected_template || "a matched template"}
                    {typeof templateMeta.match_score === "number"
                      ? `, match score ${Math.round(
                          templateMeta.match_score * 100
                        )}%`
                      : ""}
                    . {templateMeta.reason}
                  </>
                ) : (
                  <>
                    No draft generated.{" "}
                    {templateMeta.reason ||
                      "No suitable template matched this request."}
                    {typeof templateMeta.match_score === "number" &&
                    typeof templateMeta.min_match_score === "number"
                      ? ` Best match was ${Math.round(
                          templateMeta.match_score * 100
                        )}%, below the ${Math.round(
                          templateMeta.min_match_score * 100
                        )}% minimum.`
                      : ""}
                  </>
                )}
              </TemplateMetaBox>
            )}
            {response && (
              <ResponseText data-testid="main-reply">
                <ResponseTextHeader>
                  <CopyButton onClick={handleCopy}>
                    {copied ? "Copied!" : "Copy"}
                  </CopyButton>
                </ResponseTextHeader>
                {renderFormattedText(responseSections.draft || response)}
              </ResponseText>
            )}
            {cleanChanges && (
              <ChangesBox data-testid="changes-summary">
                <strong>Changes</strong>
                <p>{cleanChanges}</p>
              </ChangesBox>
            )}
            {agentTrace.length > 0 && (
              <AgentTraceBox data-testid="agent-trace">
                <strong>Agent trace</strong>
                {agentTrace.map((item, index) => (
                  <p key={`${item.node}-${index}`}>
                    <span>{item.node}</span>
                    {summarizeTrace(item)}
                  </p>
                ))}
              </AgentTraceBox>
            )}
            {agentEvaluation?.checks && (
              <AgentTraceBox data-testid="agent-evaluation">
                <strong>
                  Evaluator {agentEvaluation.passed ? "passed" : "failed"}
                </strong>
                {(agentEvaluation.source || agentEvaluation.reason) && (
                  <p>
                    <span>{agentEvaluation.source || "source"}</span>
                    {agentEvaluation.reason || ""}
                  </p>
                )}
                {Object.entries(agentEvaluation.checks).map(
                  ([name, passed]) => (
                    <p key={name}>
                      <span>{passed ? "Pass" : "Fail"}</span>
                      {labelEvaluationCheck(name)}
                    </p>
                  )
                )}
              </AgentTraceBox>
            )}
          </ResponseBox>
        )}
      </ChatBox>

      <HistoryPanel>
        <HistoryResizeHandle
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize conversation history"
          onMouseDown={handleHistoryResizeStart}
        />
        <h2>Conversation History</h2>
        {visibleSessions.length === 0 && <p>No saved sessions yet.</p>}
        {visibleSessions.length > 0 && (
          <SessionList data-testid="session-list">
            {visibleSessions.map((session) => {
              const expanded = expandedSessionId === session.session_id;
              const messages = sessionMessages(session);
              return (
                <SessionItem
                  key={session.session_id}
                  data-testid={`session-item-${session.session_id}`}
                  data-active={session.session_id === sessionId}
                >
                  <SessionRow
                    type="button"
                    onClick={() =>
                      setExpandedSessionId(expanded ? null : session.session_id)
                    }
                    aria-expanded={expanded}
                  >
                    <div>
                      <h3>{session.title || "Untitled session"}</h3>
                      <SessionMeta>
                        {session.message_count || messages.length} messages ·{" "}
                        {formatSessionDate(session.updated_at)}
                      </SessionMeta>
                    </div>
                    <span>{expanded ? "▲" : "▼"}</span>
                  </SessionRow>
                  {expanded && (
                    <SessionMessages>
                      {session.session_id !== sessionId && (
                        <SessionOpenButton
                          type="button"
                          onClick={() => restoreSessionById(session.session_id)}
                        >
                          Open session
                        </SessionOpenButton>
                      )}
                      {messages.map((msg, index) => (
                        <Message
                          key={`${session.session_id}-${index}`}
                          className={msg.role}
                          role={msg.role}
                        >
                          <strong>
                            {msg.role === "user" ? "You" : "HeyWrite"}:
                          </strong>
                          <div>{renderFormattedText(msg.content)}</div>
                        </Message>
                      ))}
                    </SessionMessages>
                  )}
                </SessionItem>
              );
            })}
          </SessionList>
        )}
      </HistoryPanel>

      {selectedTemplate && (
        <TemplateModalBackdrop onClick={() => setSelectedTemplate(null)}>
          <TemplateModal ref={previewRef} role="dialog" aria-modal="true" aria-labelledby="template-preview-title" onClick={(event) => event.stopPropagation()}>
            <TemplateModalHeader>
              <div>
                <h3 id="template-preview-title">{cleanCopyText(selectedTemplate.title)}</h3>
                <p>Template preview · {selectedTemplate.language}</p>
              </div>
              <ModalCloseButton
                type="button"
                onClick={() => setSelectedTemplate(null)}
              >
                Close
              </ModalCloseButton>
            </TemplateModalHeader>
            <TemplateModalBody>
              <GenerateButton type="button" onClick={() => {
                setChosenTemplate({ id: selectedTemplate.id, title: cleanCopyText(selectedTemplate.title) });
                setSelectedTemplate(null);
              }}>Use this template</GenerateButton>
              <p>{cleanCopyText(selectedTemplate.description)}</p>
              <TemplateTagRow>
                {(selectedTemplate.tags || []).map((tag) => (
                  <TemplateTag key={tag}>{cleanCopyText(tag)}</TemplateTag>
                ))}
              </TemplateTagRow>
              <p className="preview-hint">Replace the highlighted fields with your details.</p>
              <TemplateDocument>{renderTemplatePreview(selectedTemplate.content)}</TemplateDocument>
            </TemplateModalBody>
          </TemplateModal>
        </TemplateModalBackdrop>
      )}
    </PageContainer>
  );
};

export default function App() {
  return <Auth Workspace={Home} />;
}
