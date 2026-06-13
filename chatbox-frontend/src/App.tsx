import React, { useEffect, useState } from "react";
import {
  PageContainer,
  ChatBox,
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
  HistoryPanel,
  Message,
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
  ModalCloseButton,
} from "./App-style";

import {
  toneOptions,
  languageOptions,
  CustomSelect,
} from "./components/option";

type Template = {
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

type TemplateMeta = {
  used_template?: boolean;
  selected_template?: string;
  selected_template_id?: string;
  match_score?: number;
  reason?: string;
};

const emptyTemplateForm = {
  title: "",
  description: "",
  tags: "",
  content: "",
};

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

const Home: React.FC = () => {
  const [intent, setIntent] = useState<string>("");
  const [style, setStyle] = useState<string>("Formal");
  const [language, setLanguage] = useState<string>("English");
  const [response, setResponse] = useState<string>("");
  const [lastResponse, setLastResponse] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [history, setHistory] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateMeta, setTemplateMeta] = useState<TemplateMeta | null>(null);
  const [templateForm, setTemplateForm] = useState(emptyTemplateForm);
  const [templateMessage, setTemplateMessage] = useState<string>("");
  const [templateLibraryOpen, setTemplateLibraryOpen] =
    useState<boolean>(false);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(
    null
  );
  const [activeTemplateId, setActiveTemplateId] = useState<string | null>(null);

  const BASE_URL =
    process.env.NODE_ENV === "development"
      ? "http://localhost:8000"
      : process.env.REACT_APP_API_URL || "";
  const responseSections = splitResponseSections(response);

  const loadTemplates = async () => {
    if (!BASE_URL && process.env.NODE_ENV !== "development") return;
    try {
      const res = await fetch(`${BASE_URL}/templates`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setTemplates(data.templates || []);
    } catch (error) {
      console.error("Failed to load templates:", error);
    }
  };

  useEffect(() => {
    loadTemplates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSubmit = async () => {
    if (!intent.trim()) return;
    setLoading(true);
    setLastResponse(response);
    setResponse("");
    setTemplateMeta(null);

    try {
      const res = await fetch(`${BASE_URL}/write`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent, style, language, history }),
      });

      if (!res.ok)
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      const data = await res.json();
      setResponse(data.reply || "");
      setActiveTemplateId(null);
      setHistory((prev) => [
        ...prev,
        { role: "user", content: intent },
        { role: "assistant", content: data.reply || "" },
      ]);
    } catch (error: any) {
      console.error("Error in handleSubmit:", error);
      alert(`Generation failed: ${error.message}`);
      setResponse("");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitWithTemplate = async () => {
    if (!intent.trim()) return;
    setLoading(true);
    setLastResponse(response);
    setResponse("");
    setTemplateMeta(null);
    setActiveTemplateId(null);

    try {
      const res = await fetch(`${BASE_URL}/write_with_template`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent, style, language, history }),
      });

      if (!res.ok)
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      const data = await res.json();

      setResponse(data.reply || "");
      setTemplateMeta(data.template_meta || null);
      setActiveTemplateId(data.template_meta?.selected_template_id || null);
      setHistory((prev) => [
        ...prev,
        { role: "user", content: intent },
        { role: "assistant", content: data.reply || "" },
      ]);
    } catch (error: any) {
      console.error("Error in handleSubmitWithTemplate:", error);
      alert(`Generation failed: ${error.message}`);
      setResponse("");
    } finally {
      setLoading(false);
    }
  };

  const handleContinueEditing = async () => {
    if (!intent.trim() || !response.trim()) return;
    const previousResponse = response;
    const currentDraft = responseSections.draft || response;
    setLoading(true);
    setLastResponse(response);
    setResponse("");
    setTemplateMeta(null);

    try {
      const res = await fetch(`${BASE_URL}/continue_editing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          intent,
          style,
          language,
          history,
          current_draft: currentDraft,
          active_template_id: activeTemplateId,
        }),
      });

      if (!res.ok)
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      const data = await res.json();

      setResponse(data.reply || "");
      setTemplateMeta(data.template_meta || null);
      setActiveTemplateId(
        data.template_meta?.selected_template_id || activeTemplateId
      );
      setHistory((prev) => [
        ...prev,
        { role: "user", content: intent },
        { role: "assistant", content: data.reply || "" },
      ]);
    } catch (error: any) {
      console.error("Error in handleContinueEditing:", error);
      alert(`Continue editing failed: ${error.message}`);
      setResponse(previousResponse);
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

  const handleCopy = () => {
    if (response) {
      navigator.clipboard.writeText(response);
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

    setTemplateMessage("Saving template...");
    try {
      const res = await fetch(`${BASE_URL}/templates`, {
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

  return (
    <PageContainer className="page-home">
      <SideStack>
        <Panel>
          <DropdownHeader
            type="button"
            onClick={() => setTemplateLibraryOpen((open) => !open)}
            aria-expanded={templateLibraryOpen}
          >
            <h2>Template Library</h2>
            <span>
              {templates.length} {templateLibraryOpen ? "▲" : "▼"}
            </span>
          </DropdownHeader>
          {templateLibraryOpen && (
            <TemplateList>
              {templates.length === 0 && (
                <TemplateItem>
                  <h3>No templates loaded</h3>
                  <p>Create one below or start the backend to load templates.</p>
                </TemplateItem>
              )}
              {templates.map((template) => (
                <TemplateItem key={template.id}>
                  <TemplateRow
                    type="button"
                    onClick={() => setSelectedTemplate(template)}
                  >
                    <h3>{template.title}</h3>
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
          <ResetButton onClick={() => setHistory([])}>
            🔄 Reset Conversation
          </ResetButton>
        </ResponseControls>

        <GenerateButtons>
          <GenerateButton onClick={handleSubmitWithTemplate} disabled={loading}>
            Generate with Template
          </GenerateButton>
          <GenerateButton
            onClick={handleContinueEditing}
            disabled={loading || !response}
          >
            Continue Editing
          </GenerateButton>
          <GenerateButton onClick={handleSubmit} disabled={loading}>
            {loading
              ? "Writing... It takes around 1 minute"
              : "✨ Generate something wild"}
          </GenerateButton>
        </GenerateButtons>

        {response && (
          <ResponseBox>
            <ResponseTitle>Your Draft</ResponseTitle>
            {templateMeta && (
              <TemplateMetaBox>
                <strong>Template source</strong>
                {templateMeta.used_template ? (
                  <>
                    Used {templateMeta.selected_template || "a matched template"}
                    {typeof templateMeta.match_score === "number"
                      ? `, distance ${templateMeta.match_score.toFixed(2)}`
                      : ""}
                    . {templateMeta.reason}
                  </>
                ) : (
                  <>
                    No template used.{" "}
                    {templateMeta.reason || "The agent fell back to wild mode."}
                  </>
                )}
              </TemplateMetaBox>
            )}
            <ResponseText data-testid="main-reply">
              <ResponseTextHeader>
                <CopyButton onClick={handleCopy}>
                  {copied ? "Copied!" : "Copy"}
                </CopyButton>
              </ResponseTextHeader>
              <div>{responseSections.draft || response}</div>
            </ResponseText>
            {responseSections.changes && (
              <ChangesBox data-testid="changes-summary">
                <strong>Changes</strong>
                <p>{responseSections.changes}</p>
              </ChangesBox>
            )}
          </ResponseBox>
        )}
      </ChatBox>

      <HistoryPanel>
        <h2>Conversation History</h2>
        {history.map((msg, index) => (
          <Message key={index} className={msg.role} role={msg.role}>
            <strong>{msg.role === "user" ? "You" : "HeyWrite"}:</strong>
            <p>{msg.content}</p>
          </Message>
        ))}
      </HistoryPanel>

      {selectedTemplate && (
        <TemplateModalBackdrop onClick={() => setSelectedTemplate(null)}>
          <TemplateModal onClick={(event) => event.stopPropagation()}>
            <TemplateModalHeader>
              <div>
                <h3>{selectedTemplate.title}</h3>
                <p>{selectedTemplate.language}</p>
              </div>
              <ModalCloseButton
                type="button"
                onClick={() => setSelectedTemplate(null)}
              >
                Close
              </ModalCloseButton>
            </TemplateModalHeader>
            <TemplateModalBody>
              <p>{selectedTemplate.description}</p>
              <TemplateTagRow>
                {(selectedTemplate.tags || []).map((tag) => (
                  <TemplateTag key={tag}>{tag}</TemplateTag>
                ))}
              </TemplateTagRow>
              <pre>{selectedTemplate.content}</pre>
            </TemplateModalBody>
          </TemplateModal>
        </TemplateModalBackdrop>
      )}
    </PageContainer>
  );
};

export default Home;
