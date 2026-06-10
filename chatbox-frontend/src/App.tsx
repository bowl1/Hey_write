import React, { useState } from "react";
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
  HistoryPanel,
  Message,
} from "./App-style";

import {
  toneOptions,
  languageOptions,
  CustomSelect,
} from "./components/option";

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

  const BASE_URL =
    process.env.NODE_ENV === "development"
      ? "http://localhost:8000"
      : process.env.REACT_APP_API_URL;

  const handleSubmit = async () => {
    if (!intent.trim()) return;
    setLoading(true);
    setLastResponse(response);
    setResponse("");

    const res = await fetch(`${BASE_URL}/write`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intent, style, language, history }),
    });

    const data = await res.json();
    setResponse(data.reply);
    setHistory((prev) => [
      ...prev,
      { role: "user", content: intent },
      { role: "assistant", content: data.reply },
    ]);
    setLoading(false);
  };

  const handleSubmitWithTemplate = async () => {
    if (!intent.trim()) return;
    setLoading(true);
    setLastResponse(response);
    setResponse("");

    try {
      const res = await fetch(`${BASE_URL}/write_with_template`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent, style, language, history }),
      });

      if (!res.ok)
        throw new Error(`Server returned ${res.status}: ${await res.text()}`);
      const data = await res.json();

      setResponse(data.reply);
      setHistory((prev) => [
        ...prev,
        { role: "user", content: intent },
        { role: "assistant", content: data.reply },
      ]);
    } catch (error: any) {
      console.error("Error in handleSubmitWithTemplate:", error);
      alert(`Generation failed: ${error.message}`);
      setResponse("");
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

  return (
    <PageContainer className="page-home">
      <ChatBox>
        <Title>Hey Write!</Title>
        <Subtitle>Tell me what you'd like to say. I'll help you say it well.</Subtitle>

        <IntentTextarea
          rows={5}
          placeholder="Describe what you want to write, e.g., an invitation email for a meeting"
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
          <GenerateButton onClick={handleSubmit} disabled={loading}>
            {loading
              ? "Writing... It takes around 1 minute"
              : "✨ Generate something wild"}
          </GenerateButton>
        </GenerateButtons>

        {response && (
          <ResponseBox>
            <ResponseTitle>Your Draft</ResponseTitle>
            <CopyButton onClick={handleCopy}>
              {copied ? "Copied!" : "Copy"}
            </CopyButton>
            <ResponseText data-testid="main-reply">{response}</ResponseText>
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
    </PageContainer>
  );
};

export default Home;
