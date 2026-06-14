import styled, { keyframes } from "styled-components";
import { StylesConfig } from "react-select";

const fadeUp = keyframes`
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
`;

// Tokens
const clay   = "#c45c2b";   // main accent — terracotta / burnt sienna
const clayDk = "#a0471f";   // hover
const ink    = "#1c1815";   // headings
const body   = "#3d3530";   // body text
const muted  = "#8a7d74";   // labels / placeholders
const paper  = "#fffdf9";   // card surface
const linen  = "#f5f0e8";   // page background
const border = "#e4dbd0";   // subtle warm border
const borderFocus = "#c45c2b";
const radiusMd = "8px";
const radiusSm = "7px";

// ─── Layout ───────────────────────────────────────────────────────────────────

export const PageContainer = styled.div<{ historyWidth?: number }>`
  display: grid;
  grid-template-columns: 300px minmax(560px, 660px) ${(props) =>
      props.historyWidth || 300}px;
  justify-content: center;
  align-items: flex-start;
  min-height: 100vh;
  padding: 3.5rem 2rem;
  gap: 1.75rem;

  @media (max-width: 1280px) {
    grid-template-columns: 260px minmax(500px, 1fr) 260px;
    gap: 1rem;
    padding: 2.5rem 1rem;
  }

  @media (max-width: 900px) {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.75rem 1rem;
    gap: 1.25rem;
  }
`;

// ─── Main Card ────────────────────────────────────────────────────────────────

export const ChatBox = styled.div`
  background: ${paper};
  padding: 2.75rem 2.75rem 2.5rem;
  border-radius: 4px;
  box-shadow:
    0 1px 3px rgba(60, 40, 20, 0.06),
    0 8px 32px rgba(60, 40, 20, 0.09);
  border: 1px solid ${border};
  width: 100%;
  animation: ${fadeUp} 0.4s ease both;

  @media (max-width: 768px) {
    padding: 1.75rem 1.5rem;
  }
`;

// ─── Title ────────────────────────────────────────────────────────────────────

export const Title = styled.h1`
  font-family: "Playfair Display", Georgia, serif;
  font-size: 2.4rem;
  font-weight: 700;
  text-align: center;
  color: ${ink};
  margin: 0 0 0.35rem;
  letter-spacing: -0.01em;
  line-height: 1.2;

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

export const Subtitle = styled.p`
  text-align: center;
  font-family: "Lora", Georgia, serif;
  font-style: italic;
  color: ${muted};
  font-size: 0.95rem;
  margin: 0 0 2.25rem;
  line-height: 1.5;
`;

// ─── Textarea ─────────────────────────────────────────────────────────────────

export const IntentTextarea = styled.textarea`
  width: 100%;
  box-sizing: border-box;
  padding: 1rem 1.2rem;
  font-size: 0.975rem;
  font-family: "Lora", Georgia, serif;
  line-height: 1.7;
  color: ${body};
  background: ${linen};
  border: 1.5px solid ${border};
  border-radius: 3px;
  resize: vertical;
  outline: none;
  min-height: 130px;
  margin-bottom: 1.5rem;
  transition: border-color 0.2s ease, background 0.2s ease;

  &::placeholder {
    color: ${muted};
    font-style: italic;
  }

  &:focus {
    border-color: ${borderFocus};
    background: #fffefb;
  }

  @media (max-width: 768px) {
    font-size: 0.9rem;
    padding: 0.875rem 1rem;
  }
`;

// ─── Selectors ────────────────────────────────────────────────────────────────

export const Selectors = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-bottom: 0.5rem;

  @media (max-width: 480px) {
    grid-template-columns: 1fr;
  }
`;

export const SelectGroup = styled.div`
  display: flex;
  flex-direction: column;
`;

export const Label = styled.label`
  display: block;
  margin-bottom: 0.4rem;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: ${muted};
`;

export const StyledSelect: StylesConfig<{ value: string; label: string }, false> = {
  container: (base) => ({ ...base, width: "100%" }),
  control: (base, state) => ({
    ...base,
    width: "100%",
    padding: "0.05rem 0.1rem",
    fontSize: "0.9rem",
    fontFamily: '"Inter", sans-serif',
    borderRadius: "3px",
    borderColor: state.isFocused ? clay : border,
    boxShadow: state.isFocused ? `0 0 0 2px rgba(196, 92, 43, 0.15)` : "none",
    backgroundColor: linen,
    transition: "border-color 0.2s",
    "&:hover": { borderColor: "#c9a98a" },
    "@media (max-width: 768px)": { fontSize: "0.85rem" },
  }),
  option: (base, state) => ({
    ...base,
    fontSize: "0.9rem",
    fontFamily: '"Inter", sans-serif',
    backgroundColor: state.isSelected ? clay : state.isFocused ? "#f5ede4" : paper,
    color: state.isSelected ? "#fff" : body,
    cursor: "pointer",
  }),
  menu: (base) => ({ ...base, borderRadius: "3px", border: `1px solid ${border}`, boxShadow: "0 4px 16px rgba(60,40,20,0.1)" }),
  menuList: (base) => ({ ...base, padding: "0.25rem" }),
  dropdownIndicator: (base) => ({ ...base, color: muted }),
  indicatorSeparator: () => ({ display: "none" }),
  singleValue: (base) => ({ ...base, color: body }),
};

// ─── Controls Row ─────────────────────────────────────────────────────────────

export const ResponseControls = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1.25rem;
`;

// ─── Buttons ──────────────────────────────────────────────────────────────────

export const Button = styled.button`
  font-family: "Inter", sans-serif;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 0.4rem 0.875rem;
  border-radius: ${radiusSm};
  border: 1px solid ${border};
  background: transparent;
  color: ${muted};
  cursor: pointer;
  transition:
    background 0.15s,
    border-color 0.15s,
    color 0.15s,
    box-shadow 0.15s,
    transform 0.1s;

  &:hover {
    background: #f0e9de;
    border-color: #c9a98a;
    color: ${body};
    box-shadow: 0 2px 8px rgba(60, 40, 20, 0.08);
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }

  @media (max-width: 768px) {
    font-size: 0.75rem;
    padding: 0.35rem 0.7rem;
  }
`;

export const UndoButton = styled(Button)``;
export const ResetButton = styled(Button)``;

export const GenerateButtons = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.875rem;
  margin-top: 1.75rem;

  @media (max-width: 480px) {
    grid-template-columns: 1fr;
  }
`;

export const GenerateButton = styled.button`
  font-family: "Inter", sans-serif;
  width: 100%;
  font-size: 0.9rem;
  font-weight: 600;
  padding: 0.8rem 1rem;
  border-radius: ${radiusMd};
  cursor: pointer;
  transition:
    background 0.15s ease,
    border-color 0.15s ease,
    transform 0.1s ease,
    box-shadow 0.15s ease;
  letter-spacing: 0.01em;

  /* Default: outlined */
  background: #fffaf3;
  color: ${clay};
  border: 1.5px solid ${clay};
  box-shadow: 0 1px 4px rgba(60, 40, 20, 0.05);

  &:hover:not(:disabled) {
    background: #fdf1ea;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(196, 92, 43, 0.14);
  }

  /* Last child: filled */
  &:last-child {
    background: ${clay};
    color: #fffdf9;
    border-color: ${clay};
    box-shadow: 0 4px 14px rgba(196, 92, 43, 0.25);

    &:hover:not(:disabled) {
      background: ${clayDk};
      border-color: ${clayDk};
      box-shadow: 0 4px 14px rgba(196, 92, 43, 0.35);
      transform: translateY(-1px);
    }
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  @media (max-width: 768px) {
    font-size: 0.875rem;
    padding: 0.7rem 0.875rem;
  }
`;

// ─── Response Area ────────────────────────────────────────────────────────────

export const ResponseBox = styled.div`
  margin-top: 2.25rem;
  border-top: 1px solid ${border};
  padding-top: 1.75rem;
  animation: ${fadeUp} 0.3s ease both;
`;

export const ResponseTitle = styled.h2`
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: ${muted};
  margin: 0 0 0.875rem;
`;

export const CopyButton = styled.button`
  font-family: "Inter", sans-serif;
  font-size: 0.75rem;
  font-weight: 500;
  padding: 0.3rem 0.75rem;
  border-radius: ${radiusSm};
  border: 1px solid ${border};
  background: transparent;
  color: ${muted};
  cursor: pointer;
  transition:
    background 0.15s,
    border-color 0.15s,
    color 0.15s,
    box-shadow 0.15s,
    transform 0.1s;

  &:hover {
    background: #f0e9de;
    border-color: #c9a98a;
    color: ${body};
    box-shadow: 0 2px 8px rgba(60, 40, 20, 0.08);
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }
`;

export const ResponseText = styled.div`
  display: grid;
  gap: 0.875rem;
  font-family: "Lora", Georgia, serif;
  font-size: 0.975rem;
  line-height: 1.8;
  color: ${body};
  white-space: pre-line;
  padding: 1.25rem 1.5rem;
  background: ${linen};
  border: 1px solid ${border};
  border-left: 3px solid ${clay};
  border-radius: 2px;

  p {
    margin: 0;
    white-space: pre-line;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    overflow-x: auto;
    display: block;
    font-family: "Inter", sans-serif;
    font-size: 0.84rem;
    line-height: 1.45;
    white-space: normal;
  }

  th,
  td {
    min-width: 120px;
    padding: 0.55rem 0.65rem;
    border: 1px solid ${border};
    text-align: left;
    vertical-align: top;
  }

  th {
    background: #fffaf3;
    color: ${ink};
    font-weight: 700;
  }

  td {
    background: ${paper};
  }

  @media (max-width: 768px) {
    font-size: 0.9rem;
    padding: 1rem 1.1rem;
    line-height: 1.7;
  }
`;

export const ResponseTextHeader = styled.div`
  display: flex;
  justify-content: flex-end;
  align-items: center;
`;

export const ChangesBox = styled.div`
  margin-top: 0.875rem;
  padding: 0.85rem 1rem;
  background: #fffaf3;
  border: 1px solid ${border};
  border-left: 3px solid ${muted};
  border-radius: 2px;
  font-family: "Inter", sans-serif;
  font-size: 0.82rem;
  line-height: 1.55;
  color: ${body};
  white-space: pre-line;

  strong {
    display: block;
    margin-bottom: 0.35rem;
    font-size: 0.68rem;
    color: ${muted};
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  p {
    margin: 0;
  }
`;

export const AgentTraceBox = styled.div`
  margin-top: 0.875rem;
  padding: 0.85rem 1rem;
  background: #fffdf9;
  border: 1px solid ${border};
  border-radius: 2px;
  font-family: "Inter", sans-serif;
  font-size: 0.78rem;
  line-height: 1.45;
  color: ${muted};

  strong {
    display: block;
    margin-bottom: 0.45rem;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  p {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.2rem 0;
  }

  span {
    color: ${body};
    font-weight: 600;
  }
`;

export const TemplateMetaBox = styled.div`
  clear: both;
  margin-bottom: 0.875rem;
  padding: 0.8rem 1rem;
  border: 1px solid ${border};
  border-left: 3px solid ${clay};
  background: #fffaf3;
  font-family: "Inter", sans-serif;
  font-size: 0.8rem;
  color: ${body};
  line-height: 1.5;

  strong {
    display: block;
    font-size: 0.68rem;
    color: ${muted};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.25rem;
  }
`;

// ─── History Panel ────────────────────────────────────────────────────────────

export const HistoryPanel = styled.div`
  background: ${paper};
  padding: 1.75rem;
  border-radius: 4px;
  box-shadow:
    0 1px 3px rgba(60, 40, 20, 0.05),
    0 6px 20px rgba(60, 40, 20, 0.07);
  border: 1px solid ${border};
  width: 100%;
  min-width: 240px;
  max-width: 520px;
  flex-shrink: 0;
  box-sizing: border-box;
  height: fit-content;
  position: sticky;
  top: 3rem;
  isolation: isolate;
  animation: ${fadeUp} 0.4s ease 0.1s both;

  @media (max-width: 1280px) {
    min-width: 220px;
  }

  h2 {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: ${muted};
    margin: 0 0 1.1rem;
    font-family: "Inter", sans-serif;
  }

  @media (max-width: 900px) {
    width: 100%;
    max-width: 660px;
    position: static;
    padding: 1.25rem;
  }

  > p {
    margin: 0;
    color: ${muted};
    font-family: "Inter", sans-serif;
    font-size: 0.82rem;
    line-height: 1.5;
  }
`;

export const HistoryResizeHandle = styled.div`
  position: absolute;
  top: 0.75rem;
  bottom: 0.75rem;
  left: -0.55rem;
  width: 0.7rem;
  cursor: col-resize;
  z-index: 2;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0.32rem;
    width: 2px;
    border-radius: 999px;
    background: transparent;
    transition: background 0.15s ease;
  }

  &:hover::before {
    background: ${clay};
  }

  @media (max-width: 900px) {
    display: none;
  }
`;

export const SessionList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  max-height: calc(100vh - 8rem);
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 0.25rem;

  @media (max-width: 900px) {
    max-height: 420px;
  }
`;

export const SessionItem = styled.div`
  border: 1px solid ${border};
  border-left: 3px solid ${border};
  background: ${linen};
  border-radius: ${radiusMd};
  overflow: hidden;

  &[data-active="true"] {
    border-left-color: ${clay};
    background: #fffaf3;
  }
`;

export const SessionRow = styled.button`
  width: 100%;
  border: 0;
  background: transparent;
  padding: 0.65rem 0.75rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  text-align: left;
  cursor: pointer;
  color: ${body};
  font-family: "Inter", sans-serif;
  transition: background 0.15s ease;

  &:hover {
    background: #fffaf3;
  }

  h3 {
    margin: 0 0 0.2rem;
    color: ${ink};
    font-size: 0.82rem;
    line-height: 1.3;
    font-family: "Inter", sans-serif;
    font-weight: 600;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  span {
    flex-shrink: 0;
    color: ${muted};
    font-size: 0.68rem;
  }
`;

export const SessionMeta = styled.div`
  color: ${muted};
  font-size: 0.68rem;
  line-height: 1.35;
`;

export const SessionMessages = styled.div`
  border-top: 1px solid ${border};
  background: ${paper};
  padding: 0.7rem;
  max-height: 360px;
  overflow-y: auto;
  overflow-x: hidden;
  overscroll-behavior: contain;

  @media (max-width: 900px) {
    max-height: 300px;
  }

  p {
    white-space: pre-line;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    display: block;
    overflow-x: auto;
    margin-top: 0.35rem;
    font-size: 0.74rem;
  }

  th,
  td {
    min-width: 90px;
    padding: 0.4rem 0.5rem;
    border: 1px solid ${border};
    text-align: left;
    vertical-align: top;
  }

  th {
    background: #fffaf3;
    color: ${ink};
  }
`;

export const SessionOpenButton = styled.button`
  width: 100%;
  margin-bottom: 0.65rem;
  padding: 0.45rem 0.65rem;
  border-radius: ${radiusSm};
  border: 1px solid ${border};
  background: #fffaf3;
  color: ${clay};
  font-family: "Inter", sans-serif;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  transition:
    background 0.15s,
    border-color 0.15s,
    box-shadow 0.15s,
    transform 0.1s;

  &:hover {
    border-color: ${clay};
    background: #fdf1ea;
    box-shadow: 0 2px 8px rgba(196, 92, 43, 0.12);
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }
`;

export const SideStack = styled.div`
  width: 300px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;

  @media (max-width: 1280px) {
    width: 260px;
  }

  @media (max-width: 900px) {
    width: 100%;
    max-width: 660px;
  }
`;

export const Panel = styled.div`
  background: ${paper};
  padding: 1.5rem;
  border-radius: 4px;
  box-shadow:
    0 1px 3px rgba(60, 40, 20, 0.05),
    0 6px 20px rgba(60, 40, 20, 0.07);
  border: 1px solid ${border};
  box-sizing: border-box;
  animation: ${fadeUp} 0.4s ease 0.1s both;

  h2 {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: ${muted};
    margin: 0 0 1.1rem;
    font-family: "Inter", sans-serif;
  }
`;

export const DropdownHeader = styled.button`
  width: 100%;
  border: 0;
  background: transparent;
  padding: 0;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  color: ${muted};
  font-family: "Inter", sans-serif;
  border-radius: ${radiusSm};

  h2 {
    margin: 0;
  }

  span {
    font-size: 0.75rem;
    color: ${muted};
  }
`;

export const TemplateList = styled.div`
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  height: 320px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 0.25rem;
`;

export const TemplateItem = styled.div`
  border: 1px solid ${border};
  background: ${linen};
  border-radius: ${radiusMd};
  overflow: hidden;
`;

export const TemplateRow = styled.button`
  width: 100%;
  border: 0;
  background: transparent;
  padding: 0.55rem 0.7rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  text-align: left;
  cursor: pointer;
  color: ${body};
  font-family: "Inter", sans-serif;
  transition: background 0.15s ease;

  &:hover {
    background: #fffaf3;
  }

  h3 {
    margin: 0;
    color: ${ink};
    font-size: 0.82rem;
    font-family: "Inter", sans-serif;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  span {
    flex-shrink: 0;
    color: ${muted};
    font-size: 0.68rem;
  }
`;

export const TemplateDetail = styled.div`
  padding: 0.7rem;
  border-top: 1px solid ${border};
  background: ${paper};

  p {
    margin: 0.25rem 0;
    color: ${body};
    font-size: 0.76rem;
    line-height: 1.45;
  }

  pre {
    margin: 0.55rem 0 0;
    max-height: 160px;
    overflow: auto;
    white-space: pre-wrap;
    font-family: "Lora", Georgia, serif;
    font-size: 0.74rem;
    line-height: 1.5;
    color: ${body};
    background: ${linen};
    border: 1px solid ${border};
    padding: 0.6rem;
    border-radius: 2px;
  }
`;

export const TemplateModalBackdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  background: rgba(28, 24, 21, 0.36);
`;

export const TemplateModal = styled.div`
  width: min(760px, 100%);
  max-height: min(78vh, 760px);
  overflow: hidden;
  background: ${paper};
  border: 1px solid ${border};
  border-radius: 4px;
  box-shadow:
    0 8px 28px rgba(28, 24, 21, 0.18),
    0 24px 70px rgba(28, 24, 21, 0.22);
  display: flex;
  flex-direction: column;
`;

export const TemplateModalHeader = styled.div`
  padding: 1.25rem 1.4rem 1rem;
  border-bottom: 1px solid ${border};
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;

  h3 {
    margin: 0 0 0.35rem;
    color: ${ink};
    font-family: "Inter", sans-serif;
    font-size: 1rem;
  }

  p {
    margin: 0;
    color: ${muted};
    font-family: "Inter", sans-serif;
    font-size: 0.78rem;
  }
`;

export const TemplateModalBody = styled.div`
  padding: 1.2rem 1.4rem 1.4rem;
  overflow: auto;

  p {
    margin: 0 0 0.8rem;
    color: ${body};
    font-family: "Inter", sans-serif;
    font-size: 0.84rem;
    line-height: 1.55;
    white-space: pre-line;
  }

  pre,
  table {
    margin: 0.9rem 0 0;
    font-size: 0.9rem;
    line-height: 1.65;
    color: ${body};
  }

  pre {
    white-space: pre-wrap;
    font-family: "Lora", Georgia, serif;
    background: ${linen};
    border: 1px solid ${border};
    padding: 1rem;
    border-radius: 2px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    display: block;
    overflow-x: auto;
    font-family: "Inter", sans-serif;
  }

  th,
  td {
    min-width: 130px;
    padding: 0.55rem 0.65rem;
    border: 1px solid ${border};
    text-align: left;
    vertical-align: top;
  }

  th {
    background: #fffaf3;
    color: ${ink};
    font-weight: 700;
  }
`;

export const ModalCloseButton = styled.button`
  flex-shrink: 0;
  border: 1px solid ${border};
  background: transparent;
  color: ${muted};
  border-radius: ${radiusSm};
  padding: 0.25rem 0.55rem;
  cursor: pointer;
  font-family: "Inter", sans-serif;
  font-size: 0.76rem;
  transition:
    background 0.15s,
    border-color 0.15s,
    color 0.15s,
    box-shadow 0.15s,
    transform 0.1s;

  &:hover {
    background: #f0e9de;
    border-color: #c9a98a;
    color: ${body};
    box-shadow: 0 2px 8px rgba(60, 40, 20, 0.08);
    transform: translateY(-1px);
  }

  &:active {
    transform: translateY(0);
  }
`;

export const TemplateTagRow = styled.div`
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
  margin-top: 0.45rem;
`;

export const TemplateTag = styled.span`
  padding: 0.15rem 0.4rem;
  border: 1px solid ${border};
  background: ${paper};
  color: ${muted};
  font-size: 0.68rem;
  border-radius: 2px;
`;

export const TemplateForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
`;

export const TemplateInput = styled.input`
  width: 100%;
  box-sizing: border-box;
  padding: 0.65rem 0.75rem;
  border: 1px solid ${border};
  background: ${linen};
  color: ${body};
  border-radius: 2px;
  font-family: "Inter", sans-serif;
  font-size: 0.82rem;
  outline: none;

  &:focus {
    border-color: ${clay};
    background: #fffefb;
  }
`;

export const TemplateTextarea = styled.textarea`
  width: 100%;
  box-sizing: border-box;
  min-height: 120px;
  resize: vertical;
  padding: 0.65rem 0.75rem;
  border: 1px solid ${border};
  background: ${linen};
  color: ${body};
  border-radius: 2px;
  font-family: "Lora", Georgia, serif;
  font-size: 0.85rem;
  line-height: 1.55;
  outline: none;

  &:focus {
    border-color: ${clay};
    background: #fffefb;
  }
`;

interface MessageProps {
  role: "user" | "assistant";
}

export const Message = styled.div<MessageProps>`
  margin-bottom: 0.625rem;
  padding: 0.75rem 0.875rem;
  border-radius: 2px;
  font-size: 0.85rem;
  font-family: "Inter", sans-serif;
  line-height: 1.55;
  animation: ${fadeUp} 0.3s ease both;
  border-left: 2px solid
    ${(props) => (props.role === "user" ? clay : "#8a7d74")};
  background: ${(props) =>
    props.role === "user" ? "#fdf5ee" : "#f7f4ee"};

  strong {
    display: block;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
    color: ${(props) => (props.role === "user" ? clay : muted)};
  }

  p {
    margin: 0;
    color: ${body};
  }
`;
