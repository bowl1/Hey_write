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

// ─── Layout ───────────────────────────────────────────────────────────────────

export const PageContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: flex-start;
  min-height: 100vh;
  padding: 3.5rem 2rem;
  gap: 1.75rem;

  @media (max-width: 900px) {
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
  max-width: 660px;
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
  border-radius: 3px;
  border: 1px solid ${border};
  background: transparent;
  color: ${muted};
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;

  &:hover {
    background: #f0e9de;
    border-color: #c9a98a;
    color: ${body};
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
  grid-template-columns: 1fr 1fr;
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
  border-radius: 3px;
  cursor: pointer;
  transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
  letter-spacing: 0.01em;

  /* Default: outlined */
  background: transparent;
  color: ${clay};
  border: 1.5px solid ${clay};

  &:hover:not(:disabled) {
    background: #fdf1ea;
    transform: translateY(-1px);
  }

  /* Last child: filled */
  &:last-child {
    background: ${clay};
    color: #fffdf9;
    border-color: ${clay};
    box-shadow: 0 2px 8px rgba(196, 92, 43, 0.25);

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
  float: right;
  font-family: "Inter", sans-serif;
  font-size: 0.75rem;
  font-weight: 500;
  padding: 0.3rem 0.75rem;
  border-radius: 3px;
  border: 1px solid ${border};
  background: transparent;
  color: ${muted};
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
  margin-bottom: 0.625rem;

  &:hover {
    background: #f0e9de;
    border-color: #c9a98a;
    color: ${body};
  }
`;

export const ResponseText = styled.div`
  font-family: "Lora", Georgia, serif;
  font-size: 0.975rem;
  line-height: 1.8;
  color: ${body};
  white-space: pre-line;
  clear: both;
  padding: 1.25rem 1.5rem;
  background: ${linen};
  border: 1px solid ${border};
  border-left: 3px solid ${clay};
  border-radius: 2px;

  @media (max-width: 768px) {
    font-size: 0.9rem;
    padding: 1rem 1.1rem;
    line-height: 1.7;
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
  width: 300px;
  flex-shrink: 0;
  box-sizing: border-box;
  height: fit-content;
  position: sticky;
  top: 3rem;
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

  @media (max-width: 900px) {
    width: 100%;
    max-width: 660px;
    position: static;
    padding: 1.25rem;
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
