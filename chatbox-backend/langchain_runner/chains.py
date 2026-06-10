import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from .prompts import qa_prompt, summary_map_prompt, summary_reduce_prompt

# Ensure we load the repo root .env before instantiating the LLM
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


def build_llm(model: str = "deepseek-chat", temperature: float = 0.3, **kwargs: Any):
    api_key = kwargs.pop("api_key", None) or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set; cannot initialize DeepSeek client")
    return ChatDeepSeek(model=model, temperature=temperature, api_key=api_key, **kwargs)


def build_chain(prompt, model=None):
    """LCEL-style chain -> returns string output."""
    return prompt | (model or get_llm()) | StrOutputParser()


llm = None
qa_chain = None
summary_map_chain = None
summary_reduce_chain = None


def get_llm():
    global llm
    if llm is None:
        llm = build_llm()
    return llm


def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        qa_chain = build_chain(qa_prompt)
    return qa_chain


def get_summary_map_chain():
    global summary_map_chain
    if summary_map_chain is None:
        summary_map_chain = build_chain(summary_map_prompt)
    return summary_map_chain


def get_summary_reduce_chain():
    global summary_reduce_chain
    if summary_reduce_chain is None:
        summary_reduce_chain = build_chain(summary_reduce_prompt)
    return summary_reduce_chain
