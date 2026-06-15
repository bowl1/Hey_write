# ✨ HeyWrite - AI Smart Writing Assistant

HeyWrite is a smart AI-powered writing assistant that helps you draft professional content based on your intent, tone, and language — all with just one sentence.

Now enhanced with Retrieval-Augmented Generation (RAG), custom templates, multi-turn memory, and more — HeyWrite makes writing faster, smarter, and more personalized.

---

## 🌐 Live Demo

- https://hey-write.vercel.app/

---

## 🚀 Features

-  Generate instant writing drafts from your one-sentence intent
- Choose between two generation modes:
    - Generate with templates: AI automatically matches your input to a relevant predefined template using vector search
    - Generate something wild: Freely generate content without relying on templates
- Your intent is automatically matched to the most relevant template using vector search. If no suitable template is found, the app suggests using wild mode for freeform generation.

- Maintains **multi-turn conversation history**
- Revisit previous results and modify based on that
- Clear chat history and start a new conversation
- Summarize and highlight the changes made between the previous version and the newly generated content
- Control tone and style: Formal, Casual, Polite Push, Concise & Direct, Humorous, or Creative
- Supports **English**, **Chinese**, and **Danish**
- One-click copy of generated content

---

## ⚙️ Tech Stack

| Category        | Technology                            |
|----------------|----------------------------------------|
| **Frontend**    | React + TypeScript (deployed via Vercel)|
| **Backend**     | FastAPI (deployed on Render) |
| **AI Model**    | DeepSeek Chat API                     |
| **Embedding**   | OpenAI Embeddings (`text-embedding-3-small`)  |
| **Template Store**| Postgres                             |
| **Vector Store**| pgvector                              |
| **Agent Framework**| LangGraph controlled agent loop + LangChain model adapters |
| **Deployment**  | Render + Docker+ GitHub Actions |
---


## 🧱 Template DB / Vector Index

Templates and template embeddings are stored in Postgres with pgvector.
Set these environment variables before running template mode:

```bash
cd chatbox-backend
export DATABASE_URL="postgresql://user:password@localhost:5432/heywrite"
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export AGENT_LLM_EVALUATOR_ENABLED="true"
export TEMPLATE_MIN_MATCH_SCORE="0.65"
```

To initialize the `vector` extension, create template indexes, seed bundled JSON templates into Postgres, and refresh template embeddings:

```bash
cd chatbox-backend
./.venv/bin/python langchain_runner/build_vectorstore.py
```


## 📂 Architecture Overview

The main AI endpoint is:

```text
POST /agent/run
```

The frontend sends an explicit action:

```text
new_task          -> Generate with Template
continue_editing  -> Continue Editing
wild              -> Generate something wild
```

LangGraph agent loop:

```text
load_session
  -> planner
  -> tool node
     - retrieve_template
     - generate_draft
     - load_active_template
     - revise_draft
     - generate_wild
  -> observe
  -> planner
  -> finalize
  -> evaluator
  -> if failed: revise_with_feedback -> evaluator
  -> persist_state
```

The planner loops until the draft is complete or the step limit is reached.
Action constraints still apply:

- `new_task` can retrieve templates and generate a new draft.
- `continue_editing` loads the active template and revises the current draft without reselecting templates.
- `wild` generates without template retrieval.

Template mode uses hybrid retrieval:

```text
query
  -> pgvector semantic top-k
  -> BM25 keyword top-k
  -> merge candidates
  -> weighted rerank
  -> return top templates with scores and matched terms
```

The agent only accepts a template when its hybrid match score is at least `TEMPLATE_MIN_MATCH_SCORE`, which defaults to `0.65`.

Agent state and observability are stored in Postgres:

```text
agent_sessions
agent_runs
messages
```

The frontend stores the current `session_id` in `localStorage` and restores it with:

```text
GET /agent/session/{session_id}
```

The right-side Conversation History is session-based and loads recent sessions with:

```text
GET /agent/sessions
```

Users can expand a session to inspect its messages, then open that session to recover the active draft, active template, message history, last trace, and last evaluator result as long as the same backend database is still available.
Evaluation and local test sessions using prefixes such as `eval-`, `test-`, and `smoke-` are filtered out of this session list.

Each response includes `reply`, `template_meta`, `evaluation`, `state`, and `trace` so the UI can show which template was used, why it was used, what the agent did, and whether the output passed evaluator checks.

Evaluator checks:

- Preserves original structure when revising an existing draft.
- Completes the user's request.
- Uses the correct template behavior for the selected action.
- Includes a `Changes` section.
- Avoids obvious unrelated fabrication.

The evaluator combines deterministic guardrails with optional LLM-as-judge:

- Deterministic checks handle hard constraints such as `Changes`, template action behavior, and explicitly requested names or numbers.
- LLM-as-judge handles semantic checks such as whether the user request was actually completed and whether the revision stayed on scope.
- If the evaluator fails, the graph performs one `revise_with_feedback` retry before persisting the final result.

Run eval cases:

```bash
cd chatbox-backend
./.venv/bin/python tests/evals/run_writing_agent_eval.py
```

## 🖥️ Local Run

Prerequisites:
- Python 3
- Node.js + npm
- Postgres with pgvector

Run frontend + backend with one command:

```bash
./run_local.sh
```

Script behavior:
- Auto-creates `chatbox-backend/.venv` if missing
- Starts backend at `http://127.0.0.1:8000`
- Starts frontend via CRA dev server

Stop:
- Close the frontend terminal session (the script then stops backend as well)


## 📸 Screenshots

### Web Demo

![Demo](https://github.com/user-attachments/assets/bca2437f-75ee-4b74-9d54-91a74065c3bc)

---


### Web UI

<img src="images/web0.png" alt="Web UI 0" width="1000">  

#### 📄 User Case: Project Weekly Report  
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
  <img src="images/web1.png" alt="Web UI 1" width="1000">  
  Template Library  
  <img src="images/web3.png" alt="Web UI 3" width="1000">   
  Agent trace & Evaluation 
    <img src="images/web2.png" alt="Web UI 2" width="1000">  
</div>

#### ⚠️ No Matched Template (using "Generate with templates" button)  
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
  <img src="images/web4.png" alt="Web UI 4" width="1000"> 
   Agent trace & Evaluation 
  <img src="images/web5.png" alt="Web UI 4" width="1000"> 
</div>


---
