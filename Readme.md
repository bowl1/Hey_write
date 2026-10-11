# ✨ HeyWrite - AI Smart Writing Assistant

HeyWrite is a smart AI-powered writing assistant that helps you draft professional content based on your intent, tone, and language — all with just one sentence.

Now enhanced with Retrieval-Augmented Generation (RAG), custom templates, multi-turn memory, and more — HeyWrite makes writing faster, smarter, and more personalized.

---

## 🌐 Live Demo

- https://hey-write.vercel.app/

---

## 📸 Screenshots

### Web Demo

https://github.com/user-attachments/assets/bca2437f-75ee-4b74-9d54-91a74065c3bc

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
| **Observability** | Langfuse — agent traces, model calls, evaluation results, token usage, and latency |
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
export TEMPLATE_MIN_MATCH_SCORE="0.60"
```

To initialize the `vector` extension, create template indexes, seed bundled JSON templates into Postgres, and refresh template embeddings:

```bash
cd chatbox-backend
./.venv/bin/python langchain_runner/build_vectorstore.py --tenant YOUR_SUPABASE_USER_UUID
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
  -> pgvector semantic top-k over template metadata, structure, and content
  -> BM25 keyword top-k over template metadata, structure, and content
  -> merge candidates
  -> weighted rerank
  -> return top templates with scores and matched terms
```

The agent only accepts a template when its hybrid match score is at least `TEMPLATE_MIN_MATCH_SCORE`, which defaults to `0.60`.

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

Generation, revisions, and the LLM evaluator receive bounded conversation context: up to 20 recent user/assistant messages, 4,000 characters per message, and 12,000 content characters total. Existing sessions load their history from the tenant's database instead of trusting the client copy. The latest request overrides conflicting older instructions.

Only nonempty drafts that pass evaluation and are not blocked update the saved draft and active template. Failed candidates remain in `agent_runs` for inspection, while the previous valid draft and its settings are preserved. Rejected candidates do not enter the assistant conversation memory. The API reports `state.draft_accepted` and `state.current_draft`; the UI retains the saved draft and explains a rejected attempt, including after reopening the session.

## Langfuse tracing

The backend uses the [official LangChain/LangGraph callback integration](https://langfuse.com/integrations/frameworks/langchain) with Langfuse Python SDK v4. Each `/agent/run` produces a `heywrite-agent` trace containing graph nodes, nested model calls, evaluation output, and retries. Traces are grouped by the existing session ID and include the database `run_id` for correlation. LLM observations capture token usage and latency when reported by the model provider.

Install the backend requirements, then add project keys from Langfuse Settings to the root `.env` (or `chatbox-backend/.env`):

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENABLED=true
```

Use the URL shown in your project's settings for your region or self-hosted instance. Configure the same variables in Render for the deployed backend. Restart the backend after changing configuration, generate a draft, then inspect Traces and Sessions in Langfuse.

Without both keys, tracing is disabled automatically. Set `LANGFUSE_TRACING_ENABLED=false` to disable it explicitly. Tracing records prompts, conversation history, drafts, and graph state in the configured Langfuse project. Delivery is asynchronous, and the backend drains pending events on graceful shutdown. Template embedding calls are direct SDK calls and are not captured as separate model observations by this graph callback.

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

## 🧪 Testing

### Automated checks

Backend tests cover authentication, tenant isolation, conversation memory, template selection, evaluation and retries, and Langfuse integration. Frontend tests cover sign-in, template browsing and selection, highlighted templates, and draft interactions.

```bash
# Backend: offline regression tests
cd chatbox-backend
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Optional PostgreSQL integration test (test data is rolled back)
RUN_DB_TESTS=1 ./.venv/bin/python -m unittest discover -s tests -p 'test_tenancy_db.py'

# Frontend
cd ../chatbox-frontend
CI=true npm test -- --watchAll=false --runInBand
npm run build
```

### Template matching quality

The benchmark covers **34 labeled English/Chinese requests**: 30 public-template requests, private-template priority, ambiguous private matches, and two unrelated requests that should be rejected.

It uses real OpenAI embeddings and PostgreSQL retrieval, with draft generation mocked. It checks the selected template and whether the agent should generate, ask the user to choose, or reject the request. Finding the correct template in the top three alone does not count as a successful selection.

```bash
cd chatbox-backend
.venv/bin/python -m tests.evals.run_template_retrieval_eval --output /tmp/heywrite-template-quality.json
```

Fixtures run in an isolated workspace and are rolled back. Embedding calls incur API costs; failed quality cases produce a nonzero exit status. Add labeled cases in `chatbox-backend/tests/evals/template_retrieval_cases.jsonl` to expand coverage.

### Verified results

| Check | Passed / executed | Pass rate | Notes |
| --- | ---: | ---: | --- |
| Backend regression tests | 27 / 27 | 100% | 1 optional database test skipped |
| Frontend tests | 20 / 20 | 100% | 3 suites passed |
| Frontend production build | 1 / 1 | 100% | Build succeeded |
| Template decisions, threshold 0.65 | 28 / 34 | 82.4% | Recorded retrieval baseline |
| Template decisions, threshold 0.60 | 33 / 34 | 97.1% | Recorded-score replay; current default |

The **0.65 baseline** uses real retrieval results; the **0.60 result** replays those recorded candidate scores through the selection gate. It is not a fresh embedding run. The remaining failure selects a meeting agenda for an English meeting-summary request.

| Retrieval metric | Correct / total | Rate |
| --- | ---: | ---: |
| Top-1 template accuracy | 31 / 32 | 96.9% |
| Recall@3 | 32 / 32 | 100% |
| Private priority and ambiguous-choice behavior | 2 / 2 | 100% |
| Unrelated-request rejection | 2 / 2 | 100% |

See the [baseline report](chatbox-backend/tests/evals/reports/template-retrieval-baseline.md) and [recorded results](chatbox-backend/tests/evals/reports/template-retrieval-baseline.json). These results measure template matching, not generated prose quality.

### Generated content evaluation

For a running backend, use a confirmed user's access token to run the writing-agent cases:

```bash
cd chatbox-backend
EVAL_ACCESS_TOKEN="your-user-access-token" ./.venv/bin/python tests/evals/run_writing_agent_eval.py
```

This exercises actual generation and revision. Review request fulfillment, template structure, tone/language, preserved facts, and the `Changes` section alongside the returned evaluator result and Langfuse trace. No prose-quality pass rate is claimed in the tables above.
