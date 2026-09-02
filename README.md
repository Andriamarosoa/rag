# RAG — WebSocket + Codex MCP + sessions + JSON rules + code agents

Prototype backend for a functional assistant where the **frontend can ask arbitrary questions**. The application does not enumerate every possible question. Instead it combines:

- a persistent WebSocket chat/session layer;
- Codex as the local agent/reasoning harness through MCP;
- Ollama as the configured local model provider behind Codex;
- a single model pass for semantic rule classification + answer reasoning;
- rolling context compaction around 50k estimated tokens;
- functional rules defined in JSON;
- code-defined agents/actions such as `send_email`;
- explicit user confirmation before write actions;
- code-agent suggestions are structured JSON, so worker models do not need native tool/function-calling support.

## Architecture

```text
Frontend
   │
   │ WebSocket /ws
   ▼
FastAPI
   │
   ├── SessionStore (SQLite)
   │      ├── user_id
   │      ├── chat_id
   │      ├── messages
   │      ├── rolling summary
   │      └── Codex thread_id
   │
   ├── ContextManager
   │      └── compacts old context at ~50k tokens
   │
   ├── RuleEngine
   │      ├── exposes + validates semantic PRE rules
   │      ├── enforces accepted rule actions locally
   │      └── evaluates deterministic POST rules
   │
   ├── AgentRegistry
   │      └── send_email (code)
   │
   └── CodexService
          └── one assistant_decision call
                ├── semantic rule classification
                ├── answer reasoning
                └── agent suggestion
                     │
                     ▼
                Codex MCP stdio
                     │
                     ▼
                  Ollama
```

For a normal chat request below the context-compaction threshold, the intended model path is therefore:

```text
User message
   ↓
FastAPI
   ↓
Codex / Ollama     assistant_decision
   ↓
{
  matched_rule,
  rule_confidence,
  status,
  answer,
  suggested_agent,
  suggested_agent_args
}
   ↓
RuleEngine validates/enforces locally
   ↓
POST rules + persistence
   ↓
WebSocket response
```

There is no separate `pre_rule_matching` model request followed by a second `assistant_reasoning` request anymore.

## Important Codex note (September 2026)

This prototype contains the requested MCP connection using `codex mcp-server` and the historical `codex` / `codex-reply` tools.

OpenAI deprecated `codex mcp-server` on **2026-08-24** in favor of **Codex App Server**. The Codex integration is intentionally isolated in `app/codex/client.py`, so the transport can be swapped later without changing WebSocket, session, context, rules, or agents.

The MCP adapter invokes the `codex` tool with `sandbox=read-only` and `approval-policy=never` by default, so this reasoning bridge does not need shell escalation.

If your installed Codex version still exposes MCP:

```bash
codex mcp-server
```

Otherwise implement an App Server adapter behind the same `CodexService` API.

## Context management

The context is persisted by `user_id + chat_id` in SQLite.

Default behavior:

```text
0 .. 49,999 estimated tokens
    -> keep normal history

>= 50,000
    -> keep ~12k newest tokens verbatim
    -> summarize older messages into a rolling summary
    -> discard compacted raw messages
    -> continue from summary + recent messages
```

Configuration:

```env
CONTEXT_COMPACT_AT_TOKENS=50000
CONTEXT_KEEP_RECENT_TOKENS=12000
CONTEXT_SUMMARY_TARGET_TOKENS=8000
```

The tokenizer is intentionally model-agnostic and estimates roughly `characters / 4`. Replace `ContextManager.estimate_tokens()` with the tokenizer used by your active local model when exact accounting matters.

Context summarization is the main exception to the one-model-call chat path: if compaction is required, `context_summarization` is an additional Ollama request.

## Functional rules

Rules live in `config/rules.json`; the code does not contain `if user asks exact phrase A` logic.

### Example: semantic rule

```json
{
  "id": "password_reset",
  "phase": "pre",
  "description": "The user asks how to recover, reset, change, or regain access to a forgotten password.",
  "when": { "type": "semantic" },
  "then": {
    "type": "respond",
    "canonical_answer": "The user must contact their administrator to reset the password.",
    "reformulate": true,
    "allow_new_facts": false
  }
}
```

So all of these can map to the same functional rule:

```text
I forgot my password
How can I reset my password?
I cannot access my account anymore
Where do I change my password?
```

Codex/Ollama performs the semantic match **inside the same `assistant_decision` request that also reasons about the answer**. The local RuleEngine then checks the returned rule id/confidence before applying it.

For a `respond` rule:

- `reformulate=false`: FastAPI ignores the model wording and returns `canonical_answer` exactly;
- `reformulate=true`: the wording already produced during `assistant_decision` is used, with `canonical_answer` as fallback;
- no second reformulation model call is made.

This keeps mandatory business behavior in JSON while avoiding a second sequential Ollama request.

### Example: deterministic output rule

```json
{
  "id": "math_calculation",
  "phase": "pre",
  "priority": 200,
  "description": "The user asks to perform a mathematical calculation.",
  "when": { "type": "semantic" },
  "then": {
    "type": "respond",
    "canonical_answer": "hahahaha",
    "reformulate": false,
    "allow_new_facts": false
  }
}
```

If this rule is accepted, the final answer is enforced locally as exactly `hahahaha`.

### Example: fallback rule

```json
{
  "id": "no_answer_suggest_email",
  "phase": "post",
  "when": {
    "type": "result_state",
    "field": "status",
    "operator": "in",
    "value": ["not_found", "insufficient_information"]
  },
  "then": {
    "type": "suggest_agent",
    "agent": "send_email",
    "label": "Send an email",
    "requires_confirmation": true
  }
}
```

This is a **functional state rule**, not a phrase matcher. POST rules are evaluated locally and do not require an Ollama call.

## Code-defined agents

Agents are registered in Python:

```python
agents.register(SendEmailAgent(settings))
```

Each agent exposes a specification:

```text
name
description
input_schema
write_action
requires_confirmation
```

Write actions are not executed merely because the model suggests them. The frontend must explicitly send `confirmed: true`.

## WebSocket protocol

Connect:

```text
ws://localhost:8765/ws
```

### Send a chat message

```json
{
  "type": "chat.message",
  "request_id": "req-1",
  "user_id": "user-123",
  "chat_id": null,
  "data": {
    "text": "I forgot my password. What should I do?"
  }
}
```

The server returns the generated `chat_id`. Reuse it on following turns.

### Main model event

A normal request should show one model operation:

```text
model.request.started    operation=assistant_decision
model.request.completed  operation=assistant_decision
```

The model event also reports `provider=codex_ollama` and the configured model name.

### Example assistant result

```json
{
  "type": "assistant.completed",
  "request_id": "req-1",
  "user_id": "user-123",
  "chat_id": "...",
  "data": {
    "status": "answered",
    "answer": "...",
    "matched_rule": "password_reset",
    "actions": []
  }
}
```

### No answer -> suggest email

```json
{
  "type": "assistant.completed",
  "data": {
    "status": "insufficient_information",
    "answer": "I do not have enough reliable information to answer this question.",
    "actions": [
      {
        "type": "suggest_agent",
        "agent": "send_email",
        "label": "Send an email",
        "requires_confirmation": true
      }
    ]
  }
}
```

### Execute the email agent

First request without confirmation can be used as a preview/guard:

```json
{
  "type": "agent.execute",
  "request_id": "email-1",
  "user_id": "user-123",
  "chat_id": "...",
  "data": {
    "agent": "send_email",
    "arguments": {
      "to": "help@example.com",
      "subject": "Need help",
      "body": "Hello, I need help with ..."
    },
    "confirmed": false
  }
}
```

The server returns `confirmation_required`.

After the user confirms in the UI, send the same event with:

```json
"confirmed": true
```

## Run

Prerequisites:

- Python 3.11+
- Codex CLI installed and authenticated/configured
- a Codex version that still supports `codex mcp-server`, or a future App Server adapter

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8765
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Health check:

```text
GET http://localhost:8765/health
```

## Local-model-only setup

This repository does **not** hard-code an OpenAI-hosted model. The Docker entrypoint configures Codex to use the Ollama provider and the remote OpenAI-compatible Ollama endpoint.

Do not store model/API credentials in this repository. Keep them in your local Codex/Ollama configuration or environment.

## Next architectural step

Because `codex mcp-server` is now deprecated, the next version should implement:

```text
CodexGateway
├── McpCodexClient       # legacy/requested prototype
└── AppServerCodexClient # recommended current transport
```

The rest of the application should remain unchanged.
