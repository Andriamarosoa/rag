# WebSocket workflow events

Every server message uses the same envelope:

```json
{
  "type": "rules.pre.matched",
  "request_id": "request-uuid",
  "user_id": "demo-user",
  "chat_id": "chat-uuid",
  "data": {}
}
```

`type` is the machine-readable event name. The frontend should branch on `type`, never on human-readable text.

## Typical chat flow

A normal request uses one native Ollama model pass for semantic pre-rule classification and assistant reasoning:

```text
assistant.started
flow.started
session.ready
message.user.persisted
context.snapshot
context.compaction.skipped | context.compaction.started
  model.request.started          operation=context_summarization
  model.request.completed
  context.compaction.completed
context.ready
rules.pre.started                integrated_with_reasoning=true
reasoning.started                integrated_rule_matching=true
model.request.started            provider=ollama_native operation=assistant_decision thinking_mode=disabled
model.request.completed          provider=ollama_native operation=assistant_decision
rules.pre.decision_parsed
rules.pre.matched | rules.pre.no_match

# If a rule directly answers:
rule.action.started
rule.action.completed            second_model_call=false

# If no direct-answer rule matched, the normal answer was already produced by
# the same assistant_decision model request above.
agent.suggested                  (optional)
reasoning.completed

rules.post.started
rules.post.matched | rules.post.no_match
agent.suggested                  (optional, from a post rule)
response.fallback.applied        (optional)
message.assistant.persisted
context.snapshot
context.compaction.skipped | context.compaction.started/completed
flow.completed
assistant.completed
```

Except for context compaction when the token threshold is crossed, a normal chat request therefore produces a single `model.request.started/completed` pair.

## Connection and control events

- `connection.ready`
- `pong`
- `rules.reload.started`
- `rules.reloaded`
- `rules.reload.failed`
- `error`

## Session/message events

- `flow.started`
- `session.ready`
- `session.codex_thread.updated`
- `message.user.persisted`
- `message.assistant.persisted`
- `flow.completed`
- `flow.failed`

## Context events

- `context.snapshot`
- `context.ready`
- `context.compaction.skipped`
- `context.compaction.started`
- `context.compaction.completed`

No full context or prompt is emitted in these events; only metadata such as token estimates and message counts.

## Rule events

- `rules.pre.started`
- `rules.pre.decision_parsed`
- `rules.pre.matched`
- `rules.pre.no_match`
- `rule.action.started`
- `rule.action.completed`
- `rule.action.unsupported`
- `rules.post.started`
- `rules.post.matched`
- `rules.post.no_match`

A semantic pre-rule match includes its `rule_id`, confidence, priority and action type. The rule engine does not call a model itself; it validates the rule decision returned by `assistant_decision` and enforces the configured action locally.

## Model events

- `model.request.started`
- `model.request.completed`
- `model.request.failed`
- `reasoning.started`
- `reasoning.completed`

The main operations are:

- `assistant_decision` — semantic rule classification + answer reasoning through Ollama native `/api/chat` with the real API field `think=false`
- `context_summarization` — Codex/Ollama path used only when rolling context compaction is required

### Native assistant-decision telemetry

`model.request.started` includes:

```json
{
  "provider": "ollama_native",
  "model": "qwen3:8b",
  "operation": "assistant_decision",
  "thinking_mode": "disabled",
  "thinking_control": "native_think_false",
  "prompt_tokens_estimated": 700,
  "timing_scope": "ollama_native_api"
}
```

`model.request.completed` exposes both application wall-clock timing and Ollama's native response metrics:

```json
{
  "elapsed_seconds": 2.1,
  "ollama_total_seconds": 2.0,
  "ollama_load_seconds": 0.1,
  "ollama_prompt_eval_count": 710,
  "ollama_prompt_eval_seconds": 0.8,
  "ollama_prompt_tokens_per_second": 887.5,
  "ollama_eval_count": 42,
  "ollama_eval_seconds": 1.0,
  "ollama_output_tokens_per_second": 42.0,
  "metrics_source": "ollama_native_response",
  "native_ollama_eval_metrics_available": true
}
```

The native counts/durations are preferable to the `characters / 4` estimates when available. `elapsed_seconds` remains useful to detect HTTP/application overhead outside Ollama itself.

Codex-backed operations still use `provider=codex_ollama` and report application wall-clock metrics because the legacy MCP/OpenAI Responses transport does not preserve Ollama's native timing fields.

## Agent events

When an agent is proposed by the reasoning model or a post-rule:

- `agent.suggested`

When the frontend explicitly executes an agent:

- `agent.execution.started`
- `agent.execution.completed`
- `agent.execution.failed`
- `agent.result`

Write agents can still require `confirmed=true` before their code is allowed to run.

## Example: deterministic math rule

```text
rules.pre.started
reasoning.started
model.request.started            provider=ollama_native operation=assistant_decision thinking_mode=disabled
model.request.completed          operation=assistant_decision
rules.pre.matched                rule_id=math_calculation
rule.action.started
rule.action.completed            second_model_call=false
reasoning.completed
message.assistant.persisted
flow.completed
assistant.completed
```

The response is enforced locally from the rule's canonical answer (`hahahaha`); no reformulation request is made.
