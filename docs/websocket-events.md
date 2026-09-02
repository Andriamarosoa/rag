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

A normal request now uses one model pass for semantic pre-rule classification and assistant reasoning:

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
model.request.started            operation=assistant_decision
model.request.completed          operation=assistant_decision
rules.pre.decision_parsed
rules.pre.matched | rules.pre.no_match

# If a rule directly answers:
rule.action.started
rule.action.completed            second_model_call=false

# If no direct-answer rule matched, the normal answer was already produced by
# the same assistant_decision model request above.
agent.suggested                  (optional)
session.codex_thread.updated     (optional)
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

`model.request.*` includes `provider`, `model`, and `operation`. The main chat operation is now:

- `assistant_decision` — semantic rule classification + answer reasoning in one Ollama pass through Codex
- `context_summarization` — only when rolling context compaction is required

The event intentionally reports prompt/output sizes rather than prompt contents.

## Agent events

When an agent is proposed by the reasoning model or a post-rule:

- `agent.suggested`

When the frontend explicitly executes an agent:

- `agent.execution.started`
- `agent.execution.completed`
- `agent.execution.failed`
- `agent.result`

Write agents can still require `confirmed=true` before their code is allowed to run.

## Example: no answer -> email suggestion

```text
rules.pre.started
reasoning.started
model.request.started            operation=assistant_decision
model.request.completed          operation=assistant_decision
rules.pre.no_match
reasoning.completed              status=insufficient_information
rules.post.started
rules.post.matched               rule_id=no_answer_suggest_email
agent.suggested                  agent=send_email
response.fallback.applied
message.assistant.persisted
flow.completed
assistant.completed
```

## Example: deterministic math rule

```text
rules.pre.started
reasoning.started
model.request.started            operation=assistant_decision
model.request.completed          operation=assistant_decision
rules.pre.matched                rule_id=math_calculation
rule.action.started
rule.action.completed            second_model_call=false
reasoning.completed
message.assistant.persisted
flow.completed
assistant.completed
```

The response is then enforced locally from the rule's canonical answer (`hahahaha`); no reformulation request is made.
