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

A normal request can produce this sequence:

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
rules.pre.started
model.request.started            operation=pre_rule_matching
model.request.completed
rules.pre.matched | rules.pre.no_match

# If a rule directly answers:
rule.action.started
model.request.started            operation=rule_answer_reformulation (optional)
model.request.completed          (optional)
rule.action.completed

# Otherwise:
reasoning.started
model.request.started            operation=assistant_reasoning
model.request.completed
reasoning.completed
agent.suggested                  (optional)
session.codex_thread.updated     (optional)

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
- `rules.pre.matched`
- `rules.pre.no_match`
- `rule.action.started`
- `rule.action.completed`
- `rule.action.unsupported`
- `rules.post.started`
- `rules.post.matched`
- `rules.post.no_match`

A semantic pre-rule match includes its `rule_id`, confidence, priority and action type.

## Model events

- `model.request.started`
- `model.request.completed`
- `model.request.failed`
- `reasoning.started`
- `reasoning.completed`

`model.request.*` includes an `operation` field so the UI can distinguish:

- `pre_rule_matching`
- `assistant_reasoning`
- `rule_answer_reformulation`
- `context_summarization`

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
rules.pre.no_match
reasoning.started
model.request.started
model.request.completed
reasoning.completed              status=insufficient_information
rules.post.started
rules.post.matched               rule_id=no_answer_suggest_email
agent.suggested                  agent=send_email
response.fallback.applied
message.assistant.persisted
flow.completed
assistant.completed
```

The frontend can therefore render the complete decision path without inspecting or parsing the assistant answer.
