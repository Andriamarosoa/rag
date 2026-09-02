# WebSocket workflow events

Connect to `ws://localhost:8765/ws`. Every server message uses the same envelope:

```json
{
  "type": "rag.retrieval.completed",
  "request_id": "request-uuid",
  "user_id": "demo-user",
  "chat_id": "chat-uuid",
  "data": {}
}
```

Clients must branch on the machine-readable `type`, not on display text.

## Strict SKB chat flow

Send a question and an optional SKB module namespace:

```json
{
  "type": "chat.message",
  "request_id": "req-1",
  "user_id": "user-123",
  "chat_id": null,
  "data": {
    "text": "How do I reset my ESS password?",
    "module": "spay"
  }
}
```

`module` may be `null` or one of the namespaces returned by `GET /skb/modules`.
An unknown value produces an `error` event with `code=invalid_module`.

The active `chat.message` event sequence is:

```text
assistant.started
flow.started                         mode=skb_grounded, module=<namespace|null>
session.ready
message.user.persisted
rag.retrieval.started                source_host=skb.uniconsults.mu
rag.retrieval.completed              retrieved_count=N

# Emitted only when at least one trusted chunk passed the threshold:
rag.generation.started               source_only=true
rag.generation.completed             status=..., citation_count=N

message.assistant.persisted
flow.completed                       grounded=true, source_count=N
assistant.completed
```

If retrieval, MariaDB, embeddings, or generation fails, the sequence contains
`rag.failed`; the assistant then returns `status=source_unavailable`, the fixed
temporary-unavailability message, and no sources. There is never a fallback to
general model knowledge.

An unexpected exception outside that handled dependency path uses a distinct
terminal branch:

```text
flow.failed                          error=<exception class>, message=<details>
error                                code=flow_failed
```

This branch does not emit `assistant.completed`. Clients must therefore treat
either `assistant.completed` or `flow.failed` as terminal for a submitted chat
request, and should use the following `error` envelope to display the failure.

When no relevant chunk is found, or the model returns an invalid/invented
citation, generation is omitted or discarded and the result is:

```json
{
  "status": "insufficient_information",
  "answer": "Je n’ai pas trouvé cette information dans la base de connaissances Sicorax.",
  "sources": [],
  "citations": [],
  "grounded": true
}
```

An answered `assistant.completed` payload includes only application-validated
SKB citations:

```json
{
  "status": "answered",
  "answer": "...",
  "module": "spay",
  "grounded": true,
  "retrieved_count": 6,
  "citations": ["chunk-id"],
  "sources": [
    {
      "id": "chunk-id",
      "page_id": "spay:faq:faq",
      "title": "Frequently Asked Questions (FAQ)",
      "section": "Login",
      "module": "Payroll",
      "url": "http://skb.uniconsults.mu/doku.php?id=spay%3Afaq%3Afaq",
      "distance": 0.29
    }
  ],
  "actions": [],
  "matched_rules": [],
  "matched_rule": null
}
```

## Active chat events

- `assistant.started`
- `flow.started`
- `session.ready`
- `message.user.persisted`
- `rag.retrieval.started`
- `rag.retrieval.completed`
- `rag.generation.started`
- `rag.generation.completed`
- `rag.failed`
- `message.assistant.persisted`
- `flow.completed`
- `flow.failed`
- `assistant.completed`
- `error`

Events expose status and counts, not the full prompt or retrieved documents.

## Connection and explicit controls

- `connection.ready`
- `pong`
- `rules.reload.started`
- `rules.reloaded`
- `rules.reload.failed`
- `agent.execution.started`
- `agent.execution.completed`
- `agent.execution.failed`
- `agent.result`

The existing rule engine and agents remain available through explicit control
messages (`rules.reload` and `agent.execute`). They are deliberately bypassed by
`chat.message`, because their output is not necessarily supported by an SKB
source. Write agents still require `confirmed=true`.

## Legacy internal events

The codebase retains context/rule/Codex compatibility services, which can emit
events such as `context.*`, `rules.pre.*`, `reasoning.*`, `model.request.*`, and
`rules.post.*` when invoked directly by legacy tests or a future non-strict
orchestrator. These events are not part of the currently wired strict SKB chat
flow.
