# RAG — strict Sicorax Knowledge Base assistant

This project exposes a FastAPI/WebSocket assistant whose chat answers are grounded
exclusively in the public Sicorax Knowledge Base (SKB):

<http://skb.uniconsults.mu/>

The strict chat path does not use the model's general knowledge. It retrieves
relevant SKB sections from a persistent vector index, asks Qwen to answer from
those sections only, validates the citations returned by the model, and abstains
when the evidence is missing or unavailable.

## Data and answer paths

Indexing:

```text
DokuWiki index (do=index)
    -> discover existing pages and namespaces
    -> raw page export (do=export_raw)
    -> clean DokuWiki markup
    -> section-aware chunks with overlap
    -> Ollama /api/embed (bge-m3 multilingual, 1024 dimensions)
    -> MariaDB 11.8 LTS VECTOR(1024), cosine index
```

Chat:

```text
Browser -> WebSocket chat.message {text, module}
    -> embed the question with bge-m3
    -> MariaDB cosine retrieval
       (optional module filter, top-k and distance threshold)
    -> Qwen receives only the retrieved SKB chunks
    -> structured atomic claims with citation IDs and exact evidence quotes
    -> application validates every quote, citation ID, and SKB URL
    -> assistant.completed {answer, sources, ...}
```

Important guarantees:

- retrieved wiki text is treated as untrusted reference data, never as
  instructions;
- Qwen must split its answer into atomic claims, each with an exact excerpt from
  a retrieved chunk;
- the application releases a claim only when every evidence quote is found in
  its cited SKB chunk and every citation ID belongs to the retrieved set;
- public source URLs are rebuilt from validated chunks and restricted to the SKB
  host;
- no relevant chunk, invalid JSON, an invalid citation, or an unsupported answer
  produces the fixed abstention:
  `Je n’ai pas trouvé cette information dans la base de connaissances Sicorax.`;
- a retrieval or generation dependency failure produces:
  `La base de connaissances Sicorax est temporairement indisponible.`;
- there is no fallback to general model knowledge.

## Runtime components

- FastAPI and WebSocket transport;
- SQLite chat/session persistence;
- DokuWiki discovery and `export_raw` ingestion;
- Ollama `bge-m3` multilingual embeddings (1024 dimensions);
- MariaDB 11.8 LTS native `VECTOR` storage and cosine search;
- Ollama/Qwen grounded answer generation with `think=false`;
- a static HTML/JavaScript test UI;
- the existing JSON rule engine and code-agent registry.

Rules and agents remain available as explicit control operations, such as
`rules.reload` and confirmed `agent.execute`. They are deliberately bypassed
by `chat.message`: rule canonical answers and arbitrary agent output do not
produce chat answers because they are not necessarily supported by SKB.

## Supported modules

The WebSocket payload uses the namespace in the first column. Use `null` to
search all modules.

| Namespace | Display label |
| --- | --- |
| `spay` | Payroll |
| `shrm` | Human Resources |
| `sgc` | Gestion Commerciale |
| `sacc` | Accounting |
| `sfar` | Fixed Asset |
| `sef` | Equipment Follow-up |
| `sim` | Incident Management |
| `seam` | SEAM |
| `pms` | PMS |
| `sess` | SESS |

`GET /skb/modules` is the authoritative runtime representation, including the
DokuWiki start page and URL for every module.

## Quick start with Docker

Prerequisites:

- Docker Compose;
- an Ollama server reachable by the backend;
- `qwen3:8b` and `bge-m3` installed on that Ollama server.

Install the models on the configured remote Ollama server before starting the
stack. From a PowerShell terminal that can reach that server, point the Ollama
CLI explicitly at the remote endpoint:

```powershell
$env:OLLAMA_HOST = "http://100.89.128.87:11434"
ollama pull qwen3:8b
ollama pull bge-m3
```

```powershell
docker compose up -d --build
docker compose logs -f rag-backend
```

The test UI is available at <http://localhost:8765/> and the health endpoint at
<http://localhost:8765/health>.

The backend creates the MariaDB vector schema and, with
`SKB_SYNC_ON_STARTUP=true`, starts an incremental SKB sync in the background.
Every sync builds a private generation and publishes it with one atomic pointer
change only after the complete crawl succeeds. Until the first generation is
active, chat safely reports that SKB is unavailable.
Check progress instead of assuming that the first index is immediately ready:

```powershell
Invoke-RestMethod http://localhost:8765/skb/index/status
```

Run a synchronous refresh when required:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8765/skb/index/sync?wait=true"
```

See [docs/docker.md](docs/docker.md) for runtime and configuration details.

## HTTP API

### Health

```text
GET /health
```

Reports the strict pipeline, configured models, vector store type, and current
index initialization/running/error state. A healthy HTTP process does not imply
that an external Ollama or MariaDB dependency is currently reachable; inspect
`skb_index_error` and `/skb/index/status`.

### Modules

```text
GET /skb/modules
```

Example:

```json
{
  "base_url": "http://skb.uniconsults.mu/",
  "count": 10,
  "modules": [
    {
      "id": "spay",
      "label": "Payroll",
      "start_page_id": "spay:spay",
      "url": "http://skb.uniconsults.mu/doku.php?id=spay%3Aspay"
    }
  ]
}
```

### Semantic search

```text
GET /skb/search?q=<question>&module=<namespace-or-label>&limit=5
```

`module` is optional. Search embeds the query, performs cosine retrieval, and
returns matching chunks with `id`, `page_id`, `title`, `url`, `module`,
`section`, `snippet`, `score`, and `distance`. It returns HTTP 502 with
`code=skb_vector_search_failed` when a dependency fails.

### Index status

```text
GET /skb/index/status
```

The result contains:

- `initialized`, `running`, and `ready`;
- the configured signature and active generation/signature;
- `last_started_at`, `last_completed_at`, and `last_error`;
- `last_result` with discovery, fetch, indexing, embedding, deletion, and error
  counters;
- `store` with current page, chunk, and module counts when MariaDB is
  initialized.

### Index sync

```text
POST /skb/index/sync
POST /skb/index/sync?wait=true
```

Without `wait=true`, the endpoint schedules/reuses the background task and
returns its current status. With it, the request waits for that task and then
returns the final status. A failed or deferred synchronization returns HTTP 502
with the status in its error detail.

Synchronization is incremental for embedding work: unchanged pages and vectors
are copied into a private staging generation. Search continues to read the old
complete generation until the new one is atomically activated. A partial fetch
is discarded. Page removals additionally require two identical crawl snapshots
and at least 90% retention, preventing a truncated HTTP 200 response from
purging most of the corpus.

## WebSocket chat

Connect to:

```text
ws://localhost:8765/ws
```

Send:

```json
{
  "type": "chat.message",
  "request_id": "req-1",
  "user_id": "user-123",
  "chat_id": null,
  "data": {
    "text": "How do I install Payroll?",
    "module": "spay"
  }
}
```

`module` must be one of the ten namespaces above or `null`. An unknown value
returns an `error` envelope with `code=invalid_module`. Reuse the generated
`chat_id` for later messages.

An answered result includes application-validated sources:

```json
{
  "type": "assistant.completed",
  "request_id": "req-1",
  "chat_id": "chat-uuid",
  "data": {
    "status": "answered",
    "answer": "Grounded answer in the user's language.",
    "module": "spay",
    "grounded": true,
    "retrieved_count": 6,
    "citations": ["chunk-id"],
    "sources": [
      {
        "id": "chunk-id",
        "page_id": "spay:install:installation",
        "title": "Installation",
        "section": "Install Payroll",
        "module": "Payroll",
        "url": "http://skb.uniconsults.mu/doku.php?id=spay%3Ainstall%3Ainstallation",
        "distance": 0.21
      }
    ],
    "actions": [],
    "matched_rules": [],
    "matched_rule": null
  }
}
```

The frontend renders the answer as text and sources as safe HTTP(S) links. It
does not render model-provided HTML.

If the chat orchestration raises an unexpected exception, the server emits
`flow.failed`, then an `error` event with `code=flow_failed`. That failure
branch does not emit `assistant.completed`.

See [docs/websocket-events.md](docs/websocket-events.md) for the full event flow.

## Configuration

Core RAG variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SKB_BASE_URL` | `http://skb.uniconsults.mu/` | Allowed DokuWiki origin |
| `SKB_REQUEST_TIMEOUT_SECONDS` | `15` | DokuWiki request timeout |
| `SKB_CRAWL_CONCURRENCY` | `3` | Concurrent DokuWiki requests |
| `SKB_EMBEDDING_BASE_URL` | `http://100.89.128.87:11434` | Ollama embedding API root |
| `SKB_EMBEDDING_MODEL` | `bge-m3` | Multilingual embedding model |
| `SKB_EMBEDDING_DIMENSION` | `1024` | Vector dimension; must match the model and table |
| `SKB_EMBEDDING_BATCH_SIZE` | `16` | Inputs per `/api/embed` call |
| `SKB_EMBEDDING_TIMEOUT_SECONDS` | `180` | Embedding request timeout |
| `SKB_CHUNK_SIZE` | `1600` | Target maximum chunk characters |
| `SKB_CHUNK_OVERLAP` | `200` | Character overlap |
| `SKB_MIN_CHUNK_SIZE` | `80` | Small-tail merge threshold |
| `SKB_RETRIEVAL_TOP_K` | `6` | Maximum chunks for chat retrieval |
| `SKB_RETRIEVAL_MAX_DISTANCE` | `0.45` | Maximum cosine distance |
| `SKB_ANSWER_MAX_CONTEXT_CHARACTERS` | `24000` | Evidence budget sent to Qwen |
| `SKB_SYNC_ON_STARTUP` | `true` | Schedule incremental sync after startup |
| `OLLAMA_BASE_URL` | `http://100.89.128.87:11434` | Qwen native API root |
| `OLLAMA_MODEL` | `qwen3:8b` | Grounded answer model |

MariaDB variables:

| Variable | Local default | Docker value/default |
| --- | --- | --- |
| `DB_HOST` | `127.0.0.1` | `mariadb` |
| `DB_PORT` | `3307` | `3306` |
| `DB_USER` | `myuser` | `myuser` |
| `DB_PASSWORD` | `myuserpassword` | `myuserpassword` |
| `DB_NAME` | `sicorax` | `sicorax` |
| `DB_POOL_MIN_SIZE` | `1` | `1` |
| `DB_POOL_MAX_SIZE` | `5` | `5` |
| `DB_CONNECT_TIMEOUT_SECONDS` | `10` | `10` |

Use a local `.env` and replace the development database credentials outside
local development. `.env`, database data, and Codex state are ignored by Git.

`SKB_MODULE_CACHE_SECONDS` and `SKB_SEARCH_MAX_PAGES` remain as legacy
lexical-client settings; they do not control the strict persistent vector index.

### Other URLs

`SKB_BASE_URL` may replace the source with another DokuWiki origin, but the
current deployment deliberately allows only that one host. Supporting several
origins simultaneously requires an explicit source registry and a crawler
adapter per site type; ordinary HTML sites cannot be safely added by merely
changing the host allowlist.

## Local Python development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
uvicorn app.main:app --reload --host 0.0.0.0 --port 8765
```

Python 3.11+ and MariaDB 11.8 LTS or newer are required. The Ollama embedding
dimension must match `SKB_EMBEDDING_DIMENSION`; an existing incompatible vector
table is rejected rather than silently reused.
