# Docker runtime

## Topology

~~~text
Browser
   |
   | HTTP/WebSocket :8765
   v
rag-backend
   |- FastAPI and SQLite chat sessions
   |- DokuWiki indexer and chunker
   |- Ollama embedding and chat clients
   |- validated-citation answer service
   |
   +---- HTTP ----> skb.uniconsults.mu
   |                 do=index and do=export_raw
   |
   +---- HTTP ----> 100.89.128.87:11434
   |                 bge-m3 and qwen3:8b
   |
   +---- MySQL ---> mariadb:3306
                     MariaDB 11.8 LTS VECTOR cosine index

mariadb
   `- host port 3307, persistent ./mariadb_data
~~~

The strict chat.message route embeds the question, performs an exact
module-filtered MariaDB cosine search within the active generation, asks Qwen
to generate from retrieved SKB chunks only, validates citation IDs and SKB
URLs, and returns an answer with sources or an abstention. The query
deliberately bypasses the table-wide approximate vector index because active
and superseded generations can contain identical vectors.

JSON rules, Codex compatibility code, and code agents remain in the container
for explicit control operations. They are not part of chat answer generation.

## Prerequisites

- Docker Desktop and Docker Compose;
- network access from Docker to <http://skb.uniconsults.mu/>;
- an Ollama endpoint reachable from Docker;
- qwen3:8b and bge-m3 installed on that Ollama server.

Install the models on the configured remote Ollama server, not in the backend
container. From a PowerShell terminal that can reach the default remote host,
point the Ollama CLI explicitly at it:

~~~powershell
$env:OLLAMA_HOST = "http://100.89.128.87:11434"
ollama pull qwen3:8b
ollama pull bge-m3
~~~

`bge-m3` is the multilingual 1024-dimension embedding model used for both SKB
chunks and user questions.

## Start

From the repository root:

~~~powershell
docker compose up -d --build
docker compose ps
docker compose logs -f rag-backend
~~~

Open:

- UI: <http://localhost:8765/>
- health: <http://localhost:8765/health>
- index status: <http://localhost:8765/skb/index/status>

The health endpoint starts even if an external dependency is unavailable.
ok=true means the HTTP application is running; also inspect
skb_index_initialized, skb_index_running, skb_index_ready, and skb_index_error.

## Initial and incremental index sync

When SKB_SYNC_ON_STARTUP=true, startup initializes the MariaDB schema and
schedules a background sync. The server does not wait for a potentially long
first ingestion before serving HTTP.

Monitor it:

~~~powershell
Invoke-RestMethod http://localhost:8765/skb/index/status |
  ConvertTo-Json -Depth 8
~~~

Start or join a sync and wait for completion:

~~~powershell
Invoke-RestMethod -Method Post "http://localhost:8765/skb/index/sync?wait=true" |
  ConvertTo-Json -Depth 8
~~~

Start it asynchronously:

~~~powershell
Invoke-RestMethod -Method Post "http://localhost:8765/skb/index/sync"
~~~

The status object reports:

- initialized, running, and ready;
- the configured signature and active generation/signature;
- last start/completion times and last_error;
- last_result counters such as discovered_pages, fetched_pages,
  unchanged_pages, indexed_pages, failed_pages, embedded_chunks,
  upserted_chunks, and deletions;
- store.pages, store.chunks, and store.modules.

Sync is hash-based and incremental for embedding work. Unchanged pages are
copied into a private staging generation; search reads only the active complete
generation. The staging generation is atomically activated after a fully
successful crawl, so a first ingestion or model change cannot expose mixed or
partial embeddings. A partial fetch is discarded. Removals require two matching
snapshots and a retention ratio of at least 90% before activation.

## Verify the strict pipeline

List modules:

~~~powershell
Invoke-RestMethod http://localhost:8765/skb/modules |
  ConvertTo-Json -Depth 5
~~~

Test retrieval across all modules:

~~~powershell
Invoke-RestMethod "http://localhost:8765/skb/search?q=install%20payroll&limit=5" |
  ConvertTo-Json -Depth 6
~~~

Test a module filter:

~~~powershell
Invoke-RestMethod "http://localhost:8765/skb/search?q=leave&module=spay&limit=5" |
  ConvertTo-Json -Depth 6
~~~

Results include cosine distance and score=1-distance. Results beyond
SKB_RETRIEVAL_MAX_DISTANCE are excluded.

## Configuration

Create a local .env from .env.example. Important Docker defaults:

~~~env
APP_PORT=8765

SKB_BASE_URL=http://skb.uniconsults.mu/
SKB_REQUEST_TIMEOUT_SECONDS=15
SKB_CRAWL_CONCURRENCY=3
SKB_SYNC_ON_STARTUP=true

SKB_EMBEDDING_BASE_URL=http://100.89.128.87:11434
SKB_EMBEDDING_MODEL=bge-m3
SKB_EMBEDDING_DIMENSION=1024
SKB_EMBEDDING_BATCH_SIZE=16
SKB_EMBEDDING_TIMEOUT_SECONDS=180

SKB_CHUNK_SIZE=1600
SKB_CHUNK_OVERLAP=200
SKB_MIN_CHUNK_SIZE=80
SKB_RETRIEVAL_TOP_K=3
SKB_RETRIEVAL_MAX_DISTANCE=0.45
SKB_ANSWER_MAX_CONTEXT_CHARACTERS=24000

OLLAMA_BASE_URL=http://100.89.128.87:11434
OLLAMA_MODEL=qwen3:8b

DB_USER=myuser
DB_PASSWORD=myuserpassword
DB_NAME=sicorax
DB_POOL_MIN_SIZE=1
DB_POOL_MAX_SIZE=5
DB_CONNECT_TIMEOUT_SECONDS=10
~~~

Compose fixes the backend's internal database route to DB_HOST=mariadb and
DB_PORT=3306. From Windows, MariaDB is exposed on 127.0.0.1:3307.

The vector dimension is a schema contract. The backend refuses an existing
skb_chunks_v2.embedding column whose dimension differs from
SKB_EMBEDDING_DIMENSION.

SKB_MODULE_CACHE_SECONDS and SKB_SEARCH_MAX_PAGES belong to the legacy bounded
lexical crawler and do not affect the strict vector path.

## Network checks

Verify Ollama from inside the backend container:

~~~powershell
docker compose exec rag-backend curl -fsS http://100.89.128.87:11434/api/tags
~~~

Verify SKB:

~~~powershell
docker compose exec rag-backend curl -fsS "http://skb.uniconsults.mu/doku.php?do=export_raw&id=start"
~~~

Verify MariaDB:

~~~powershell
docker compose exec mariadb mariadb -umyuser -pmyuserpassword sicorax -e "SELECT VERSION();"
~~~

MariaDB 11.8 LTS or newer is required because the schema uses native
VECTOR(1024), a VECTOR INDEX with DISTANCE=cosine, and VEC_DISTANCE_COSINE.

## Persistence

~~~text
./mariadb_data -> SKB pages, chunks, embeddings, and vector index
./data         -> SQLite chat/session state
./codex_data   -> Codex compatibility state and configuration
~~~

These directories and .env are ignored by Git. Replace the example MariaDB
credentials beyond isolated local development.

## Failure behavior

- Empty or below-threshold retrieval: fixed insufficient_information
  abstention with no sources.
- MariaDB, embedding, or Qwen failure during chat: source_unavailable with no
  sources.
- Invalid citations, non-verbatim evidence quotes, or any unsupported claim:
  all generated text is discarded and replaced by the abstention.
- Sync errors appear in /skb/index/status. A synchronous `wait=true` request
  returns HTTP 502 when the generation could not be activated.
