# Docker runtime

The application runtime is split into two containers:

```text
Browser
   |
   | http/ws :8765
   v
rag-backend
   |- FastAPI
   |- Python runtime
   |- Node.js 24
   |- Codex CLI 0.146.1
   |- Codex stdio MCP child process
   |
   | Tailscale / host network route
   v
100.89.128.87:11434
   `- Ollama (Mac)

mariadb
   `- MariaDB 11.7 on host port 3307
```

Codex is intentionally installed in the same container as FastAPI because the current adapter communicates with Codex through MCP over stdin/stdout.

## Start everything

From the repository root:

```powershell
docker compose up -d --build
```

Follow the backend logs:

```powershell
docker compose logs -f rag-backend
```

Expected startup lines include:

```text
[rag] Codex: codex-cli 0.146.1
[rag] Codex model: qwen3:8b
[rag] Ollama endpoint: http://100.89.128.87:11434/v1
[rag] Ollama reachable
```

Open the WebSocket test UI:

```text
http://localhost:8765/
```

Health endpoint:

```powershell
curl.exe http://localhost:8765/health
```

## Verify Codex inside Docker

```powershell
docker compose exec rag-backend codex --version
```

Inspect the generated local-only Codex provider:

```powershell
docker compose exec rag-backend cat /root/.codex/config.toml
```

It should point to the `ollama-rag` provider and not the OpenAI provider.

## Verify Docker can reach Ollama

Windows being able to reach the Tailscale IP does not automatically prove Docker Desktop can use the same route. Test from inside the backend container:

```powershell
docker compose exec rag-backend curl -sS http://100.89.128.87:11434/v1/models
```

If this returns the Ollama model list, the complete path is available:

```text
FastAPI container -> Codex CLI -> Ollama on 100.89.128.87
```

If it fails while the same curl works directly from Windows, the remaining problem is Docker Desktop/Tailscale routing rather than FastAPI, Codex or Ollama.

## Configuration

Defaults:

```env
CODEX_VERSION=0.146.1
CODEX_MODEL=qwen3:8b
CODEX_OSS_BASE_URL=http://100.89.128.87:11434/v1
APP_PORT=8765
```

Override them in a local `.env` file if needed. `.env` is ignored by Git.

## Persistence

The following host directories survive container recreation:

```text
./mariadb_data  -> MariaDB data
./data          -> application SQLite/session state for the current prototype
./codex_data    -> Codex state/threads/config
```

`mariadb_data/` and `codex_data/` are ignored by Git.
