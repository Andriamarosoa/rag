#!/bin/sh
set -eu

CODEX_HOME_DIR="${CODEX_HOME:-/root/.codex}"
OLLAMA_BASE_URL="${CODEX_OSS_BASE_URL:-http://100.89.128.87:11434/v1}"
MODEL="${CODEX_MODEL:-qwen3:8b}"

mkdir -p "$CODEX_HOME_DIR"

cat > "$CODEX_HOME_DIR/config.toml" <<EOF
model = "$MODEL"
model_provider = "ollama-rag"

[model_providers.ollama-rag]
name = "Ollama RAG"
base_url = "$OLLAMA_BASE_URL"
wire_api = "responses"
requires_openai_auth = false
EOF

echo "[rag] Codex: $(codex --version)"
echo "[rag] Codex model: $MODEL"
echo "[rag] Ollama endpoint: $OLLAMA_BASE_URL"

if curl -fsS --max-time 4 "$OLLAMA_BASE_URL/models" >/dev/null 2>&1; then
  echo "[rag] Ollama reachable"
else
  echo "[rag] WARNING: Ollama is not reachable from this container"
fi

exec "$@"
