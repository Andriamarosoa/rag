#!/bin/sh
set -eu

CODEX_HOME_DIR="${CODEX_HOME:-/root/.codex}"
CODEX_OLLAMA_BASE_URL="${CODEX_OSS_BASE_URL:-http://100.89.128.87:11434/v1}"
NATIVE_OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://100.89.128.87:11434}"
MODEL="${CODEX_MODEL:-qwen3:8b}"
NATIVE_MODEL="${OLLAMA_MODEL:-$MODEL}"

mkdir -p "$CODEX_HOME_DIR"

cat > "$CODEX_HOME_DIR/config.toml" <<EOF
model = "$MODEL"
model_provider = "ollama-rag"

[model_providers.ollama-rag]
name = "Ollama RAG"
base_url = "$CODEX_OLLAMA_BASE_URL"
wire_api = "responses"
requires_openai_auth = false
EOF

echo "[rag] Codex: $(codex --version)"
echo "[rag] Codex model: $MODEL"
echo "[rag] Codex Ollama endpoint: $CODEX_OLLAMA_BASE_URL"
echo "[rag] Native Ollama model: $NATIVE_MODEL"
echo "[rag] Native Ollama endpoint: $NATIVE_OLLAMA_BASE_URL"

if curl -fsS --max-time 4 "$CODEX_OLLAMA_BASE_URL/models" >/dev/null 2>&1; then
  echo "[rag] Codex Ollama endpoint reachable"
else
  echo "[rag] WARNING: Codex Ollama endpoint is not reachable from this container"
fi

exec "$@"
