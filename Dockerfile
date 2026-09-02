FROM node:24-bookworm-slim

ARG CODEX_VERSION=0.146.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    CODEX_HOME=/root/.codex

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g "@openai/codex@${CODEX_VERSION}" \
    && codex --version

WORKDIR /app

COPY . /app

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && sed 's/\r$//' /app/docker-entrypoint.sh > /usr/local/bin/rag-entrypoint \
    && chmod +x /usr/local/bin/rag-entrypoint

EXPOSE 8765

ENTRYPOINT ["/usr/local/bin/rag-entrypoint"]
CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8765}"]
