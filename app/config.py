from __future__ import annotations

import shlex
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = 8765
    database_path: Path = Path("data/rag.db")
    rules_path: Path = Path("config/rules")

    context_compact_at_tokens: int = 50_000
    context_keep_recent_tokens: int = 12_000
    context_summary_target_tokens: int = 8_000

    codex_command: str = "codex"
    codex_args: str = "mcp-server"
    codex_tool: str = "codex"
    codex_reply_tool: str = "codex-reply"
    codex_cwd: Path = Path(".")
    codex_model: str = ""
    codex_sandbox: str = "read-only"
    codex_approval_policy: str = "never"

    # Native Ollama is used for the fast integrated assistant decision so we can
    # enforce real `think=false` and collect Ollama's native timing metrics.
    ollama_base_url: str = "http://100.89.128.87:11434"
    ollama_model: str = "qwen3:8b"

    smtp_host: str = "smtp.example.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = "no-reply@example.com"
    smtp_starttls: bool = True

    @property
    def codex_arg_list(self) -> list[str]:
        return shlex.split(self.codex_args)


settings = Settings()
