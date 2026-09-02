from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any

from app.config import Settings

from .base import AgentResult, AgentSpec, CodeAgent


class SendEmailAgent(CodeAgent):
    spec = AgentSpec(
        name="send_email",
        description="Send an email to a recipient. This is a write action and always requires user confirmation.",
        write_action=True,
        requires_confirmation=True,
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    )

    def __init__(self, settings: Settings):
        self.settings = settings

    async def execute(self, arguments: dict[str, Any]) -> AgentResult:
        recipient = str(arguments.get("to", "")).strip()
        subject = str(arguments.get("subject", "")).strip()
        body = str(arguments.get("body", "")).strip()
        if not recipient or "@" not in parseaddr(recipient)[1]:
            return AgentResult(ok=False, error="invalid_recipient")
        if not subject or not body:
            return AgentResult(ok=False, error="missing_subject_or_body")
        if self.settings.smtp_host == "smtp.example.com":
            return AgentResult(ok=False, error="smtp_not_configured")

        try:
            await asyncio.to_thread(self._send_sync, recipient, subject, body)
            return AgentResult(ok=True, data={"to": recipient, "subject": subject})
        except Exception as exc:
            return AgentResult(ok=False, error=f"smtp_error:{type(exc).__name__}")

    def _send_sync(self, recipient: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self.settings.smtp_from
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=20) as smtp:
            if self.settings.smtp_starttls:
                smtp.starttls()
            if self.settings.smtp_username:
                smtp.login(self.settings.smtp_username, self.settings.smtp_password)
            smtp.send_message(message)
