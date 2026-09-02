from __future__ import annotations

import math
from collections.abc import Sequence
from urllib.parse import urlparse

import httpx


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbeddingClient:
    """Batch embedding client for Ollama's native ``POST /api/embed`` API."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "bge-m3",
        timeout_seconds: float = 120.0,
        batch_size: int = 32,
        dimension: int = 1_024,
        truncate: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlparse(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not model.strip():
            raise ValueError("embedding model cannot be empty")
        if batch_size < 1 or dimension < 1:
            raise ValueError("batch_size and dimension must be positive")

        root = base_url.strip().rstrip("/")
        self.endpoint = f"{root}/embed" if parsed.path.rstrip("/").endswith("/api") else f"{root}/api/embed"
        self.model = model.strip()
        self.batch_size = int(batch_size)
        self.dimension = int(dimension)
        self.truncate = bool(truncate)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=max(0.1, timeout_seconds))

    async def __aenter__(self) -> OllamaEmbeddingClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _validate_embeddings(
        self, payload: object, *, expected_count: int
    ) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), list):
            raise EmbeddingError("Ollama response has no embeddings array")
        raw_embeddings = payload["embeddings"]
        if len(raw_embeddings) != expected_count:
            raise EmbeddingError(
                f"Ollama returned {len(raw_embeddings)} embeddings for {expected_count} inputs"
            )

        embeddings: list[list[float]] = []
        for index, raw in enumerate(raw_embeddings):
            if not isinstance(raw, list) or len(raw) != self.dimension:
                actual = len(raw) if isinstance(raw, list) else "non-array"
                raise EmbeddingError(
                    f"embedding {index} has dimension {actual}; expected {self.dimension}"
                )
            try:
                vector = [float(value) for value in raw]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(f"embedding {index} contains a non-number") from exc
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError(f"embedding {index} contains a non-finite value")
            embeddings.append(vector)
        return embeddings

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = [str(text).strip() for text in texts]
        if not normalized:
            return []
        if any(not text for text in normalized):
            raise ValueError("embedding inputs cannot be empty")

        output: list[list[float]] = []
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            response = await self._client.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "input": batch,
                    "truncate": self.truncate,
                },
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise EmbeddingError("Ollama returned invalid JSON") from exc
            output.extend(self._validate_embeddings(payload, expected_count=len(batch)))
        return output

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]
