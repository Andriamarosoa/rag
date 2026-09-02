from __future__ import annotations

import httpx
import pytest

from app.skb.embeddings import EmbeddingError, OllamaEmbeddingClient


@pytest.mark.asyncio
async def test_ollama_embeddings_are_sent_in_configured_batches():
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        batches.append(payload["input"])
        assert request.url.path == "/api/embed"
        return httpx.Response(
            200,
            json={"embeddings": [[float(len(text)), 0.0, 1.0] for text in payload["input"]]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaEmbeddingClient(
            "http://ollama:11434",
            client=http,
            dimension=3,
            batch_size=2,
        )
        vectors = await client.embed_texts(["one", "two", "three"])

    assert batches == [["one", "two"], ["three"]]
    assert vectors == [[3.0, 0.0, 1.0], [3.0, 0.0, 1.0], [5.0, 0.0, 1.0]]


@pytest.mark.asyncio
async def test_embedding_dimension_mismatch_fails_closed():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaEmbeddingClient(
            "http://ollama:11434", client=http, dimension=3
        )
        with pytest.raises(EmbeddingError, match="dimension"):
            await client.embed_query("question")

