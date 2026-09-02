from __future__ import annotations

import httpx
import pytest

from app.skb.dokuwiki import DokuWikiClient, DokuWikiSecurityError


def _index_html(body: str) -> str:
    return f"<html><body><ul class='idx'>{body}</ul></body></html>"


@pytest.mark.asyncio
async def test_recursive_discovery_uses_only_indexed_pages_and_excludes_system_namespaces():
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        query = request.url.params
        namespace = query.get("idx", "")
        if namespace == "":
            html = _index_html(
                """
                <li><a class="idx_dir" href="/doku.php?id=start&amp;idx=spay">spay</a></li>
                <li><a class="idx_dir" href="/doku.php?id=start&amp;idx=wiki">wiki</a></li>
                <li><a class="wikilink1" title="start" href="/doku.php?id=start">Home</a></li>
                <li><a class="wikilink2" title="missing" href="/doku.php?id=missing">Missing</a></li>
                """
            )
        elif namespace == "spay":
            html = _index_html(
                """
                <li><a class="idx_dir" href="/doku.php?id=start&amp;idx=spay%3Afaq">faq</a></li>
                <li><a class="wikilink1" title="spay:spay" href="/doku.php?id=spay:spay">Payroll</a></li>
                """
            )
        else:
            html = _index_html(
                '<li><a class="wikilink1" title="spay:faq:leave" '
                'href="/doku.php?id=spay:faq:leave">Leave</a></li>'
            )
        # This template link must never become a discovered page.
        html += '<a class="wikilink1" title="outside" href="/doku.php?id=outside">outside</a>'
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = DokuWikiClient("http://skb.uniconsults.mu/", client=http)
        pages = await client.discover_page_ids()

    assert pages == ["spay:faq:leave", "spay:spay", "start"]
    assert len(requested) == 3
    assert all("idx=wiki" not in url for url in requested)


@pytest.mark.asyncio
async def test_redirect_target_is_checked_before_following_it():
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://evil.example/index"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DokuWikiClient("http://skb.uniconsults.mu/", client=http)
        with pytest.raises(DokuWikiSecurityError):
            await client.discover_page_ids()

    assert len(requested) == 1
    assert "evil.example" not in requested[0]


@pytest.mark.asyncio
async def test_transient_server_errors_are_retried_twice():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(
            200,
            text=_index_html(
                '<a class="wikilink1" title="start" href="/doku.php?id=start">Home</a>'
            ),
            headers={"content-type": "text/html"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DokuWikiClient(
            "http://skb.uniconsults.mu/",
            client=http,
            retry_backoff_seconds=0,
        )
        assert await client.discover_page_ids() == ["start"]
    assert calls == 3


@pytest.mark.asyncio
async def test_raw_export_builds_trusted_canonical_source_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("do") == "export_raw"
        return httpx.Response(
            200,
            text="====== Pay Elements ======\n\nHow to create one.",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DokuWikiClient("http://skb.uniconsults.mu/", client=http)
        page = await client.fetch_page("spay:setup:pay_elements")

    assert page.title == "Pay Elements"
    assert page.module == "Payroll"
    assert page.source_url.startswith("http://skb.uniconsults.mu/doku.php?")
    assert "export_raw" not in page.source_url
    assert len(page.content_hash) == 64

