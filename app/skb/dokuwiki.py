from __future__ import annotations

import asyncio
import re
from hashlib import sha256
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx

from app.skb.models import WikiPage, module_for_page_id


_SAFE_DOKUWIKI_ID = re.compile(r"^[a-zA-Z0-9_.:-]+$")
_HEADING = re.compile(r"^\s*={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)


class DokuWikiError(RuntimeError):
    pass


class DokuWikiSecurityError(DokuWikiError):
    pass


class DokuWikiDiscoveryLimitError(DokuWikiError):
    pass


class _IndexLinkCollector(HTMLParser):
    """Collect only links inside DokuWiki's actual ``ul.idx`` tree."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._idx_depth = 0
        self._href: str | None = None
        self._classes: set[str] = set()
        self._title: str | None = None
        self._text: list[str] = []
        self.links: list[tuple[str, set[str], str | None, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "ul":
            if self._idx_depth:
                self._idx_depth += 1
            elif "idx" in classes:
                self._idx_depth = 1
        elif tag == "a" and self._idx_depth:
            self._href = attrs_dict.get("href")
            self._classes = classes
            self._title = attrs_dict.get("title")
            self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._idx_depth and self._href is not None:
            self.links.append(
                (
                    self._href,
                    set(self._classes),
                    self._title,
                    " ".join(self._text).strip(),
                )
            )
            self._href = None
            self._classes = set()
            self._title = None
            self._text = []
        elif tag == "ul" and self._idx_depth:
            self._idx_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            cleaned = " ".join(data.split())
            if cleaned:
                self._text.append(cleaned)


class DokuWikiClient:
    """Minimal, read-only client for a DokuWiki index and raw exports.

    Every request, including every redirect target, is checked against the host
    allowlist before it is sent.  Discovery follows only existing ``wikilink1``
    entries found in DokuWiki's index tree; ordinary article links and template
    navigation are deliberately ignored.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        max_redirects: int = 3,
        max_namespaces: int = 2_000,
        max_pages: int = 50_000,
        concurrency: int = 8,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.5,
        allowed_hosts: Iterable[str] | None = None,
        excluded_namespaces: Iterable[str] = ("wiki", "system", "playground"),
        max_response_bytes: int = 10 * 1024 * 1024,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cleaned_base = base_url.strip()
        parsed = urlparse(cleaned_base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain credentials")

        self.base_url = cleaned_base.rstrip("/") + "/"
        self.allowed_hosts = frozenset(
            item.strip().rstrip(".").casefold()
            for item in (allowed_hosts or (parsed.hostname,))
            if item.strip()
        )
        if parsed.hostname.rstrip(".").casefold() not in self.allowed_hosts:
            raise ValueError("base_url host must be included in allowed_hosts")

        self.script_url = urljoin(self.base_url, "doku.php")
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_redirects = max(0, int(max_redirects))
        self.max_namespaces = max(1, int(max_namespaces))
        self.max_pages = max(1, int(max_pages))
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.max_response_bytes = max(1_024, int(max_response_bytes))
        self.excluded_namespaces = frozenset(
            value.strip(":").casefold() for value in excluded_namespaces
        )
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "sicorax-rag-indexer/1.0"},
        )

    async def __aenter__(self) -> DokuWikiClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _validate_url(self, url: str) -> str:
        parsed = urlparse(url)
        host = (parsed.hostname or "").rstrip(".").casefold()
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or host not in self.allowed_hosts
            or parsed.username
            or parsed.password
        ):
            raise DokuWikiSecurityError(f"blocked DokuWiki URL: {url!r}")
        return url

    async def _get(self, url: str) -> httpx.Response:
        current = self._validate_url(url)
        for redirect_count in range(self.max_redirects + 1):
            response: httpx.Response | None = None
            for attempt in range(self.retry_attempts + 1):
                try:
                    async with self._semaphore:
                        response = await self._client.get(current, follow_redirects=False)
                except httpx.TimeoutException:
                    if attempt >= self.retry_attempts:
                        raise
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue

                retryable_status = response.status_code == 429 or response.status_code >= 500
                if retryable_status and attempt < self.retry_attempts:
                    retry_after = response.headers.get("retry-after", "")
                    try:
                        delay = min(10.0, max(0.0, float(retry_after)))
                    except ValueError:
                        delay = self.retry_backoff_seconds * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                break

            if response is None:
                raise AssertionError("request loop produced no response")
            if response.is_redirect:
                if redirect_count >= self.max_redirects:
                    raise DokuWikiSecurityError("too many DokuWiki redirects")
                location = response.headers.get("location")
                if not location:
                    raise DokuWikiSecurityError("DokuWiki redirect has no location")
                current = self._validate_url(urljoin(current, location))
                continue

            response.raise_for_status()
            self._validate_url(str(response.url))
            if len(response.content) > self.max_response_bytes:
                raise DokuWikiError(
                    f"DokuWiki response exceeds {self.max_response_bytes} bytes"
                )
            return response
        raise AssertionError("unreachable")

    @staticmethod
    def _clean_id(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().strip(":")
        if not cleaned or not _SAFE_DOKUWIKI_ID.fullmatch(cleaned):
            return None
        return cleaned

    def _is_excluded(self, wiki_id: str) -> bool:
        root = wiki_id.casefold().split(":", 1)[0]
        return root in self.excluded_namespaces

    def _index_url(self, namespace: str = "") -> str:
        query: dict[str, str] = {"id": "start", "do": "index"}
        if namespace:
            query["idx"] = namespace
        return f"{self.script_url}?{urlencode(query)}"

    def page_url(self, page_id: str) -> str:
        cleaned = self._clean_id(page_id)
        if cleaned is None:
            raise ValueError(f"invalid DokuWiki page id: {page_id!r}")
        return f"{self.script_url}?{urlencode({'id': cleaned})}"

    def _export_url(self, page_id: str) -> str:
        cleaned = self._clean_id(page_id)
        if cleaned is None:
            raise ValueError(f"invalid DokuWiki page id: {page_id!r}")
        return f"{self.script_url}?{urlencode({'do': 'export_raw', 'id': cleaned})}"

    def _parse_index(self, html: str) -> tuple[set[str], set[str]]:
        parser = _IndexLinkCollector()
        parser.feed(html)
        parser.close()

        namespaces: set[str] = set()
        page_ids: set[str] = set()
        for href, classes, title, _text in parser.links:
            absolute = self._validate_url(urljoin(self.base_url, href))
            parsed = urlparse(absolute)
            if parsed.path.rstrip("/") != urlparse(self.script_url).path.rstrip("/"):
                continue
            query = parse_qs(parsed.query)
            if "idx_dir" in classes:
                namespace = self._clean_id((query.get("idx") or [None])[0])
                if namespace and not self._is_excluded(namespace):
                    namespaces.add(namespace)
                continue
            if "wikilink1" not in classes:
                continue
            page_id = self._clean_id(title or (query.get("id") or [None])[0])
            if page_id and not self._is_excluded(page_id):
                page_ids.add(page_id)
        return namespaces, page_ids

    async def discover_page_ids(self) -> list[str]:
        pending = [""]
        visited: set[str] = set()
        pages: set[str] = set()

        while pending:
            namespace = pending.pop(0)
            if namespace in visited:
                continue
            if len(visited) >= self.max_namespaces:
                raise DokuWikiDiscoveryLimitError(
                    f"DokuWiki namespace limit reached ({self.max_namespaces})"
                )
            visited.add(namespace)

            response = await self._get(self._index_url(namespace))
            content_type = response.headers.get("content-type", "").casefold()
            if "html" not in content_type and not response.text.lstrip().startswith("<"):
                raise DokuWikiError(f"index is not HTML: {content_type or 'unknown'}")
            discovered_namespaces, discovered_pages = self._parse_index(response.text)
            pages.update(discovered_pages)
            if len(pages) > self.max_pages:
                raise DokuWikiDiscoveryLimitError(
                    f"DokuWiki page limit reached ({self.max_pages})"
                )
            pending.extend(sorted(discovered_namespaces - visited - set(pending)))

        return sorted(pages)

    async def fetch_page(self, page_id: str) -> WikiPage:
        cleaned = self._clean_id(page_id)
        if cleaned is None or self._is_excluded(cleaned):
            raise ValueError(f"invalid or excluded DokuWiki page id: {page_id!r}")
        response = await self._get(self._export_url(cleaned))
        content_type = response.headers.get("content-type", "").casefold()
        if "text/plain" not in content_type:
            raise DokuWikiError(
                f"raw export for {cleaned!r} is not text/plain: "
                f"{content_type or 'unknown'}"
            )
        raw_text = response.text.replace("\r\n", "\n").replace("\r", "\n")
        heading = _HEADING.search(raw_text)
        title = " ".join(heading.group(1).split()) if heading else cleaned.rsplit(":", 1)[-1]
        return WikiPage(
            page_id=cleaned,
            title=title,
            source_url=self.page_url(cleaned),
            module=module_for_page_id(cleaned),
            raw_text=raw_text,
            content_hash=sha256(raw_text.encode("utf-8")).hexdigest(),
        )

    async def fetch_pages(self, page_ids: Iterable[str] | None = None) -> list[WikiPage]:
        ids = list(page_ids) if page_ids is not None else await self.discover_page_ids()
        # Gather is strict on purpose: callers doing a destructive mirror sync must
        # decide explicitly how partial fetch failures should be handled.
        return list(await asyncio.gather(*(self.fetch_page(page_id) for page_id in ids)))
