from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from time import monotonic
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import httpx


@dataclass(slots=True)
class SkbSearchResult:
    title: str
    url: str
    snippet: str
    score: int


class _HtmlCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._anchor_text: list[str] = []
        self._title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a":
            self._href = attrs_dict.get("href")
            self._anchor_text = []
        elif tag == "title":
            self._title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            text = " ".join(self._anchor_text).strip()
            self.links.append((self._href, text))
            self._href = None
            self._anchor_text = []
        elif tag == "title":
            self._title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        self.text_parts.append(text)
        if self._href is not None:
            self._anchor_text.append(text)
        if self._title:
            self.title_parts.append(text)


class SkbClient:
    """Read-only client for the SKB knowledge site.

    The site is treated as runtime data: module names are discovered from SKB and cached,
    rather than hard-coded in the assistant. Search uses a bounded same-origin crawl so it
    works even when SKB does not expose a dedicated search API.
    """

    _GENERIC_NAV = {
        "home",
        "accueil",
        "login",
        "logout",
        "sign in",
        "search",
        "recherche",
        "help",
        "aide",
        "contact",
        "about",
        "previous",
        "next",
        "back",
        "menu",
    }

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 5.0,
        module_cache_seconds: int = 600,
        search_max_pages: int = 30,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/") + "/"
        self.module_cache_seconds = max(1, module_cache_seconds)
        self.search_max_pages = max(1, search_max_pages)
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "rag-skb-agent/1.0"},
        )
        self._modules_cache: list[str] = []
        self._modules_cached_at = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def host(self) -> str:
        return urlparse(self.base_url).netloc.casefold()

    def _same_origin_url(self, href: str, current_url: str) -> str | None:
        href = (href or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            return None
        absolute = urljoin(current_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() != self.host:
            return None
        return parsed._replace(fragment="").geturl()

    async def _fetch_html(self, url: str) -> tuple[str, str]:
        response = await self._client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.casefold() and not response.text.lstrip().startswith("<"):
            raise ValueError(f"skb_non_html_response:{content_type or 'unknown'}")
        return response.url.__str__(), response.text

    @staticmethod
    def _parse_html(html: str) -> _HtmlCollector:
        parser = _HtmlCollector()
        parser.feed(html)
        parser.close()
        return parser

    @classmethod
    def _module_candidates_from_html(cls, html: str, page_url: str) -> list[str]:
        parser = cls._parse_html(html)
        candidates: list[str] = []

        # Common JSON/script payload shapes used by SPAs and server-rendered apps.
        for pattern in (
            r'"moduleName"\s*:\s*"([^"]{2,80})"',
            r'"module_name"\s*:\s*"([^"]{2,80})"',
            r'"moduleLabel"\s*:\s*"([^"]{2,80})"',
        ):
            candidates.extend(re.findall(pattern, html, flags=re.IGNORECASE))

        for href, anchor_text in parser.links:
            text = " ".join(anchor_text.split()).strip()
            if not text or len(text) > 80 or text.casefold() in cls._GENERIC_NAV:
                continue

            absolute = urljoin(page_url, href)
            parsed = urlparse(absolute)
            path = parsed.path.strip("/")
            query = parse_qs(parsed.query)
            path_lower = path.casefold()

            explicitly_module_like = (
                "module" in path_lower
                or "module" in {key.casefold() for key in query}
                or any("module" in value.casefold() for values in query.values() for value in values)
            )
            shallow_navigation = bool(path) and len([part for part in path.split("/") if part]) <= 2

            if explicitly_module_like or shallow_navigation:
                candidates.append(text)

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = " ".join(candidate.split()).strip(" -–—:|/")
            key = normalized.casefold()
            if len(normalized) < 2 or key in cls._GENERIC_NAV or key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    async def discover_modules(self, *, force_refresh: bool = False) -> list[str]:
        now = monotonic()
        if (
            not force_refresh
            and self._modules_cache
            and now - self._modules_cached_at < self.module_cache_seconds
        ):
            return list(self._modules_cache)

        page_url, html = await self._fetch_html(self.base_url)
        modules = self._module_candidates_from_html(html, page_url)
        self._modules_cache = modules
        self._modules_cached_at = now
        return list(modules)

    @staticmethod
    def _plain_text(parser: _HtmlCollector) -> str:
        return " ".join(parser.text_parts)

    @staticmethod
    def _score_text(text: str, query: str, module: str | None) -> int:
        haystack = text.casefold()
        terms = [term for term in re.findall(r"[\w'-]+", query.casefold()) if len(term) > 1]
        score = sum(haystack.count(term) for term in terms)
        if module:
            score += 5 * haystack.count(module.casefold())
        return score

    @staticmethod
    def _snippet(text: str, query: str, limit: int = 500) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        terms = [term for term in re.findall(r"[\w'-]+", query.casefold()) if len(term) > 1]
        lower = compact.casefold()
        positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
        center = min(positions) if positions else 0
        start = max(0, center - limit // 3)
        end = min(len(compact), start + limit)
        prefix = "…" if start else ""
        suffix = "…" if end < len(compact) else ""
        return prefix + compact[start:end].strip() + suffix

    async def search(
        self,
        query: str,
        *,
        module: str | None = None,
        limit: int = 5,
    ) -> list[SkbSearchResult]:
        query = " ".join(query.split()).strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 10))

        queue = [self.base_url]
        visited: set[str] = set()
        results: list[SkbSearchResult] = []

        while queue and len(visited) < self.search_max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                final_url, html = await self._fetch_html(url)
            except Exception:
                continue

            parser = self._parse_html(html)
            text = self._plain_text(parser)
            score = self._score_text(text, query, module)
            if score > 0:
                title = " ".join(parser.title_parts).strip() or final_url
                results.append(
                    SkbSearchResult(
                        title=title,
                        url=final_url,
                        snippet=self._snippet(text, query),
                        score=score,
                    )
                )

            for href, _ in parser.links:
                candidate = self._same_origin_url(href, final_url)
                if candidate and candidate not in visited and candidate not in queue:
                    queue.append(candidate)

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]
