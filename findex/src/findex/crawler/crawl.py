
"""Polite asynchronous web crawler."""

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.robots import RobotsCache


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.in_title = False
        self.ignored = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in ("script", "style"):
            self.ignored += 1
        elif tag == "title":
            self.in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.ignored = max(0, self.ignored - 1)
        elif tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.ignored:
            return
        value = data.strip()
        if value:
            if self.in_title:
                self.title += value
            self.text_parts.append(value)


def canonicalize(url: str) -> str | None:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname

    if scheme not in ("http", "https") or not host:
        return None

    try:
        port = parts.port
    except ValueError:
        return None

    hostname = host.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"

    default_port = 443 if scheme == "https" else 80
    if port and port != default_port:
        hostname += f":{port}"

    path = re.sub(r"/+", "/", parts.path or "/")
    if path != "/" and "." not in path.rsplit("/", 1)[-1]:
        path = path.rstrip("/") + "/"

    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))

    return urlunsplit((scheme, hostname, path, query, ""))


async def crawl(
    seeds: str | Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: set[str] | None = None,
    user_agent: str = (
        "findex/0.6 "
        "(contact: https://github.com/Artem0311/python-university-labs/issues)"
    ),
    min_delay: float = 0.1,
    stats: CrawlStats | None = None,
    on_fetch: Callable[[FetchResult], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Page]:
    """Stream crawled pages without storing the whole corpus."""

    if max_pages < 1 or concurrency < 1 or per_host < 1:
        raise ValueError("Limits must be positive")
    if min_delay < 0:
        raise ValueError("Delay cannot be negative")

    statistics = stats if stats is not None else CrawlStats()
    seed_urls = [seeds] if isinstance(seeds, str) else list(seeds)

    domains = (
        {domain.lower() for domain in allowed_domains}
        if allowed_domains is not None
        else {
            host
            for seed in seed_urls
            if (host := urlsplit(seed).hostname) is not None
        }
    )

    frontier: asyncio.Queue[str] = asyncio.Queue()
    results: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()

    for seed in seed_urls:
        url = canonicalize(seed)
        if url and urlsplit(url).hostname in domains and url not in seen:
            seen.add(url)
            frontier.put_nowait(url)

    global_limit = asyncio.Semaphore(concurrency)
    host_limits: dict[str, asyncio.Semaphore] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    next_request: dict[str, float] = {}
    started = 0
    loop = asyncio.get_running_loop()

    async def throttle(host: str) -> None:
        lock = host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = next_request.get(host, 0.0) - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            next_request[host] = loop.time() + min_delay

    async with httpx.AsyncClient(
        transport=transport,
        headers={"User-Agent": user_agent},
        follow_redirects=False,
    ) as client:
        robots = RobotsCache(client, user_agent)

        async def worker() -> None:
            nonlocal started

            while True:
                url = await frontier.get()
                statistics.queue_depth = frontier.qsize()

                try:
                    if started >= max_pages:
                        continue

                    host = urlsplit(url).netloc
                    host_limit = host_limits.setdefault(
                        host, asyncio.Semaphore(per_host)
                    )

                    async with global_limit, host_limit:
                        if host not in robots.cache:
                            await throttle(host)

                        if not await robots.allowed(url):
                            continue

                        await throttle(host)

                        if started >= max_pages:
                            continue

                        started += 1
                        statistics.in_flight += 1

                        try:
                            result = await fetch(client, url)
                        finally:
                            statistics.in_flight -= 1

                    if on_fetch is not None:
                        on_fetch(result)

                    if result.status != 200:
                        statistics.errors += 1
                        continue

                    parser = PageParser()
                    parser.feed(
                        result.content.decode("utf-8", errors="replace")
                    )

                    statistics.pages += 1
                    page = Page(
                        url=url,
                        title=parser.title,
                        text=" ".join(parser.text_parts),
                        fetched_at=datetime.now(UTC).isoformat(),
                    )
                    await results.put(page)

                    for href in parser.links:
                        target = canonicalize(urljoin(url, href))
                        if target is None or target in seen:
                            continue
                        if urlsplit(target).hostname not in domains:
                            continue
                        seen.add(target)
                        frontier.put_nowait(target)

                    statistics.queue_depth = frontier.qsize()

                finally:
                    frontier.task_done()

        async def finish() -> None:
            await frontier.join()
            await results.put(None)

        async with asyncio.TaskGroup() as group:
            workers = [
                group.create_task(worker())
                for _ in range(concurrency)
            ]
            monitor = group.create_task(finish())

            try:
                while True:
                    page = await results.get()
                    if page is None:
                        break
                    yield page
            finally:
                monitor.cancel()
                for task in workers:
                    task.cancel()
