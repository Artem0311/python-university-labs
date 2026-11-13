
"""Tests for the asynchronous crawler."""

import asyncio

import httpx

from findex.crawler.crawl import crawl
from findex.crawler.fetch import fetch
from findex.crawler.robots import RobotsCache


def test_fetch_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Hello!")

    async def check() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await fetch(client, "https://example.com/")
            assert result.status == 200
            assert result.content == b"Hello!"
            assert result.error is None

    asyncio.run(check())


def test_fetch_retries_429() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
            )

        return httpx.Response(200, text="OK")

    async def check() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await fetch(client, "https://example.com/")
            assert result.status == 200
            assert attempts == 2

    asyncio.run(check())


def test_fetch_does_not_retry_404() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404)

    async def check() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await fetch(client, "https://example.com/")
            assert result.status == 404
            assert attempts == 1

    asyncio.run(check())


def test_robots_cache() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1

        return httpx.Response(
            200,
            text="User-agent: *\nDisallow: /private\n",
        )

    async def check() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            robots = RobotsCache(client, "findex")

            assert await robots.allowed(
                "https://example.com/public"
            )
            assert not await robots.allowed(
                "https://example.com/private"
            )
            assert requests == 1

    asyncio.run(check())


def test_crawl_page_limit_and_duplicates() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))

        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private\n",
            )

        if request.url.path == "/":
            return httpx.Response(
                200,
                text=(
                    "<title>Home</title>"
                    '<a href="/second">Second</a>'
                    '<a href="/second#fragment">Duplicate</a>'
                    '<a href="/private">Private</a>'
                ),
            )

        return httpx.Response(
            200,
            text="<title>Second</title>Page two",
        )

    async def check() -> None:
        pages = [
            page
            async for page in crawl(
                "https://example.com/",
                max_pages=2,
                concurrency=2,
                per_host=1,
                min_delay=0,
                transport=httpx.MockTransport(handler),
            )
        ]

        assert len(pages) == 2
        assert len({page.url for page in pages}) == 2

        assert sum(
            "/robots.txt" in url for url in requested
        ) == 1

        assert not any(
            "/private" in url for url in requested
        )

    asyncio.run(check())
