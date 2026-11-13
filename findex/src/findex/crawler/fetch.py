
import asyncio
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int | None
    content: bytes
    elapsed: float
    error: str | None = None


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                date = parsedate_to_datetime(retry_after)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                return max(
                    0.0,
                    (date - datetime.now(UTC)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass

    return min(8.0, 2.0**attempt) + random.uniform(0, 1)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
) -> FetchResult:
    started = time.perf_counter()

    for attempt in range(4):
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url)

        except (TimeoutError, httpx.TimeoutException):
            if attempt == 3:
                return FetchResult(
                    url, None, b"",
                    time.perf_counter() - started,
                    "timeout",
                )
            await asyncio.sleep(_retry_delay(attempt))
            continue

        except httpx.RequestError as exc:
            return FetchResult(
                url, None, b"",
                time.perf_counter() - started,
                str(exc),
            )

        if (
            response.status_code == 429
            or response.status_code >= 500
        ) and attempt < 3:
            await asyncio.sleep(
                _retry_delay(
                    attempt,
                    response.headers.get("Retry-After"),
                )
            )
            continue

        return FetchResult(
            url,
            response.status_code,
            response.content,
            time.perf_counter() - started,
        )

    return FetchResult(
        url, None, b"",
        time.perf_counter() - started,
        "retry limit exceeded",
    )
