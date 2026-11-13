
import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str):
        self.client = client
        self.user_agent = user_agent
        self.cache: dict[str, RobotFileParser | bool] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)

        if parts.scheme not in ("http", "https") or not parts.netloc:
            return False

        host = f"{parts.scheme}://{parts.netloc.lower()}"

        if host not in self.cache:
            lock = self.locks.setdefault(host, asyncio.Lock())

            async with lock:
                if host not in self.cache:
                    try:
                        async with asyncio.timeout(10):
                            response = await self.client.get(
                                f"{host}/robots.txt",
                                follow_redirects=True,
                            )

                        if response.status_code == 404:
                            self.cache[host] = True
                        elif response.status_code == 200:
                            parser = RobotFileParser()
                            parser.parse(response.text.splitlines())
                            self.cache[host] = parser
                        else:
                            self.cache[host] = False

                    except (TimeoutError, httpx.RequestError):
                        self.cache[host] = False

        rules = self.cache[host]

        if isinstance(rules, bool):
            return rules

        return rules.can_fetch(self.user_agent, url)
