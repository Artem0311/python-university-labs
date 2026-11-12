"""Завантаження навчального корпусу з української Вікіпедії для findex.

Запуск із каталогу findex: uv run python fetch_corpus.py
Файли зберігаються в data/wiki/ та не додаються до Git.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API = "https://uk.wikipedia.org/w/api.php"
USER_AGENT = (
    "findex-lab01/0.1 (university educational corpus; "
    "https://github.com/Artem0311/python-university-labs)"
)
TOPICS = (
    "програмування",
    "алгоритм",
    "програмне забезпечення",
    "комп'ютерні науки",
    "інформаційні технології",
)
TARGET = 100
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "wiki"


def api_request(**params: str | int) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    request = Request(
        API + "?" + urlencode(params),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            return payload
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Не вдалося отримати дані з Вікіпедії: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    titles: list[str] = []
    seen: set[str] = set()
    for topic in TOPICS:
        result = api_request(action="query", list="search", srsearch=topic,
                             srnamespace=0, srlimit=35)
        for hit in result.get("query", {}).get("search", []):
            title = hit["title"]
            if title not in seen:
                seen.add(title)
                titles.append(title)
            if len(titles) >= TARGET + 20:
                break
        time.sleep(0.2)

    count = 0
    sources: list[str] = ["file\ttitle\turl"]
    for offset in range(0, len(titles), 10):
        batch = titles[offset:offset + 10]
        data = api_request(action="query", prop="extracts", titles="|".join(batch),
                           exlimit="max", explaintext=1, redirects=1)
        for page in data.get("query", {}).get("pages", []):
            title = page.get("title", "")
            content = page.get("extract", "").strip()
            if len(content) < 150:  # пропускаємо порожні сторінки й короткі заглушки
                continue
            count += 1
            safe_name = re.sub(r"[^\w-]+", "_", title, flags=re.UNICODE).strip("_")[:65]
            filename = f"{count:04d}_{safe_name}.txt"
            (OUTPUT_DIR / filename).write_text(content + "\n", encoding="utf-8")
            url = "https://uk.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            sources.append(f"{filename}\t{title}\t{url}")
            if count >= TARGET:
                break
        print(f"Завантажено документів: {count}")
        if count >= TARGET:
            break
        time.sleep(0.2)

    (OUTPUT_DIR / "sources.tsv").write_text("\n".join(sources) + "\n", encoding="utf-8")
    if count == 0:
        raise SystemExit("Не отримано текстів. Перевірте інтернет і доступ до Вікіпедії.")
    print(f"Готово: {count} текстів у data/wiki/ (джерела: sources.tsv)")
    print("Джерело: українська Вікіпедія (CC BY-SA), https://uk.wikipedia.org/")


if __name__ == "__main__":
    main()
