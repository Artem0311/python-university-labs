
"""Read text and JSONL documents lazily."""

import json
import logging
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import NamedTuple, cast
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    """A document's ID, source path, and contents."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents from .txt files and crawler JSONL."""

    root = Path(root)

    if root.is_file() and root.suffix == ".jsonl":
        paths = iter([root])
    elif root.is_dir():
        paths = chain(root.rglob("*.txt"), root.rglob("*.jsonl"))
    else:
        raise NotADirectoryError(f"Corpus not found: {root}")

    for path in paths:
        if not path.is_file():
            continue

        if path.suffix == ".jsonl":
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            data: object = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSONL in %s", path)
                            continue

                        if not isinstance(data, dict):
                            continue

                        record = cast(dict[str, object], data)

                        url = record.get("url")
                        text = record.get("text")

                        if not isinstance(url, str):
                            continue

                        if not isinstance(text, str):
                            continue

                        source = Path(urlsplit(url).path or "/index")

                        yield Document(
                            doc_id=url,
                            path=source,
                            text=text,
                        )

            except (OSError, UnicodeError) as exc:
                logger.warning("Cannot read %s: %s", path, exc)

        else:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                logger.warning(
                    "Skipping unreadable document %s: %s",
                    path,
                    exc,
                )
                continue

            yield Document(
                doc_id=path.relative_to(root).as_posix(),
                path=path,
                text=text,
            )
