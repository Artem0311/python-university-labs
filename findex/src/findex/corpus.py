"""Read a plain-text corpus lazily, one document at a time."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    """A document's stable relative ID, source path, and contents."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield UTF-8 ``.txt`` files in a directory tree one by one.

    The generator keeps only one document's contents in memory at a time.
    Files with invalid UTF-8 or read errors are logged and skipped.

    Args:
        root: Directory containing plain-text documents.

    Yields:
        A :class:`Document` for each successfully read file.

    Raises:
        NotADirectoryError: If the root is not an existing directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory does not exist: {root}")

    for path in root.rglob("*.txt"):
        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            logger.warning("Skipping unreadable document %s: %s", path, exc)
            continue

        yield Document(doc_id=path.relative_to(root).as_posix(), path=path, text=text)
