
"""Save and load inverted indexes using pickle or JSON."""

import json
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from findex.index import DocMeta, Index, Posting
from findex.timing import timed


def save_index(index: Index, path: str | Path) -> None:
    """Save an index as JSON or binary pickle."""
    path = Path(path)

    if path.suffix.lower() == ".json":
        payload = {
            "postings": {
                term: [[p.doc_id, p.tf] for p in postings]
                for term, postings in index.postings.items()
            },
            "doc_lengths": index.doc_lengths,
            "doc_meta": {
                doc_id: {
                    "path": meta.path,
                    "title": meta.title,
                }
                for doc_id, meta in index.doc_meta.items()
            },
            "positions": {
                term: {
                    doc_id: list(offsets)
                    for doc_id, offsets in documents.items()
                }
                for term, documents in index.positions.items()
            },
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
    else:
        with path.open("wb") as file:
            pickle.dump(
                index,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


@contextmanager
def open_index(path: str | Path) -> Generator[Index]:
    """Open an index and safely close its file."""
    path = Path(path)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

            index = Index(
                postings={
                    term: [
                        Posting(doc_id=doc_id, tf=tf)
                        for doc_id, tf in postings
                    ]
                    for term, postings in payload["postings"].items()
                },
                doc_lengths={
                    int(doc_id): length
                    for doc_id, length in payload["doc_lengths"].items()
                },
                doc_meta={
                    int(doc_id): DocMeta(
                        path=meta["path"],
                        title=meta["title"],
                    )
                    for doc_id, meta in payload["doc_meta"].items()
                },
                positions={
                    term: {
                        int(doc_id): tuple(offsets)
                        for doc_id, offsets in documents.items()
                    }
                    for term, documents in payload.get(
                        "positions", {}
                    ).items()
                },
            )

            yield index
    else:
        # Only open pickle files from trusted sources.
        with path.open("rb") as file:
            index = pickle.load(file)

            if not isinstance(index, Index):
                raise TypeError(
                    "File does not contain a findex index"
                )

            yield index


@timed
def load_index(path: str | Path) -> Index:
    """Load a saved index without leaving a file handle open."""
    with open_index(path) as index:
        return index
