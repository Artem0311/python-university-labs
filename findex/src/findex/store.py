"""Save and load inverted indexes using pickle or JSON (Lab 02)."""

import json
import pickle
from pathlib import Path

from findex.index import DocMeta, Index, Posting


def save_index(index: Index, path: str | Path) -> None:
    """Save an index as JSON for .json files, otherwise as binary pickle."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = {
            "postings": {
                term: [[p.doc_id, p.tf] for p in postings]
                for term, postings in index.postings.items()
            },
            "doc_lengths": index.doc_lengths,
            "doc_meta": {
                doc_id: {"path": meta.path, "title": meta.title}
                for doc_id, meta in index.doc_meta.items()
            },
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
    else:
        with path.open("wb") as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str | Path) -> Index:
    """Load an index from JSON or pickle.

    SECURITY: Never unpickle untrusted files. pickle.load() can execute
    arbitrary code while loading a malicious file; use only your own index.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return Index(
            postings={
                term: [Posting(doc_id=doc_id, tf=tf) for doc_id, tf in postings]
                for term, postings in payload["postings"].items()
            },
            doc_lengths={int(doc_id): length for doc_id, length in payload["doc_lengths"].items()},
            doc_meta={
                int(doc_id): DocMeta(path=meta["path"], title=meta["title"])
                for doc_id, meta in payload["doc_meta"].items()
            },
        )

    with path.open("rb") as file:
        result = pickle.load(file)
    if not isinstance(result, Index):
        raise ValueError("File does not contain a findex index")
    return result
