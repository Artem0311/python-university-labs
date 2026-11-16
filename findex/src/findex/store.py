
"""Save and load inverted indexes using pickle or JSON."""

import json
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

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


def save_gap_memmap(index: Index, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gaps_parts: list[NDArray[np.int32]] = []
    tf_parts: list[NDArray[np.int32]] = []
    offsets: dict[str, tuple[int, int]] = {}
    position = 0

    for term, (doc_ids, tfs) in sorted(index.numpy_postings.items()):
        order = np.argsort(doc_ids)
        sorted_ids = doc_ids[order]
        gaps_parts.append(np.diff(sorted_ids, prepend=np.int32(0)))
        tf_parts.append(tfs[order])
        offsets[term] = (position, position + len(doc_ids))
        position += len(doc_ids)

    gaps = (
        np.concatenate(gaps_parts)
        if gaps_parts else np.empty(0, dtype=np.int32)
    )
    tfs = (
        np.concatenate(tf_parts)
        if tf_parts else np.empty(0, dtype=np.int32)
    )

    max_gap = int(np.max(gaps, initial=0))
    max_tf = int(np.max(tfs, initial=0))
    gap_dtype = np.uint8 if max_gap <= 255 else (
        np.uint16 if max_gap <= 65535 else np.uint32
    )
    tf_dtype = np.uint8 if max_tf <= 255 else (
        np.uint16 if max_tf <= 65535 else np.uint32
    )

    np.save(directory / "gaps.npy", gaps.astype(gap_dtype))
    np.save(directory / "tfs.npy", tfs.astype(tf_dtype))
    (directory / "terms.json").write_text(
        json.dumps(offsets, ensure_ascii=False),
        encoding="utf-8",
    )


def load_gap_memmap(
    directory: str | Path,
) -> dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]]:
    directory = Path(directory)
    gaps = np.load(directory / "gaps.npy", mmap_mode="r")
    tfs = np.load(directory / "tfs.npy", mmap_mode="r")
    offsets = cast(
        dict[str, list[int]],
        json.loads((directory / "terms.json").read_text(encoding="utf-8")),
    )

    return {
        term: (
            np.cumsum(gaps[start:end], dtype=np.int32),
            np.asarray(tfs[start:end], dtype=np.int32),
        )
        for term, (start, end) in offsets.items()
    }
