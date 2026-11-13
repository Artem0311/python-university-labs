
"""Build inverted indexes from documents and file paths."""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Literal

from findex.corpus import Document
from findex.timing import timed
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

ExecutorKind = Literal["serial", "threads", "processes"]


@dataclass(slots=True)
class BuildMetrics:
    """Timing measurements for index construction."""

    merge_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Posting:
    """A document ID and the frequency of a term."""

    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Relative document path and title."""

    path: str
    title: str


def _empty_positions() -> dict[str, dict[int, tuple[int, ...]]]:
    """Create a correctly typed empty positions dictionary."""
    return {}


@dataclass
class Index(Mapping[str, list[Posting]]):
    """An inverted index with Python mapping behavior."""

    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    positions: dict[str, dict[int, tuple[int, ...]]] = field(
        default_factory=_empty_positions
    )

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, documents={self.num_docs})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        if self.num_docs == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        """Return the number of tokens in a document."""
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Return the number of documents containing a term."""
        return len(self.postings.get(term, []))

    def __getstate__(self) -> dict[str, object]:
        """Exclude runtime caches from saved pickle indexes."""
        return {
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "doc_meta": self.doc_meta,
            "positions": self.positions,
        }

    def __setstate__(
        self,
        state: dict[str, object] | tuple[object, dict[str, object]],
    ) -> None:
        """Support old Lab 02 indexes without positions."""
        if isinstance(state, tuple):
            _, slot_values = state
            self.__dict__.update(slot_values)
        else:
            self.__dict__.update(state)

        self.__dict__.setdefault("positions", {})


def _build_from_documents(
    documents: Iterable[tuple[int, Document]],
    *,
    with_positions: bool = False,
) -> Index:
    """Build an index from documents with assigned numeric IDs."""

    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    positions: dict[str, dict[int, tuple[int, ...]]] = {}

    for doc_id, document in documents:
        tokens = list(tokenize(document.text))
        counts = Counter(tokens)

        doc_lengths[doc_id] = len(tokens)
        doc_meta[doc_id] = DocMeta(
            path=document.doc_id,
            title=document.path.stem,
        )

        for term, frequency in counts.items():
            postings[term].append(
                Posting(doc_id=doc_id, tf=frequency)
            )

        if with_positions:
            term_positions: defaultdict[str, list[int]] = defaultdict(list)

            for position, term in enumerate(tokens):
                term_positions[term].append(position)

            for term, offsets in term_positions.items():
                positions.setdefault(term, {})[doc_id] = tuple(offsets)

    return Index(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        positions=positions,
    )


def build_partial(
    paths: Sequence[Path],
    *,
    root: Path,
    start_doc_id: int = 0,
    with_positions: bool = False,
) -> Index:
    """Build a partial index by reading files inside the worker."""
    if start_doc_id < 0:
        raise ValueError("start_doc_id must be non-negative")

    root = Path(root)

    def numbered_documents() -> Iterator[tuple[int, Document]]:
        for offset, source in enumerate(paths):
            path = Path(source)

            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                logger.warning(
                    "Skipping unreadable document %s: %s",
                    path,
                    exc,
                )
                continue

            document = Document(
                doc_id=path.relative_to(root).as_posix(),
                path=path,
                text=text,
            )
            yield start_doc_id + offset, document

    return _build_from_documents(
        numbered_documents(),
        with_positions=with_positions,
    )


def merge(partials: Iterable[Index]) -> Index:
    """Merge partial indexes and preserve ascending document IDs."""

    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    positions: dict[str, dict[int, tuple[int, ...]]] = {}

    for partial in partials:
        for doc_id, meta in partial.doc_meta.items():
            if doc_id in doc_meta:
                raise ValueError(f"Duplicate document ID: {doc_id}")
            doc_meta[doc_id] = meta

        doc_lengths.update(partial.doc_lengths)

        for term, entries in partial.postings.items():
            postings[term].extend(entries)

        for term, term_positions in partial.positions.items():
            positions.setdefault(term, {}).update(term_positions)

    id_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(doc_meta))
    }

    merged_postings = {
        term: sorted(
            (
                Posting(
                    doc_id=id_mapping[posting.doc_id],
                    tf=posting.tf,
                )
                for posting in entries
            ),
            key=lambda posting: posting.doc_id,
        )
        for term, entries in postings.items()
    }

    merged_lengths = {
        id_mapping[doc_id]: length
        for doc_id, length in doc_lengths.items()
    }

    merged_meta = {
        id_mapping[doc_id]: meta
        for doc_id, meta in doc_meta.items()
    }

    merged_positions = {
        term: {
            id_mapping[doc_id]: offsets
            for doc_id, offsets in term_positions.items()
        }
        for term, term_positions in positions.items()
    }

    return Index(
        postings=merged_postings,
        doc_lengths=merged_lengths,
        doc_meta=merged_meta,
        positions=merged_positions,
    )


def _merge_with_metrics(
    partials: Iterable[Index],
    metrics: BuildMetrics | None,
) -> Index:
    """Measure merging separately from partial index construction."""
    started = perf_counter()
    result = merge(partials)
    if metrics is not None:
        metrics.merge_seconds = perf_counter() - started
    return result


@timed
def build_index(
    documents: Iterable[Document],
    *,
    with_positions: bool = False,
) -> Index:
    """Build an index using the original document-based API."""
    partial = _build_from_documents(
        enumerate(documents),
        with_positions=with_positions,
    )
    return merge([partial])


@timed
def index_corpus(
    root: Path,
    *,
    with_positions: bool = False,
    workers: int = 1,
    executor: ExecutorKind = "serial",
    limit: int | None = None,
    metrics: BuildMetrics | None = None,
) -> Index:
    """Build a corpus index using serial, thread, or process execution."""
    root = Path(root)

    if not root.is_dir():
        raise NotADirectoryError(
            f"Corpus directory does not exist: {root}"
        )

    if workers < 1:
        raise ValueError("workers must be at least 1")

    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Unknown executor: {executor}")

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    paths = [
        path
        for path in root.rglob("*.txt")
        if path.is_file()
    ]

    if limit is not None:
        paths = paths[:limit]

    if executor == "serial" or not paths:
        partial = build_partial(
            paths,
            root=root,
            start_doc_id=0,
            with_positions=with_positions,
        )
        return _merge_with_metrics([partial], metrics)

    # Distribute consecutive path ranges among workers.
    chunk_size = max(1, (len(paths) + workers - 1) // workers)

    chunks = [
        (start, paths[start:start + chunk_size])
        for start in range(0, len(paths), chunk_size)
    ]

    pool_type = (
        ThreadPoolExecutor
        if executor == "threads"
        else ProcessPoolExecutor
    )

    with pool_type(max_workers=workers) as pool:
        futures = [
            pool.submit(
                build_partial,
                chunk,
                root=root,
                start_doc_id=start,
                with_positions=with_positions,
            )
            for start, chunk in chunks
        ]

        # result() propagates exceptions raised inside workers.
        partials = [future.result() for future in futures]

    return _merge_with_metrics(partials, metrics)


def main() -> None:
    """Build and save an index from the command line."""
    import argparse
    import tracemalloc
    from time import perf_counter

    from findex.store import save_index

    parser = argparse.ArgumentParser(
        description="Build and save a findex index"
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Store token positions for phrase queries",
    )
    args = parser.parse_args()

    tracemalloc.start()
    started = perf_counter()

    try:
        index = index_corpus(
            args.corpus,
            with_positions=args.positions,
        )
        save_index(index, args.out)
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Indexed {index.num_docs} documents")
    print(f"Unique terms: {len(index)}")
    print(f"Positions enabled: {bool(index.positions)}")
    print(f"Saved index: {args.out}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Peak memory: {peak:,} bytes")


if __name__ == "__main__":
    main()
