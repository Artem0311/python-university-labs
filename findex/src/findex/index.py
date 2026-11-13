
"""Build an inverted index from the lazy document pipeline."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.timing import timed
from findex.tokenize import tokenize


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


@timed
def build_index(
    documents: Iterable[Document],
    *,
    with_positions: bool = False,
) -> Index:
    """Build an inverted index in one pass over the documents."""

    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    positions: dict[str, dict[int, tuple[int, ...]]] = {}

    for doc_id, document in enumerate(documents):
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
            term_positions: defaultdict[str, list[int]] = (
                defaultdict(list)
            )

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


def index_corpus(
    root: Path,
    *,
    with_positions: bool = False,
) -> Index:
    """Build an index using the existing lazy corpus reader."""
    return build_index(
        iter_documents(root),
        with_positions=with_positions,
    )


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
