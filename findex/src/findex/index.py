"""Build an inverted index from the lazy document pipeline (Lab 02, M1)."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    """A document ID and the frequency of a term in that document."""

    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Relative document path and human-readable title."""

    path: str
    title: str


@dataclass(slots=True)
class Index:
    """The inverted index and document information needed for searching."""

    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


def build_index(documents: Iterable[Document]) -> Index:
    """Build postings in one pass, keeping only one document's counts at a time.

    IDs are assigned in traversal order. As IDs increase, each term's postings
    are already sorted by ``doc_id`` without an additional sorting pass.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, document in enumerate(documents):
        counts = Counter(tokenize(document.text))
        doc_lengths[doc_id] = sum(counts.values())
        doc_meta[doc_id] = DocMeta(
            path=document.doc_id,
            title=document.path.stem,
        )
        for term, frequency in counts.items():
            postings[term].append(Posting(doc_id=doc_id, tf=frequency))

    return Index(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def index_corpus(root: Path) -> Index:
    """Build an index using the existing lazy UTF-8 corpus reader."""
    return build_index(iter_documents(root))

def main() -> None:
    """Build and save an index from the command line."""
    import argparse
    import tracemalloc
    from time import perf_counter

    from findex.index import index_corpus as build_canonical_index
    from findex.store import save_index

    parser = argparse.ArgumentParser(
        description="Build and save a findex index"
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tracemalloc.start()
    started = perf_counter()

    try:
        index = build_canonical_index(args.corpus)
        save_index(index, args.out)
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Indexed {len(index.doc_meta)} documents")
    print(f"Unique terms: {len(index.postings)}")
    print(f"Saved index: {args.out}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Peak memory: {peak:,} bytes")


if __name__ == "__main__":
    main()
