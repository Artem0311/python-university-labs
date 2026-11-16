from pathlib import Path

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.search import search


def test_benchmark_search(benchmark, sample_index: Index, sample_corpus: Path) -> None:
    results = benchmark(search, sample_index, "python", corpus_root=sample_corpus)
    assert results


def test_benchmark_build_index(benchmark, sample_corpus: Path) -> None:
    index = benchmark(lambda: build_index(iter_documents(sample_corpus)))
    assert index.num_docs == 3
