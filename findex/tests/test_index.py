
"""Tests for inverted index construction."""

from pathlib import Path

import pytest

from findex.index import Index, index_corpus


def doc_id_for(index: Index, filename: str) -> int:
    """Find the document ID using its filename."""
    return next(
        doc_id
        for doc_id, meta in index.doc_meta.items()
        if meta.path == filename
    )


def test_index_contains_three_documents(sample_index: Index) -> None:
    assert sample_index.num_docs == 3


def test_index_contains_expected_terms(sample_index: Index) -> None:
    assert set(sample_index.postings) == {
        "python",
        "code",
        "tests",
    }


def test_document_paths(sample_index: Index) -> None:
    paths = {
        meta.path
        for meta in sample_index.doc_meta.values()
    }

    assert paths == {"a.txt", "b.txt", "c.txt"}


def test_document_lengths(sample_index: Index) -> None:
    expected = {
        "a.txt": 3,
        "b.txt": 2,
        "c.txt": 2,
    }

    for filename, length in expected.items():
        doc_id = doc_id_for(sample_index, filename)
        assert sample_index.doc_length(doc_id) == length


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("python", 2),
        ("code", 2),
        ("tests", 2),
        ("missing", 0),
    ],
)
def test_document_frequency(
    sample_index: Index,
    term: str,
    expected: int,
) -> None:
    assert sample_index.df(term) == expected


def test_term_frequency(sample_index: Index) -> None:
    doc_id = doc_id_for(sample_index, "a.txt")

    posting = next(
        posting
        for posting in sample_index.postings["python"]
        if posting.doc_id == doc_id
    )

    assert posting.tf == 2


def test_postings_are_sorted(sample_index: Index) -> None:
    for postings in sample_index.postings.values():
        doc_ids = [posting.doc_id for posting in postings]
        assert doc_ids == sorted(doc_ids)


def test_token_positions(sample_index: Index) -> None:
    doc_id = doc_id_for(sample_index, "a.txt")

    assert sample_index.positions["python"][doc_id] == (0, 1)
    assert sample_index.positions["code"][doc_id] == (2,)


def test_index_without_positions(sample_corpus: Path) -> None:
    index = index_corpus(
        sample_corpus,
        with_positions=False,
    )

    assert index.num_docs == 3
    assert index.positions == {}


def test_empty_corpus(tmp_path: Path) -> None:
    index = index_corpus(tmp_path)

    assert index.num_docs == 0
    assert len(index) == 0
    assert index.avg_doc_length == 0.0


def test_average_document_length(sample_index: Index) -> None:
    assert sample_index.avg_doc_length == pytest.approx(7 / 3)


def test_index_mapping_access(sample_index: Index) -> None:
    assert sample_index["python"] == sample_index.postings["python"]
    assert "python" in sample_index
    assert "missing" not in sample_index
