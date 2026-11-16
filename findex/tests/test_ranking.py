
"""Tests for TF-IDF and BM25 ranking."""

from pathlib import Path

import pytest

from findex.index import Index, Posting
from findex.query import parse
from findex.ranking import BM25, Scorer, TfIdf
from findex.search import search


@pytest.fixture(
    params=[
        pytest.param(TfIdf(), id="tfidf"),
        pytest.param(BM25(), id="bm25"),
    ]
)
def scorer(request: pytest.FixtureRequest) -> Scorer:
    """Provide both ranking implementations."""
    return request.param


def get_posting(
    index: Index,
    term: str,
    filename: str,
) -> Posting:
    """Find a posting by term and document filename."""
    doc_id = next(
        doc_id
        for doc_id, meta in index.doc_meta.items()
        if meta.path == filename
    )

    return next(
        posting
        for posting in index.postings[term]
        if posting.doc_id == doc_id
    )


def test_positive_score(
    sample_index: Index,
    scorer: Scorer,
) -> None:
    """Existing terms must receive a positive score."""
    posting = get_posting(sample_index, "python", "a.txt")

    assert scorer.score("python", posting, sample_index) > 0


def test_zero_frequency_score(
    sample_index: Index,
    scorer: Scorer,
) -> None:
    """Zero term frequency must produce zero relevance."""
    posting = get_posting(sample_index, "python", "a.txt")
    empty_posting = Posting(doc_id=posting.doc_id, tf=0)

    assert scorer.score("python", empty_posting, sample_index) == 0.0


def test_higher_frequency_increases_score(
    sample_index: Index,
    scorer: Scorer,
) -> None:
    """A higher term frequency must increase relevance."""
    posting = get_posting(sample_index, "python", "a.txt")

    lower = Posting(doc_id=posting.doc_id, tf=1)
    higher = Posting(doc_id=posting.doc_id, tf=2)

    assert scorer.score(
        "python", higher, sample_index
    ) > scorer.score(
        "python", lower, sample_index
    )


def test_ranked_search_order(
    sample_index: Index,
    sample_corpus: Path,
    scorer: Scorer,
) -> None:
    """Ranked search must return the most relevant document first."""
    results = search(
        sample_index,
        "python",
        scorer=scorer,
        k=2,
        corpus_root=sample_corpus,
    )

    assert len(results) == 2
    assert results[0].path == "a.txt"
    assert results[0].score >= results[1].score


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("python", ("python",)),
        ("python OR code", ("python", "code")),
        ("python AND code", ("python", "code")),
    ],
)
def test_numpy_ranking_matches_previous(
    sample_index: Index,
    sample_corpus: Path,
    scorer: Scorer,
    query: str,
    terms: tuple[str, ...],
) -> None:
    matching = parse(query).evaluate(sample_index)
    scores = {
        doc_id: sum(
            scorer.score(term, posting, sample_index)
            for term in terms
            for posting in sample_index.postings.get(term, [])
            if posting.doc_id == doc_id
        )
        for doc_id in matching
    }
    expected = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )
    actual = search(
        sample_index,
        query,
        scorer=scorer,
        k=3,
        corpus_root=sample_corpus,
    )
    assert [result.doc_id for result in actual] == expected
