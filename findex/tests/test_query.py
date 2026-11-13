
"""Tests for Boolean query parsing and evaluation."""

import pytest

from findex.index import DocMeta, Index, Posting
from findex.query import Phrase, Term, parse


def test_complex_query_tree():
    query = 'python AND (async OR await) NOT java "event loop"'

    expected = (
        Term("python")
        & (Term("async") | Term("await"))
        & ~Term("java")
        & Phrase("event loop")
    )

    assert parse(query) == expected


def test_operator_precedence():
    assert parse("python OR java AND async") == (
        Term("python") | (Term("java") & Term("async"))
    )

    assert parse("NOT python OR java") == (
        ~Term("python") | Term("java")
    )


def test_phrase_matching():
    index = Index(
        postings={
            "event": [Posting(0, 1), Posting(1, 1)],
            "loop": [Posting(0, 1), Posting(1, 1)],
        },
        doc_lengths={0: 3, 1: 3},
        doc_meta={
            0: DocMeta("a.txt", "A"),
            1: DocMeta("b.txt", "B"),
        },
        positions={
            "event": {0: (0,), 1: (0,)},
            "loop": {0: (1,), 1: (2,)},
        },
    )

    assert parse('"event loop"').evaluate(index) == {0}

    assert parse(
        'event AND NOT "event loop"'
    ).evaluate(index) == {1}


def test_query_operators():
    assert Term("python") & Term("java") == parse(
        "python AND java"
    )

    assert Term("python") | Term("java") == parse(
        "python OR java"
    )

    assert ~Term("python") == parse("NOT python")


def test_invalid_query():
    with pytest.raises(ValueError):
        parse("python AND")
