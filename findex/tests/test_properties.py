
"""Property-based tests using Hypothesis."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import Index, build_index

# Private merge helpers are intentionally tested to verify set operations.
from findex.search import _merge_and, _merge_not, _merge_or  # pyright: ignore[reportPrivateUsage]
from findex.store import load_index, save_index

WORDS = ("python", "code", "test", "search")

DOCUMENTS = st.lists(
    st.lists(st.sampled_from(WORDS), max_size=8),
    max_size=5,
)


def make_index(document_words: list[list[str]]) -> Index:
    """Build an index from generated documents."""
    documents = (
        Document(
            doc_id=f"doc_{i}.txt",
            path=Path(f"doc_{i}.txt"),
            text=" ".join(words),
        )
        for i, words in enumerate(document_words)
    )

    return build_index(documents, with_positions=True)


@settings(max_examples=25)
@given(document_words=DOCUMENTS)
def test_save_load_roundtrip(
    document_words: list[list[str]],
) -> None:
    """Saving and loading must preserve the index."""
    original = make_index(document_words)

    with TemporaryDirectory() as directory:
        for suffix in (".json", ".bin"):
            path = Path(directory) / f"index{suffix}"
            save_index(original, path)
            restored = load_index(path)

            assert restored == original


@settings(max_examples=25)
@given(document_words=DOCUMENTS)
def test_postings_are_always_sorted(
    document_words: list[list[str]],
) -> None:
    """Posting IDs must be sorted and unique."""
    index = make_index(document_words)

    for postings in index.postings.values():
        doc_ids = [posting.doc_id for posting in postings]

        assert doc_ids == sorted(set(doc_ids))


@settings(max_examples=25)
@given(
    left=st.sets(st.integers(0, 100), max_size=20),
    right=st.sets(st.integers(0, 100), max_size=20),
)
def test_merge_matches_set_operations(
    left: set[int],
    right: set[int],
) -> None:
    """Merge algorithms must match Python set operations."""
    a = sorted(left)
    b = sorted(right)

    assert _merge_and(a, b) == sorted(left & right)
    assert _merge_or(a, b) == sorted(left | right)
    assert _merge_not(a, b) == sorted(left - right)
