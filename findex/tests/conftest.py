
"""Shared fixtures for findex tests."""

from pathlib import Path

import pytest

from findex.index import Index, index_corpus


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a small corpus with three text documents."""
    documents = {
        "a.txt": "python python code",
        "b.txt": "python tests",
        "c.txt": "code tests",
    }

    for filename, content in documents.items():
        (tmp_path / filename).write_text(
            content,
            encoding="utf-8",
        )

    return tmp_path


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    """Build a positional index for the sample corpus."""
    return index_corpus(
        sample_corpus,
        with_positions=True,
    )
