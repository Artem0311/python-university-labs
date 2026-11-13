
"""Tests for saving and loading indexes."""

from pathlib import Path

import pytest

from findex.index import Index
from findex.store import load_index, save_index


@pytest.mark.parametrize("extension", [".json", ".bin"])
def test_save_load_index(
    sample_index: Index,
    tmp_path: Path,
    extension: str,
) -> None:
    """Saved indexes must be restored without data loss."""
    path = tmp_path / f"index{extension}"

    save_index(sample_index, path)
    restored = load_index(path)

    assert path.exists()
    assert restored == sample_index


def test_loaded_index_preserves_positions(
    sample_index: Index,
    tmp_path: Path,
) -> None:
    """JSON indexes must preserve token positions."""
    path = tmp_path / "index.json"

    save_index(sample_index, path)
    restored = load_index(path)

    assert restored.positions == sample_index.positions


def test_load_missing_index(tmp_path: Path) -> None:
    """Loading a missing index must raise FileNotFoundError."""
    path = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError):
        load_index(path)
