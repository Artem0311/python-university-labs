
"""Tests for the findex command-line interface."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index
from findex.store import load_index, save_index

runner = CliRunner()


@pytest.fixture
def saved_index(
    sample_index: Index,
    tmp_path: Path,
) -> Path:
    """Save a test index for CLI commands."""
    path = tmp_path / "index.bin"
    save_index(sample_index, path)
    return path


def test_cli_help() -> None:
    """The CLI must display its available commands."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout


def test_cli_index(
    sample_corpus: Path,
    tmp_path: Path,
) -> None:
    """The index command must create a saved index."""
    output_path = tmp_path / "created.json"

    result = runner.invoke(
        app,
        [
            "index",
            str(sample_corpus),
            "--out",
            str(output_path),
            "--positions",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert load_index(output_path).num_docs == 3


def test_cli_search_table(
    saved_index: Path,
    sample_corpus: Path,
) -> None:
    """Search must display a Rich results table."""
    result = runner.invoke(
        app,
        [
            "search",
            str(saved_index),
            "python",
            "--k",
            "2",
            "--corpus-root",
            str(sample_corpus),
        ],
    )

    assert result.exit_code == 0
    assert "Search results" in result.stdout
    assert "a.txt" in result.stdout


def test_cli_search_json(
    saved_index: Path,
    sample_corpus: Path,
) -> None:
    """JSONL output must contain valid JSON objects."""
    result = runner.invoke(
        app,
        [
            "search",
            str(saved_index),
            "python",
            "--json",
            "--corpus-root",
            str(sample_corpus),
        ],
    )

    assert result.exit_code == 0

    lines = result.stdout.strip().splitlines()
    data = [json.loads(line) for line in lines]

    assert len(data) == 2
    assert all("doc_id" in item for item in data)
    assert all("score" in item for item in data)


def test_cli_stats(saved_index: Path) -> None:
    """Stats must display information about the index."""
    result = runner.invoke(
        app,
        ["stats", str(saved_index)],
    )

    assert result.exit_code == 0
    assert "Index statistics" in result.stdout
    assert "Documents" in result.stdout


def test_cli_invalid_query(saved_index: Path) -> None:
    """Invalid queries must fail without a traceback."""
    result = runner.invoke(
        app,
        ["search", str(saved_index), "python AND"],
    )

    assert result.exit_code == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_missing_index(tmp_path: Path) -> None:
    """Missing indexes must produce a controlled error."""
    missing = tmp_path / "missing.bin"

    result = runner.invoke(
        app,
        ["search", str(missing), "python"],
    )

    assert result.exit_code == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
