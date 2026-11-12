"""Перевірки потокового завантаження документів (M1)."""

import inspect
from pathlib import Path

import pytest

from findex.corpus import iter_documents


def test_iter_documents_is_generator() -> None:
    assert inspect.isgeneratorfunction(iter_documents)


def test_reads_utf8_text_and_relative_id(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Привіт, Python!", encoding="utf-8")

    documents = list(iter_documents(tmp_path))

    assert len(documents) == 1
    assert documents[0].doc_id == "note.txt"
    assert documents[0].path == tmp_path / "note.txt"
    assert documents[0].text == "Привіт, Python!"


def test_finds_nested_txt_files_and_ignores_other_extensions(tmp_path: Path) -> None:
    nested = tmp_path / "notes"
    nested.mkdir()
    (nested / "sample.txt").write_text("Текст", encoding="utf-8")
    (nested / "ignore.md").write_text("Не документ", encoding="utf-8")

    assert [doc.doc_id for doc in iter_documents(tmp_path)] == ["notes/sample.txt"]


def test_skips_bad_utf8_and_keeps_good_documents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    (tmp_path / "good.txt").write_text("Добре", encoding="utf-8")

    documents = list(iter_documents(tmp_path))

    assert [doc.doc_id for doc in documents] == ["good.txt"]
    assert "Skipping unreadable document" in caplog.text


def test_missing_corpus_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list(iter_documents(tmp_path / "absent"))
