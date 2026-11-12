"""Обов'язкові перевірки потокового токенізатора (M2)."""

from collections.abc import Iterator

import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python PYTHON python", ["python", "python", "python"]),
        ("Привіт, світе!", ["привіт", "світе"]),
        ("п'ять п’ять пʼять", ["п'ять", "п'ять", "п'ять"]),
        ("веб-застосунок та пошук—індекс", ["веб", "застосунок", "та", "пошук", "індекс"]),
        ("2026 Python3 версія_2", ["2026", "python3", "версія", "2"]),
        ("cafe\u0301 café", ["café", "café"]),
        ("  \n...!!! ", []),
    ],
)
def test_tokenize_rules(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_is_lazy_iterator() -> None:
    result = tokenize("Один два")
    assert isinstance(result, Iterator)
    assert next(result) == "один"
    assert next(result) == "два"
    with pytest.raises(StopIteration):
        next(result)
