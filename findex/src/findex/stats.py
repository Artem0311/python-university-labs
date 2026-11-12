"""Статистика текстового корпусу за один прохід (M3).

Запуск із кореня проєкту: uv run python -m findex.stats data/
Обмеження кількості документів: uv run python -m findex.stats data/ --limit 10
"""

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import NamedTuple

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


class CorpusStats(NamedTuple):
    """Підрахунки за корпусом і 50 найуживаніших термінів."""

    documents: int
    tokens: int
    vocabulary: int
    top_terms: list[tuple[str, int]]


def collect_stats(documents: Iterable[Document]) -> CorpusStats:
    """Обробити потік документів один раз, не створюючи списку токенів."""
    term_counts: Counter[str] = Counter()
    document_count = 0
    token_count = 0

    for document in documents:
        document_count += 1
        for token in tokenize(document.text):
            term_counts[token] += 1
            token_count += 1

    return CorpusStats(
        documents=document_count,
        tokens=token_count,
        vocabulary=len(term_counts),
        top_terms=term_counts.most_common(50),
    )


def positive_int(raw: str) -> int:
    """Значення для --limit має бути додатним цілим числом."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--limit має бути цілим числом") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--limit має бути більшим за нуль")
    return value


def main() -> None:
    """Команда для підрахунку статистики, часу та пікової пам'яті."""
    parser = argparse.ArgumentParser(description="Статистика корпусу findex")
    parser.add_argument("corpus", type=Path, help="Каталог із файлами .txt")
    parser.add_argument(
        "--limit", type=positive_int, default=None, metavar="N",
        help="Обробити не більше ніж N документів",
    )
    args = parser.parse_args()

    if not args.corpus.is_dir():
        parser.error(f"Каталог не знайдено: {args.corpus}")

    documents = iter_documents(args.corpus)
    if args.limit is not None:
        documents = islice(documents, args.limit)

    tracemalloc.start()
    start = time.perf_counter()
    try:
        stats = collect_stats(documents)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Документів: {stats.documents}")
    print(f"Токенів: {stats.tokens}")
    print(f"Унікальних термінів: {stats.vocabulary}")
    print("Топ-50 термінів:")
    for term, count in stats.top_terms:
        print(f"  {term}: {count}")
    print(f"Час: {elapsed:.4f} с")
    print(f"Пікова пам'ять (tracemalloc): {peak / (1024 * 1024):.3f} MiB")


if __name__ == "__main__":
    main()
