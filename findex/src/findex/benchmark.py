"""Порівняння жадібної та лінивої обробки корпусу (M4).

Запуск: uv run python -m findex.benchmark data/ --limit 100
"""

import argparse
import gc
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.stats import CorpusStats, collect_stats, positive_int
from findex.tokenize import tokenize


def eager_stats(root: Path, limit: int | None) -> CorpusStats:
    """Спершу зберегти всі документи й усі списки токенів у пам'яті."""
    documents = list(islice(iter_documents(root), limit))
    token_lists = [list(tokenize(document.text)) for document in documents]
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return CorpusStats(
        documents=len(documents),
        tokens=sum(len(tokens) for tokens in token_lists),
        vocabulary=len(counts),
        top_terms=counts.most_common(50),
    )


def lazy_stats(root: Path, limit: int | None) -> CorpusStats:
    """Обробити документи послідовно без списків документів і токенів."""
    return collect_stats(islice(iter_documents(root), limit))


def measure(action: Callable[[], CorpusStats]) -> tuple[CorpusStats, float, float]:
    """Виміряти час і пік Python-пам'яті окремим запуском tracemalloc."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = action()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Порівняння пам'яті: списки та генератори")
    parser.add_argument("corpus", type=Path, help="Каталог із .txt-документами")
    parser.add_argument("--limit", type=positive_int, default=None, metavar="N")
    args = parser.parse_args()
    if not args.corpus.is_dir():
        parser.error(f"Каталог не знайдено: {args.corpus}")

    eager, eager_time, eager_peak = measure(lambda: eager_stats(args.corpus, args.limit))
    lazy, lazy_time, lazy_peak = measure(lambda: lazy_stats(args.corpus, args.limit))
    if eager != lazy:
        parser.error("Результати eager та lazy відрізняються; порівняння некоректне")

    print("Версія             Документів    Пікова пам'ять     Час")
    print(f"eager (списки)     {eager.documents:>10}    {eager_peak:>10.3f} MiB    {eager_time:.4f} с")
    print(f"lazy (генератори)  {lazy.documents:>10}    {lazy_peak:>10.3f} MiB    {lazy_time:.4f} с")
    print(f"Токенів: {lazy.tokens}; унікальних термінів: {lazy.vocabulary}")


if __name__ == "__main__":
    main()
