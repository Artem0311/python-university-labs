
"""Boolean and ranked search over an inverted index."""

import logging
from dataclasses import dataclass, field, replace
from functools import lru_cache
from heapq import nlargest
from pathlib import Path
from typing import Literal

from findex.index import Index
from findex.query import And, Not, Or, Phrase, Query, Term, parse
from findex.ranking import BM25, Scorer
from findex.snippet import make_snippet
from findex.timing import timed
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class SearchResult:
    """A ranked search result ordered by relevance score."""

    score: float
    doc_id: int = field(compare=False)
    path: str = field(compare=False)
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)


def _merge_and(left: list[int], right: list[int]) -> list[int]:
    """Return the intersection of sorted document IDs."""
    result: list[int] = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1

    return result


def _merge_or(left: list[int], right: list[int]) -> list[int]:
    """Return the union of sorted document IDs."""
    result: list[int] = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def _merge_not(left: list[int], right: list[int]) -> list[int]:
    """Return sorted IDs from left that are absent from right."""
    result: list[int] = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            j += 1

    result.extend(left[i:])
    return result


def search_index(
    index: Index,
    query: str,
    engine: Literal["merge", "set"] = "merge",
) -> list[int]:
    """Legacy Boolean search from Lab 02."""
    if engine not in {"merge", "set"}:
        raise ValueError("engine must be 'merge' or 'set'")

    tokens: list[str] = []

    for item in query.split():
        if item.upper() in {"AND", "OR", "NOT"}:
            tokens.append(item.upper())
        else:
            tokens.extend(tokenize(item))

    if not tokens:
        return []

    all_docs = sorted(index.doc_meta)

    def operand(pos: int) -> tuple[list[int], int]:
        negate = False

        while pos < len(tokens) and tokens[pos] == "NOT":
            negate = not negate
            pos += 1

        if pos == len(tokens) or tokens[pos] in {"AND", "OR"}:
            raise ValueError("Expected a search term after an operator")

        term = tokens[pos]
        matches = [
            posting.doc_id
            for posting in index.postings.get(term, [])
        ]

        if negate:
            if engine == "merge":
                matches = _merge_not(all_docs, matches)
            else:
                matches = sorted(set(all_docs) - set(matches))

        return matches, pos + 1

    result, pos = operand(0)

    while pos < len(tokens):
        if tokens[pos] in {"AND", "OR"}:
            operator = tokens[pos]
            pos += 1
        else:
            operator = "AND"

        right, pos = operand(pos)

        if engine == "merge":
            result = (
                _merge_and(result, right)
                if operator == "AND"
                else _merge_or(result, right)
            )
        else:
            result_set = set(result)
            right_set = set(right)
            result = sorted(
                result_set & right_set
                if operator == "AND"
                else result_set | right_set
            )

    return result


def _cached_matching_ids(index: Index, query: str) -> set[int]:
    """Cache query-to-document matching separately for each index."""
    cached = index.__dict__.get("_cached_query_ids")

    if cached is None:
        @lru_cache(maxsize=128)
        def evaluate(text: str) -> tuple[int, ...]:
            return tuple(sorted(parse(text).evaluate(index)))

        cached = evaluate
        index.__dict__["_cached_query_ids"] = cached

    previous_hits = cached.cache_info().hits
    result = cached(query)

    status = (
        "hit"
        if cached.cache_info().hits > previous_hits
        else "miss"
    )

    logger.debug("[cache] %s: %r", status, query)

    return set(result)


def _positive_terms(
    node: Query,
    negated: bool = False,
) -> list[str]:
    """Extract terms that should contribute to ranking."""

    if isinstance(node, Not):
        return _positive_terms(node.expression, not negated)

    if isinstance(node, Term):
        return [] if negated else [node.value]

    if isinstance(node, Phrase):
        return [] if negated else list(tokenize(node.text))

    if isinstance(node, (And, Or)):
        return (
            _positive_terms(node.left, negated)
            + _positive_terms(node.right, negated)
        )

    return []


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 5,
    corpus_root: str | Path = "data",
) -> list[SearchResult]:
    """Evaluate a query tree and rank matching documents."""
    if k <= 0 or not query.strip():
        return []

    if scorer is None:
        scorer = BM25()

    tree = parse(query)
    matching_ids = _cached_matching_ids(index, query)

    if not matching_ids:
        return []

    scores = {
        doc_id: 0.0
        for doc_id in sorted(matching_ids)
    }

    terms = dict.fromkeys(_positive_terms(tree))

    for term in terms:
        for posting in index.postings.get(term, []):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer.score(
                    term,
                    posting,
                    index,
                )

    results = [
        SearchResult(
            score=score,
            doc_id=doc_id,
            path=index.doc_meta[doc_id].path,
            title=index.doc_meta[doc_id].title,
        )
        for doc_id, score in scores.items()
    ]

    top_results = nlargest(k, results)
    terms = tuple(dict.fromkeys(_positive_terms(tree)))
    enriched: list[SearchResult] = []

    for result in top_results:
        document_path = Path(corpus_root) / result.path

        try:
            text = document_path.read_text(encoding="utf-8")
            snippet = make_snippet(text, terms)
        except (OSError, UnicodeError):
            snippet = ""

        enriched.append(replace(result, snippet=snippet))

    return enriched


def main() -> None:
    """Search a saved index from the command line."""
    import argparse
    import tracemalloc
    from time import perf_counter

    from findex.store import load_index

    parser = argparse.ArgumentParser(
        description="Search documents using a saved index"
    )
    parser.add_argument("index_path", type=Path)
    parser.add_argument("query")
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
    )
    args = parser.parse_args()

    tracemalloc.start()
    started = perf_counter()

    try:
        index = load_index(args.index_path)
        results = search_index(index, args.query, args.engine)
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Found {len(results)} documents:")

    for doc_id in results:
        meta = index.doc_meta[doc_id]
        print(f"  {doc_id}: {meta.path} ({meta.title})")

    print(f"Engine: {args.engine}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Peak memory: {peak:,} bytes")


if __name__ == "__main__":
    main()
