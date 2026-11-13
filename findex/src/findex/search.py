"""Boolean search over the sorted posting lists of the inverted index."""

from findex.index import Index
from findex.tokenize import tokenize


def _merge_and(left: list[int], right: list[int]) -> list[int]:
    """Return the intersection of sorted document IDs using two pointers."""
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
    """Return the union of sorted document IDs without duplicates."""
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


def search_index(index: Index, query: str, engine: str = "merge") -> list[int]:
    """Evaluate a Boolean query, returning sorted document IDs.

    Adjacent terms imply AND; explicit AND, OR and unary NOT are supported.
    Operators are evaluated from left to right, without parentheses.
    For example: ``python programming``, ``python OR java``, ``python NOT java``.
    """
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
        matches = [posting.doc_id for posting in index.postings.get(term, [])]
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
            result = _merge_and(result, right) if operator == "AND" else _merge_or(result, right)
        else:
            result_set = set(result)
            right_set = set(right)
            result = sorted(result_set & right_set if operator == "AND" else result_set | right_set)

    return result

def main() -> None:
    """Search a saved index from the command line."""
    import argparse
    import tracemalloc
    from pathlib import Path
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