
"""Benchmark serial, threaded and process-based indexing."""

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from typing import TypedDict, cast

from findex.index import BuildMetrics, ExecutorKind, index_corpus

EXECUTORS: tuple[ExecutorKind, ...] = (
    "serial",
    "threads",
    "processes",
)


class Measurement(TypedDict):
    """Results of one indexing benchmark."""

    executor: ExecutorKind
    workers: int
    documents: int
    terms: int
    wall_seconds: float
    cpu_seconds: float
    rss_mib: float
    merge_seconds: float


def measure_once(
    corpus: Path,
    executor: ExecutorKind,
    workers: int,
) -> Measurement:
    """Measure indexing in an isolated Python process."""
    metrics = BuildMetrics()

    children_before = resource.getrusage(
        resource.RUSAGE_CHILDREN
    )

    cpu_before = process_time()
    wall_before = perf_counter()

    index = index_corpus(
        corpus,
        workers=workers,
        executor=executor,
        with_positions=True,
        metrics=metrics,
    )

    wall_seconds = perf_counter() - wall_before
    parent_cpu = process_time() - cpu_before

    children_after = resource.getrusage(
        resource.RUSAGE_CHILDREN
    )

    children_cpu = (
        children_after.ru_utime
        + children_after.ru_stime
        - children_before.ru_utime
        - children_before.ru_stime
    )

    parent_usage = resource.getrusage(
        resource.RUSAGE_SELF
    )

    # On macOS, ru_maxrss is measured in bytes.
    rss_mib = parent_usage.ru_maxrss / (1024 * 1024)

    return {
        "executor": executor,
        "workers": workers,
        "documents": index.num_docs,
        "terms": len(index),
        "wall_seconds": wall_seconds,
        "cpu_seconds": parent_cpu + children_cpu,
        "rss_mib": rss_mib,
        "merge_seconds": metrics.merge_seconds,
    }


def run_isolated(
    corpus: Path,
    executor: ExecutorKind,
    workers: int,
) -> Measurement:
    """Run a benchmark in a fresh Python interpreter."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(corpus),
        "--executor",
        executor,
        "--workers",
        str(workers),
        "--once",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Benchmark failed: {executor}, "
            f"workers={workers}"
        )

    return cast(
        Measurement,
        json.loads(completed.stdout),
    )


def main() -> None:
    """Run benchmarks and report median measurements."""
    parser = argparse.ArgumentParser(
        description="Benchmark findex concurrency."
    )

    parser.add_argument("corpus", type=Path)

    parser.add_argument(
        "--executor",
        choices=EXECUTORS,
        default="serial",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/findex-lab05-results.csv"),
    )

    parser.add_argument("--once", action="store_true")

    args = parser.parse_args()

    corpus = cast(Path, args.corpus)
    executor = cast(ExecutorKind, args.executor)
    workers = cast(int, args.workers)
    repeats = cast(int, args.repeats)
    output_path = cast(Path, args.out)
    once = cast(bool, args.once)

    if not corpus.is_dir():
        parser.error("Corpus directory does not exist")

    if workers < 1:
        parser.error("Workers must be positive")

    if once:
        result = measure_once(
            corpus,
            executor,
            workers,
        )
        print(json.dumps(result))
        return

    if repeats < 3:
        parser.error("At least 3 repetitions are required")

    counts: list[int] = list(
        dict.fromkeys(
            [1, 2, 4, 8, os.cpu_count() or 1]
        )
    )

    rows: list[tuple[Measurement, int]] = []

    for mode in EXECUTORS:
        for count in counts:
            for repeat in range(1, repeats + 1):
                print(
                    f"{mode}, workers={count}, "
                    f"run={repeat}/{repeats}",
                    flush=True,
                )

                result = run_isolated(
                    corpus,
                    mode,
                    count,
                )

                rows.append((result, repeat))

    fields = [
        "executor",
        "workers",
        "repeat",
        "documents",
        "terms",
        "wall_seconds",
        "cpu_seconds",
        "rss_mib",
        "merge_seconds",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
        )

        writer.writeheader()

        for result, repeat in rows:
            writer.writerow(
                {**result, "repeat": repeat}
            )

    baseline = median(
        result["wall_seconds"]
        for result, _ in rows
        if result["executor"] == "serial"
        and result["workers"] == 1
    )

    print()
    print(
        "Executor   Workers  Wall(s)  CPU(s)"
        "  RSS(MiB)  Merge(s)  Speedup"
    )

    for mode in EXECUTORS:
        for count in counts:
            group = [
                result
                for result, _ in rows
                if result["executor"] == mode
                and result["workers"] == count
            ]

            wall = median(
                r["wall_seconds"] for r in group
            )

            cpu = median(
                r["cpu_seconds"] for r in group
            )

            rss = median(
                r["rss_mib"] for r in group
            )

            merge = median(
                r["merge_seconds"] for r in group
            )

            speedup = baseline / wall

            print(
                f"{mode:<11}"
                f"{count:>7}"
                f"{wall:>9.3f}"
                f"{cpu:>8.3f}"
                f"{rss:>10.1f}"
                f"{merge:>10.3f}"
                f"{speedup:>9.2f}x"
            )

    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
