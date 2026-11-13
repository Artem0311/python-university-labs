
"""Command-line interface for the findex search engine."""

import json
import logging
import sys
from collections import Counter
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
from typing import Annotated, Never

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from findex.index import ExecutorKind, index_corpus
from findex.ranking import BM25, TfIdf
from findex.search import search as ranked_search
from findex.store import load_index, save_index
from findex.tokenize import tokenize

app = typer.Typer(
    help="Build and search local text indexes.",
    add_completion=False,
)
output = Console()
diagnostics = Console(stderr=True)
logger = logging.getLogger(__name__)


class ScorerName(str, Enum):
    """Available relevance scoring algorithms."""

    bm25 = "bm25"
    tfidf = "tfidf"


class ExecutorName(str, Enum):
    """Available indexing execution strategies."""

    serial = "serial"
    threads = "threads"
    processes = "processes"


def _fail(message: str) -> Never:
    """Show an expected error without a Python traceback."""
    typer.echo(f"Error: {message.splitlines()[0]}", err=True)
    raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase log detail.",
        ),
    ] = 0,
) -> None:
    """Configure logging once, at the CLI entry point."""
    level = (
        logging.DEBUG
        if verbose >= 2
        else logging.INFO
        if verbose
        else logging.WARNING
    )

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


@app.command("index")
def index_command(
    corpus: Annotated[
        Path,
        typer.Argument(help="Directory of UTF-8 .txt files."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Path to the saved index."),
    ],
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Enable phrase search."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            min=1,
            help="Number of indexing workers.",
        ),
    ] = 1,
    executor: Annotated[
        ExecutorName,
        typer.Option(
            "--executor",
            help="Indexing execution strategy.",
        ),
    ] = ExecutorName.processes,
) -> None:
    """Build an inverted index with the selected executor."""
    if not corpus.is_dir():
        _fail(f"Corpus directory not found: {corpus}")

    executor_kind: ExecutorKind = executor.value

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=diagnostics,
        transient=True,
    ) as progress:
        progress.add_task("Indexing documents", total=None)

        # Keep timing diagnostics out of standard output.
        with redirect_stdout(sys.stderr):
            result = index_corpus(
                corpus,
                with_positions=positions,
                workers=workers,
                executor=executor_kind,
                limit=limit,
            )

            save_index(result, out)

    logger.info(
        "Indexed %d documents using %s with %d workers",
        result.num_docs,
        executor.value,
        workers,
    )

    output.print(
        f"Indexed {result.num_docs} documents; "
        f"{len(result)} terms."
    )
    output.print(f"Executor: {executor.value}")
    output.print(f"Workers: {workers}")
    output.print(f"Saved index: {out}")


@app.command("search")
def search_command(
    index_path: Annotated[
        Path,
        typer.Argument(help="Saved JSON or binary index."),
    ],
    query: Annotated[
        str,
        typer.Argument(help="Boolean or phrase query."),
    ],
    k: Annotated[
        int,
        typer.Option("--k", min=1),
    ] = 10,
    scorer: Annotated[
        ScorerName,
        typer.Option("--scorer"),
    ] = ScorerName.bm25,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write JSON lines to stdout.",
        ),
    ] = False,
    corpus_root: Annotated[
        Path,
        typer.Option(
            "--corpus-root",
            help="Original corpus directory for snippets.",
        ),
    ] = Path("data"),
) -> None:
    """Search an index and show ranked results."""
    if not query.strip():
        _fail("Empty search query")

    try:
        with redirect_stdout(sys.stderr):
            index = load_index(index_path)
            algorithm = (
                BM25()
                if scorer == ScorerName.bm25
                else TfIdf()
            )
            results = ranked_search(
                index,
                query,
                scorer=algorithm,
                k=k,
                corpus_root=corpus_root,
            )
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        KeyError,
    ) as exc:
        _fail(str(exc))

    if as_json:
        for result in results:
            typer.echo(
                json.dumps(
                    {
                        "doc_id": result.doc_id,
                        "path": result.path,
                        "title": result.title,
                        "score": result.score,
                        "snippet": result.snippet,
                    },
                    ensure_ascii=False,
                )
            )
        return

    table = Table(title="Search results")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Document")
    table.add_column("Snippet")

    terms = [
        word
        for word in tokenize(query)
        if word.upper() not in {"AND", "OR", "NOT"}
    ]

    for position, result in enumerate(results, start=1):
        snippet = Text(result.snippet)
        snippet.highlight_words(
            terms,
            style="bold yellow",
            case_sensitive=False,
        )
        table.add_row(
            str(position),
            f"{result.score:.3f}",
            result.path,
            snippet,
        )

    output.print(table)

    if not results:
        output.print("No matching documents.")


@app.command("stats")
def stats_command(
    index_path: Annotated[
        Path,
        typer.Argument(help="Saved JSON or binary index."),
    ],
) -> None:
    """Display corpus statistics stored in an index."""
    try:
        with redirect_stdout(sys.stderr):
            index = load_index(index_path)
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        KeyError,
    ) as exc:
        _fail(str(exc))

    summary = Table(title="Index statistics")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Documents", str(index.num_docs))
    summary.add_row(
        "Tokens",
        str(sum(index.doc_lengths.values())),
    )
    summary.add_row("Unique terms", str(len(index)))
    output.print(summary)

    counts = Counter(
        {
            term: sum(posting.tf for posting in postings)
            for term, postings in index.postings.items()
        }
    )

    popular = Table(title="Top 50 terms")
    popular.add_column("Term")
    popular.add_column("Occurrences", justify="right")

    for term, count in counts.most_common(50):
        popular.add_row(term, str(count))

    output.print(popular)
