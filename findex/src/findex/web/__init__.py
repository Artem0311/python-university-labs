"""HTTP API for the findex search engine."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.responses import Response

from findex.index import Index
from findex.ranking import BM25, TfIdf
from findex.search import search as ranked_search
from findex.store import load_index

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("index.bin")
    corpus_root: Path = Path("data")
    log_level: str = "INFO"
    port: int = 8000


def get_settings() -> Settings:
    return Settings()


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=200)
    k: int = Field(default=10, ge=1, le=100)
    scorer: Literal["bm25", "tfidf"] = "bm25"
    page: int = Field(default=1, ge=1)

    @field_validator("q")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be empty")
        return value


class SearchResult(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    results: list[SearchResult]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    text: str


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    application.state.index = None
    application.state.started_at = perf_counter()

    try:
        application.state.index = load_index(settings.index_path)
    except (OSError, ValueError, TypeError, EOFError, KeyError):
        logger.exception("Unable to load index: %s", settings.index_path)

    yield
    application.state.index = None


app = FastAPI(title="findex", lifespan=lifespan)
app.state.index = None
app.state.started_at = perf_counter()

WEB_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


def get_index(request: Request) -> Index:
    index = cast(Index | None, request.app.state.index)
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return index


@app.middleware("http")
async def handle_errors(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = uuid4().hex
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Request failed: id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    params: Annotated[SearchRequest, Depends()],
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchResponse:
    started = perf_counter()
    scorer = BM25() if params.scorer == "bm25" else TfIdf()

    try:
        matches = ranked_search(
            index,
            params.q,
            scorer=scorer,
            k=index.num_docs,
            corpus_root=settings.corpus_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    start = (params.page - 1) * params.k
    selected = matches[start : start + params.k]

    return SearchResponse(
        query=params.q,
        total=len(matches),
        took_ms=round((perf_counter() - started) * 1000, 3),
        results=[
            SearchResult(
                doc_id=item.doc_id,
                title=item.title,
                score=item.score,
                snippet=item.snippet,
            )
            for item in selected
        ],
    )


@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def document_endpoint(
    doc_id: int,
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentResponse:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        text = (settings.corpus_root / meta.path).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError):
        raise HTTPException(
            status_code=404, detail="Document unavailable"
        ) from None

    return DocumentResponse(
        doc_id=doc_id,
        title=meta.title,
        text=text,
    )


@app.get("/stats")
def stats_endpoint(
    request: Request,
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, int | float]:
    try:
        size = settings.index_path.stat().st_size
    except OSError:
        size = 0

    return {
        "index_size_bytes": size,
        "documents": index.num_docs,
        "tokens": sum(index.doc_lengths.values()),
        "vocabulary": len(index),
        "uptime_seconds": round(
            perf_counter() - cast(float, request.app.state.started_at), 3
        ),
    }


@app.get("/health")
def health_endpoint(request: Request) -> dict[str, str]:
    if request.app.state.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {"status": "ok"}



def highlight_snippet(snippet: str) -> Markup:
    """Render the search engine's bold markers as safe HTML."""
    parts: list[Markup] = []
    for position, part in enumerate(snippet.split("**")):
        escaped = escape(part)
        if position % 2:
            parts.append(Markup("<mark>") + escaped + Markup("</mark>"))
        else:
            parts.append(escaped)
    return Markup("").join(parts)


@app.get("/", response_class=HTMLResponse)
def search_page(
    request: Request,
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    q: Annotated[str, Query(max_length=200)] = "",
    page: Annotated[int, Query(ge=1)] = 1,
) -> Response:
    """Render the search form and paginated results."""
    matches = (
        ranked_search(
            index,
            q,
            scorer=BM25(),
            k=index.num_docs,
            corpus_root=settings.corpus_root,
        )
        if q.strip() and index.num_docs
        else []
    )

    per_page = 10
    total = len(matches)
    selected = matches[(page - 1) * per_page : page * per_page]

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "q": q,
            "page": page,
            "total": total,
            "total_pages": max(1, (total + per_page - 1) // per_page),
            "results": [
                {
                    "doc_id": item.doc_id,
                    "title": item.title,
                    "score": item.score,
                    "snippet": highlight_snippet(item.snippet),
                }
                for item in selected
            ],
            "document": None,
        },
    )


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def document_page(
    doc_id: int,
    request: Request,
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Display the full text of an indexed document."""
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        text = (settings.corpus_root / meta.path).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError):
        raise HTTPException(
            status_code=404, detail="Document unavailable"
        ) from None

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "document": {"title": meta.title, "text": text},
        },
    )
