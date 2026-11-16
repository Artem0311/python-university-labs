"""Tests for the findex HTTP API."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web import Settings, app, get_index, get_settings


@pytest.fixture
def client(
    sample_index: Index,
    sample_corpus: Path,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_index] = lambda: sample_index
    app.dependency_overrides[get_settings] = lambda: Settings(
        index_path=sample_corpus / "missing-index.bin",
        corpus_root=sample_corpus,
    )
    app.state.index = None

    test_client = TestClient(app)
    yield test_client

    test_client.close()
    app.dependency_overrides.clear()
    app.state.index = None


def test_search_and_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "python", "k": 1})
    second = client.get(
        "/search", params={"q": "python", "k": 1, "page": 2}
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["total"] == 2
    assert len(first.json()["results"]) == 1
    assert (
        first.json()["results"][0]["doc_id"]
        != second.json()["results"][0]["doc_id"]
    )


def test_invalid_search(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python", "k": 0})
    assert response.status_code == 422


def test_document(client: TestClient, sample_index: Index) -> None:
    doc_id = next(iter(sample_index.doc_meta))
    response = client.get(f"/docs/{doc_id}")

    assert response.status_code == 200
    assert response.json()["doc_id"] == doc_id
    assert response.json()["text"]


def test_unknown_document(client: TestClient) -> None:
    response = client.get("/docs/999999")
    assert response.status_code == 404


def test_stats(client: TestClient) -> None:
    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json()["documents"] == 3


def test_health_without_index(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 503


def test_openapi_docs(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
