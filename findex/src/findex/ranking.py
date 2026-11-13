
"""TF-IDF and BM25 ranking algorithms."""

from dataclasses import dataclass
from math import log, sqrt
from typing import Protocol

from findex.index import Index, Posting


class Scorer(Protocol):
    """Interface shared by all ranking algorithms."""

    def score(
        self,
        term: str,
        posting: Posting,
        index: Index,
    ) -> float:
        ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """TF-IDF scoring with logarithmic TF and length normalization."""

    def score(
        self,
        term: str,
        posting: Posting,
        index: Index,
    ) -> float:
        if posting.tf <= 0 or index.num_docs == 0:
            return 0.0

        tf = 1.0 + log(posting.tf)
        idf = log(
            (index.num_docs + 1) / (index.df(term) + 1)
        ) + 1.0

        doc_length = max(1, index.doc_lengths[posting.doc_id])

        return tf * idf / sqrt(doc_length)


@dataclass(frozen=True, slots=True)
class BM25:
    """BM25 ranking with term saturation and length normalization."""

    k1: float = 1.5
    b: float = 0.75

    def score(
        self,
        term: str,
        posting: Posting,
        index: Index,
    ) -> float:
        if posting.tf <= 0 or index.num_docs == 0:
            return 0.0

        df = index.df(term)
        idf = log(
            1.0
            + (index.num_docs - df + 0.5) / (df + 0.5)
        )

        avg_length = max(index.avg_doc_length, 1.0)
        doc_length = index.doc_lengths[posting.doc_id]

        normalization = self.k1 * (
            1.0 - self.b
            + self.b * doc_length / avg_length
        )

        tf_weight = (
            posting.tf * (self.k1 + 1.0)
            / (posting.tf + normalization)
        )

        return idf * tf_weight
