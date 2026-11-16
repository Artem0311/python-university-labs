
"""TF-IDF and BM25 ranking algorithms."""

from dataclasses import dataclass
from math import log, sqrt
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

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

    def score_array(
        self,
        term: str,
        doc_ids: NDArray[np.int32],
        tfs: NDArray[np.int32],
        index: Index,
    ) -> NDArray[np.float64]:
        idf = log((index.num_docs + 1) / (index.df(term) + 1)) + 1.0
        lengths = np.maximum(index.numpy_lengths[doc_ids], 1)
        values = (1.0 + np.log(np.maximum(tfs, 1))) * idf / np.sqrt(lengths)
        return np.asarray(np.where(tfs > 0, values, 0.0), dtype=np.float64)


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

    def score_array(
        self,
        term: str,
        doc_ids: NDArray[np.int32],
        tfs: NDArray[np.int32],
        index: Index,
    ) -> NDArray[np.float64]:
        df = index.df(term)
        idf = log(1.0 + (index.num_docs - df + 0.5) / (df + 0.5))
        avg_length = max(index.avg_doc_length, 1.0)
        lengths = index.numpy_lengths[doc_ids]
        normalization = self.k1 * (
            1.0 - self.b + self.b * lengths / avg_length
        )
        values = idf * (
            tfs * (self.k1 + 1.0) / (tfs + normalization)
        )
        return np.asarray(values, dtype=np.float64)
