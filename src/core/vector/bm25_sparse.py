"""Sparse encoder based on BM25 statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re


TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class Bm25SparseEncoder:
    """Builds sparse vectors for documents and queries."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_to_index: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avg_doc_len = 1.0
        self.fitted = False

    def fit(self, documents: list[str]) -> None:
        document_count = len(documents)
        if document_count == 0:
            self.fitted = True
            return

        doc_lengths: list[int] = []
        doc_frequency: Counter[str] = Counter()
        for text in documents:
            tokens = _tokenize(text)
            doc_lengths.append(len(tokens))
            doc_frequency.update(set(tokens))

        self.avg_doc_len = max(1.0, sum(doc_lengths) / len(doc_lengths))
        ordered_terms = sorted(doc_frequency.keys())
        self.term_to_index = {term: idx for idx, term in enumerate(ordered_terms)}
        self.idf = {
            term: math.log(1 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_frequency.items()
        }
        self.fitted = True

    def encode_document(self, text: str) -> SparseVector:
        if not self.fitted:
            raise RuntimeError("BM25 encoder is not fitted.")
        tokens = _tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        tf = Counter(tokens)
        doc_len = len(tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, frequency in tf.items():
            if term not in self.term_to_index or term not in self.idf:
                continue
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score = self.idf[term] * (numerator / denominator)
            indices.append(self.term_to_index[term])
            values.append(float(score))
        return SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVector:
        if not self.fitted:
            raise RuntimeError("BM25 encoder is not fitted.")
        query_terms = Counter(_tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for term, frequency in query_terms.items():
            if term not in self.term_to_index or term not in self.idf:
                continue
            indices.append(self.term_to_index[term])
            values.append(float(self.idf[term] * frequency))
        return SparseVector(indices=indices, values=values)
