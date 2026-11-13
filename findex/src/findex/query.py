
"""Recursive-descent parser for Boolean and phrase queries."""

import re
from dataclasses import dataclass

from findex.index import Index
from findex.tokenize import tokenize


class Query:
    """Base class for composable search queries."""

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def __and__(self, other: Query) -> And:
        return And(self, other)

    def __or__(self, other: Query) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True)
class Term(Query):
    """Match documents containing a single term."""

    value: str

    def evaluate(self, index: Index) -> set[int]:
        return {
            posting.doc_id
            for posting in index.postings.get(self.value, [])
        }


@dataclass(frozen=True)
class Phrase(Query):
    """Match consecutive terms using their token positions."""

    text: str

    def evaluate(self, index: Index) -> set[int]:
        words = list(tokenize(self.text))

        if not words:
            return set()

        if len(words) == 1:
            return Term(words[0]).evaluate(index)

        if not index.positions:
            raise ValueError(
                "Phrase search requires an index with positions"
            )

        candidate_docs = Term(words[0]).evaluate(index)

        for word in words[1:]:
            candidate_docs &= Term(word).evaluate(index)

        result: set[int] = set()

        for doc_id in candidate_docs:
            starts = set(
                index.positions.get(words[0], {}).get(doc_id, ())
            )

            for offset, word in enumerate(words[1:], start=1):
                positions = index.positions.get(word, {}).get(
                    doc_id, ()
                )

                starts &= {
                    position - offset
                    for position in positions
                }

                if not starts:
                    break

            if starts:
                result.add(doc_id)

        return result


@dataclass(frozen=True)
class And(Query):
    """Match documents satisfying both expressions."""

    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return (
            self.left.evaluate(index)
            & self.right.evaluate(index)
        )


@dataclass(frozen=True)
class Or(Query):
    """Match documents satisfying either expression."""

    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return (
            self.left.evaluate(index)
            | self.right.evaluate(index)
        )


@dataclass(frozen=True)
class Not(Query):
    """Match documents not satisfying the expression."""

    expression: Query

    def evaluate(self, index: Index) -> set[int]:
        return (
            set(index.doc_meta)
            - self.expression.evaluate(index)
        )


def _lex(query: str) -> list[str]:
    """Split query into operators, terms, parentheses and phrases."""
    pattern = r'"[^"]*"|[()]|[^\s()"]+'
    tokens: list[str] = []
    end = 0

    for match in re.finditer(pattern, query):
        if query[end:match.start()].strip():
            raise ValueError("Invalid query syntax")

        tokens.append(match.group())
        end = match.end()

    if query[end:].strip():
        raise ValueError("Invalid query syntax")

    return tokens


class _Parser:
    """Recursive-descent parser with Boolean precedence."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def take(self) -> str:
        token = self.peek()

        if token is None:
            raise ValueError("Unexpected end of query")

        self.pos += 1
        return token

    def parse_or(self) -> Query:
        node = self.parse_and()

        while self.peek() is not None:
            if self.peek().upper() != "OR":
                break

            self.take()
            node = Or(node, self.parse_and())

        return node

    def parse_and(self) -> Query:
        node = self.parse_not()

        while True:
            token = self.peek()

            if token is None:
                break

            if token == ")" or token.upper() == "OR":
                break

            if token.upper() == "AND":
                self.take()

            node = And(node, self.parse_not())

        return node

    def parse_not(self) -> Query:
        token = self.peek()

        if token is not None and token.upper() == "NOT":
            self.take()
            return Not(self.parse_not())

        return self.parse_atom()

    def parse_atom(self) -> Query:
        token = self.take()

        if token == "(":
            node = self.parse_or()

            if self.take() != ")":
                raise ValueError("Expected closing parenthesis")

            return node

        if token == ")":
            raise ValueError("Unexpected closing parenthesis")

        if token.upper() in {"AND", "OR", "NOT"}:
            raise ValueError(
                f"Unexpected operator: {token}"
            )

        if token.startswith('"'):
            text = token[1:-1]

            if not list(tokenize(text)):
                raise ValueError("Empty phrase")

            return Phrase(text)

        words = list(tokenize(token))

        if not words:
            raise ValueError("Expected a search term")

        node: Query = Term(words[0])

        for word in words[1:]:
            node = And(node, Term(word))

        return node


def parse(query: str) -> Query:
    """Parse a query into an expression tree.

    Precedence:
        NOT > AND (including implicit AND) > OR
    """
    tokens = _lex(query)

    if not tokens:
        raise ValueError("Empty query")

    parser = _Parser(tokens)
    result = parser.parse_or()

    if parser.peek() is not None:
        raise ValueError(
            f"Unexpected token: {parser.peek()}"
        )

    return result
