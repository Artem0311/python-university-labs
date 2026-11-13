
"""Generate highlighted snippets for search results."""

import re
from collections.abc import Iterable


def make_snippet(
    text: str,
    terms: Iterable[str],
    window: int = 80,
) -> str:
    """Return a highlighted snippet around the best term match."""

    if window < 0:
        raise ValueError("Window must be non-negative")

    if not text:
        return ""

    words = sorted(
        {term for term in terms if term},
        key=len,
        reverse=True,
    )

    if not words:
        return text[:160] + ("..." if len(text) > 160 else "")

    pattern = re.compile(
        r"(?<!\w)(?:" +
        "|".join(re.escape(word) for word in words) +
        r")(?!\w)",
        re.IGNORECASE,
    )

    matches = list(pattern.finditer(text))

    if not matches:
        return text[:160] + ("..." if len(text) > 160 else "")

    def match_score(match: re.Match[str]) -> tuple[int, int]:
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)

        nearby = [
            other.group().casefold()
            for other in matches
            if other.start() >= start and other.end() <= end
        ]

        return len(set(nearby)), len(nearby)

    best = max(matches, key=match_score)

    start = max(0, best.start() - window)
    end = min(len(text), best.end() + window)

    snippet = text[start:end].replace("\n", " ")

    highlighted = pattern.sub(
        lambda match: f"**{match.group()}**",
        snippet,
    )

    if start > 0:
        highlighted = "..." + highlighted

    if end < len(text):
        highlighted += "..."

    return highlighted
