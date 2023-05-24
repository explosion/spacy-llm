from typing import Iterable, Tuple


def _unique(items: Iterable[str]) -> Iterable[str]:
    """Remove duplicates without changing order"""
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def find_substrings(
    text: str,
    substrings: Iterable[str],
    *,
    case_sensitive: bool = False,
    single_match: bool = False,
) -> Iterable[Tuple[int, int]]:
    """Given a list of substrings, find their character start and end positions
    in a text"""

    # Remove empty and duplicate strings, and lowercase everything if need be
    substrings = [s for s in substrings if s and len(s) > 0]
    if not case_sensitive:
        text = text.lower()
        substrings = [s.lower() for s in substrings]
    substrings = _unique(substrings)
    offsets = []
    for substring in substrings:
        search_from = 0
        # Search until one hit is found. Continue only if single_match is False.
        while True:
            start = text.find(substring, search_from)
            if start == -1:
                break
            end = start + len(substring)
            offsets.append((start, end))
            if single_match:
                break
            search_from = end
    return offsets
