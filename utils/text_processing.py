import re
from typing import Iterable

NIQQUD_RANGE = r"\u0591-\u05C7"
NIQQUD_RE = re.compile(f"[{NIQQUD_RANGE}]")
_WHITESPACE_RE = re.compile(r"\s+")
PREFIX_CHARS = "ובכלדהשה"
SEP_PATTERN = r"[\s\u00A0,.;:׳\"״\-–—·<>/]*"
HEBREW_RANGE = r"\u0590-\u05FF"


def strip_niqqud(text: str) -> str:
    """Remove Hebrew niqqud (vowel/taamim marks) from the provided text."""
    if not text:
        return ""
    return NIQQUD_RE.sub("", text)


def _prepare_terms(terms: Iterable[str]) -> list[str]:
    prepared: list[str] = []
    for term in terms:
        if not term:
            continue
        clean = strip_niqqud(term)
        clean = _WHITESPACE_RE.sub(" ", clean).strip()
        if clean:
            prepared.append(clean)
    # Preserve order but drop duplicates
    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in prepared:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)
    return unique_terms


def generate_vowel_insensitive_regex(name_he: str) -> str:
    """
    Build a regex string that matches the given Hebrew term regardless of niqqud.
    """
    terms = _prepare_terms([name_he])
    if not terms:
        return ""

    base = terms[0]
    prefix = f"(?:[{PREFIX_CHARS}][{NIQQUD_RANGE}]*['\"׳״]?\\s*)?"
    # allow niqqud, whitespace, and mild separators inside the name, but avoid trailing punctuation capture
    between = f"[{NIQQUD_RANGE}\\s\\u00A0\\-–—·]*"
    letters: list[str] = []
    for ch in base:
        if ch.isspace():
            letters.append("")
            continue
        letters.append(f"{re.escape(ch)}[{NIQQUD_RANGE}]*")

    letters = [l for l in letters if l]
    if not letters:
        return ""

    core = between.join(letters)
    end_boundary = r"(?=$|[^\u0590-\u05FF])"
    pattern = prefix + core + end_boundary
    return pattern


def build_search_patterns(*terms: str) -> list[str]:
    """Generate unique regex patterns for multiple terms/synonyms."""
    patterns: list[str] = []
    for term in _prepare_terms(terms):
        pat = generate_vowel_insensitive_regex(term)
        if pat:
            patterns.append(pat)
    return patterns
