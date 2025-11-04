"""Reference parsing helpers for the modular study service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

Collection = Literal["talmud", "bible", "mishnah", "midrash", "commentary", "unknown"]


@dataclass(slots=True)
class ParsedRef:
    """Structured representation of a study reference."""

    book: str
    chapter: Optional[int] = None
    verse: Optional[int] = None
    amud: Optional[str] = None
    page: Optional[int] = None
    segment: Optional[int] = None
    collection: Collection = "unknown"


def detect_collection(ref: str) -> Collection:
    """Return the inferred collection for a reference string."""

    if not ref:
        return "unknown"
    lowered = ref.lower()
    if " on " in lowered or "commentary" in lowered:
        return "commentary"
    talmud_tractates = {
        "shabbat",
        "berakhot",
        "pesachim",
        "ketubot",
        "gittin",
        "kiddushin",
        "bava kamma",
        "bava metzia",
        "bava batra",
        "sanhedrin",
        "makkot",
        "eruvin",
        "beitzah",
        "taanit",
        "megillah",
        "sukkah",
    }
    if any(name in lowered for name in talmud_tractates):
        return "talmud"
    bible_books = {
        "genesis",
        "exodus",
        "leviticus",
        "numbers",
        "deuteronomy",
        "joshua",
        "judges",
        "samuel",
        "kings",
        "isaiah",
        "jeremiah",
        "ezekiel",
        "proverbs",
        "psalms",
        "job",
        "chronicles",
    }
    if any(name in lowered for name in bible_books):
        return "bible"
    if "mishnah" in lowered or "mishna" in lowered:
        return "mishnah"
    if "midrash" in lowered:
        return "midrash"
    if "daf" in lowered or "amud" in lowered:
        return "talmud"
    return "unknown"


def parse_ref(ref: str) -> ParsedRef:
    """Parse a textual reference into a structured object."""

    if not ref:
        return ParsedRef(book="", collection="unknown")

    ref = ref.strip()
    collection = detect_collection(ref)
    book = ref

    # First, check if this is likely a Bible book to prioritize Bible format
    ref_lower = ref.lower()
    bible_books = ['genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings', 'isaiah', 'jeremiah', 'ezekiel', 'psalms', 'proverbs', 'job', 'song', 'ruth', 'lamentations', 'ecclesiastes', 'esther', 'daniel', 'ezra', 'nehemiah', 'chronicles']
    is_likely_bible = any(book_name in ref_lower for book_name in bible_books)

    if is_likely_bible:
        # For Bible books, prioritize Bible format
        bible_match = re.match(r"([\w\s'.]+) (\d+):(\d+)", ref)
        if bible_match:
            book = bible_match.group(1).strip()
            return ParsedRef(
                book=book,
                chapter=int(bible_match.group(2)),
                verse=int(bible_match.group(3)),
                collection="bible",
            )
        
        # Chapter-only format for Bible books
        chapter_match = re.match(r"([\w\s'.]+) (\d+)$", ref)
        if chapter_match:
            book = chapter_match.group(1).strip()
            return ParsedRef(
                book=book,
                chapter=int(chapter_match.group(2)),
                collection="bible",
            )
        
        # Only then check Talmud format for Bible books (should rarely match)
        talmud_match = re.match(r"([\w\s'.]+) (\d+)([ab])(?:[.:](\d+))?", ref)
        if talmud_match:
            book = talmud_match.group(1).strip()
            return ParsedRef(
                book=book,
                page=int(talmud_match.group(2)),
                amud=talmud_match.group(3),
                segment=int(talmud_match.group(4)) if talmud_match.group(4) else None,
                collection="talmud",
            )
    else:
        # For non-Bible books, try Talmud format first
        talmud_match = re.match(r"([\w\s'.]+) (\d+)([ab])(?:[.:](\d+))?", ref)
        if talmud_match:
            book = talmud_match.group(1).strip()
            return ParsedRef(
                book=book,
                page=int(talmud_match.group(2)),
                amud=talmud_match.group(3),
                segment=int(talmud_match.group(4)) if talmud_match.group(4) else None,
                collection="talmud",
            )

        # Bible/Mishnah format e.g. "Exodus 2:1" or "Mishnah Berakhot 1:1"
        bible_match = re.match(r"([\w\s'.]+) (\d+):(\d+)", ref)
        if bible_match:
            book = bible_match.group(1).strip()
            return ParsedRef(
                book=book,
                chapter=int(bible_match.group(2)),
                verse=int(bible_match.group(3)),
                collection=collection if collection != "unknown" else "bible",
            )

        # Chapter-only format e.g. "Genesis 2"
        chapter_match = re.match(r"([\w\s'.]+) (\d+)$", ref)
        if chapter_match:
            book = chapter_match.group(1).strip()
            return ParsedRef(
                book=book,
                chapter=int(chapter_match.group(2)),
                collection=collection if collection != "unknown" else "bible",
            )

    return ParsedRef(book=book, collection=collection)
