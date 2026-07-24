"""Formatting helpers for study segments."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class FrontSegment:
    """Segment schema consumed by the front-end."""

    ref: str
    heText: str
    enText: Optional[str]
    position: float
    meta: Dict[str, Any]


def clean_html(text: Any) -> str:
    """Normalize whitespace and drop HTML tags."""

    if not text:
        return ""
    if isinstance(text, list):
        text = " ".join([clean_html(item) for item in text if item])
    elif not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_hebrew_only(text: Any) -> str:
    """Extract Hebrew text from Sefaria responses (lists or strings)."""

    if isinstance(text, list) and text:
        return " ".join([clean_html(x) for x in text if x])
    if isinstance(text, str):
        return text
    return ""


def to_front_segments(raw_segments: List[Dict[str, Any]]) -> List[FrontSegment]:
    """Convert raw segments into ``FrontSegment`` instances."""

    front_segments: List[FrontSegment] = []
    for index, segment in enumerate(raw_segments):
        ref = segment.get("ref", "")
        he_text = segment.get("heText") or ""
        en_text = segment.get("enText")
        meta = segment.get("meta") or {}
        position = segment.get("position")
        if position is None:
            position = index / max(len(raw_segments) - 1, 1)
        front_segments.append(
            FrontSegment(ref=ref, heText=he_text, enText=en_text, position=float(position), meta=meta)
        )
    return front_segments
