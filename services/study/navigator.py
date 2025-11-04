"""Window navigation utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..sefaria_index import get_book_structure
from .parsers import detect_collection


async def generate_neighbors(
    base_ref: str,
    count: int,
    *,
    direction: str,
    sefaria_service: Any,
    index_service: Any,
) -> List[Dict[str, Any]]:
    """Return neighboring segments for ``base_ref`` in the given direction."""

    if count <= 0 or not base_ref:
        return []

    parsed_ref = _parse_ref(base_ref)
    if not parsed_ref:
        return []

    toc_data = getattr(index_service, "toc", None)
    if (not toc_data) and hasattr(index_service, "load"):
        try:
            await index_service.load()
            toc_data = getattr(index_service, "toc", None)
        except Exception:  # pragma: no cover - fallback only
            toc_data = getattr(index_service, "toc", None)
    book_name = parsed_ref["book"]
    book_structure = get_book_structure(book_name, toc_data)
    chapter_cache: Dict[int, Optional[int]] = {}
    if not book_structure and hasattr(index_service, "resolve_book_name"):
        resolver = getattr(index_service, "resolve_book_name")
        resolved = resolver(book_name) if callable(resolver) else None
        if not resolved and hasattr(index_service, "aliases"):
            resolved = getattr(index_service, "aliases", {}).get(book_name.lower())
        if resolved and resolved != book_name:
            book_name = resolved
            parsed_ref["book"] = resolved
            book_structure = get_book_structure(book_name, toc_data)

    generated: List[Dict[str, Any]] = []
    current_parts = parsed_ref.copy()

    step = 1 if direction == "next" else -1
    max_attempts = count * 5

    for _ in range(max_attempts):
        if len(generated) >= count:
            break

        candidate_ref = await _advance_ref(
            current_parts,
            step,
            direction,
            book_structure,
            chapter_cache,
            sefaria_service,
        )
        if not candidate_ref:
            break

        result = await sefaria_service.get_text(candidate_ref)
        if result.get("ok") and result.get("data"):
            generated.append(result["data"])
        else:
            if current_parts["type"] == "talmud" and direction == "next":
                if current_parts.get("amud") == "a":
                    current_parts["amud"] = "b"
                    current_parts["segment"] = 0
                    continue
                current_parts["page"] += 1
                current_parts["amud"] = "a"
                current_parts["segment"] = 0
                continue
            break

    if direction == "prev":
        generated.reverse()
    return generated


def _parse_ref(ref: str) -> Optional[Dict[str, Any]]:
    talmud_match = re.match(r"([\w\s'.]+) (\d+)([ab])(?:[.:](\d+))?", ref)
    if talmud_match:
        return {
            "type": "talmud",
            "book": talmud_match.group(1).strip(),
            "page": int(talmud_match.group(2)),
            "amud": talmud_match.group(3),
            "segment": int(talmud_match.group(4)) if talmud_match.group(4) else 1,
        }

    bible_match = re.match(r"([\w\s'.]+) (\d+):(\d+)", ref)
    if bible_match:
        return {
            "type": "bible",
            "book": bible_match.group(1).strip(),
            "chapter": int(bible_match.group(2)),
            "verse": int(bible_match.group(3)),
        }

    mishnah_match = re.match(r"Mishnah ([\w\s'.]+) (\d+):(\d+)", ref, re.IGNORECASE)
    if mishnah_match:
        return {
            "type": "mishnah",
            "book": mishnah_match.group(1).strip(),
            "chapter": int(mishnah_match.group(2)),
            "verse": int(mishnah_match.group(3)),
        }

    return None


async def _advance_ref(
    parts: Dict[str, Any],
    step: int,
    direction: str,
    book_structure: Optional[Dict[str, Any]],
    chapter_cache: Dict[int, Optional[int]],
    sefaria_service: Any,
) -> Optional[str]:
    ref_type = parts.get("type")
    if ref_type == "talmud":
        current_segment = parts.get("segment", 1) or 1
        next_segment = current_segment + step
        if next_segment > 0:
            parts["segment"] = next_segment
            return f"{parts['book']} {parts['page']}{parts['amud']}.{next_segment}"
        return None

    if ref_type in {"bible", "mishnah"} and book_structure:
        chapter_index = parts["chapter"] - 1
        new_verse = parts["verse"] + step
        lengths = book_structure.get("lengths") or []

        length = None
        if lengths and 0 <= chapter_index < len(lengths):
            length = lengths[chapter_index]
        if length is None:
            length = await _ensure_chapter_length(parts["book"], parts["chapter"], chapter_cache, sefaria_service)
        if not length:
            return None

        if 1 <= new_verse <= length:
            parts["verse"] = new_verse
        elif new_verse > length and direction == "next":
            parts["chapter"] += 1
            next_length = await _ensure_chapter_length(parts["book"], parts["chapter"], chapter_cache, sefaria_service)
            if not next_length:
                return None
            parts["verse"] = 1
        elif new_verse < 1 and direction == "prev":
            prev_chapter = parts["chapter"] - 1
            prev_length = await _ensure_chapter_length(parts["book"], prev_chapter, chapter_cache, sefaria_service)
            if not prev_length:
                return None
            parts["chapter"] = prev_chapter
            parts["verse"] = prev_length
        else:
            return None
        return f"{parts['book']} {parts['chapter']}:{parts['verse']}"
    return None


async def _ensure_chapter_length(
    book: str,
    chapter: int,
    cache: Dict[int, Optional[int]],
    sefaria_service: Any,
) -> Optional[int]:
    if chapter <= 0:
        return None
    if chapter in cache:
        return cache[chapter]

    try:
        chapter_ref = f"{book} {chapter}"
        response = await sefaria_service.get_text(chapter_ref)
        if response.get("ok") and response.get("data"):
            data = response["data"]
            segments: Optional[List[Any]] = None
            if isinstance(data, dict):
                # Prefer segmented payloads when available; fall back to raw lists/strings.
                candidates = (
                    data.get("text_segments"),
                    data.get("he_segments"),
                    data.get("text"),
                    data.get("he"),
                )
                for candidate in candidates:
                    if isinstance(candidate, list):
                        segments = [seg for seg in candidate if seg]
                        break
                    if isinstance(candidate, str):
                        split_lines = [line for line in candidate.splitlines() if line.strip()]
                        if split_lines:
                            segments = split_lines
                            break
            elif isinstance(data, list):
                segments = [seg for seg in data if seg]

            if segments is not None:
                cache[chapter] = len(segments)
                return cache[chapter]
            cache[chapter] = None
    except Exception:  # pragma: no cover
        cache[chapter] = None
    return cache.get(chapter)

