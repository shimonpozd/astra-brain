"""Range handling utilities for study segments."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .formatter import clean_html, extract_hebrew_only
from .logging import log_range_detected

logger = logging.getLogger(__name__)


def _parse_inter_chapter_bounds(ref: str) -> Optional[Tuple[str, str, int, int, int, int]]:
    """Parse a cross-chapter range like ``Genesis 1:21-2:15`` (book may be omitted on the end)."""
    if "-" not in ref:
        return None

    start_raw, end_raw = [part.strip() for part in ref.split("-", 1)]
    start_match = re.match(r"(.+?)\s+(\d+):(\d+)$", start_raw)
    end_match = re.match(r"(?:(.+?)\s+)?(\d+):(\d+)$", end_raw)

    if not start_match or not end_match:
        return None

    start_book = start_match.group(1).strip()
    start_chapter = int(start_match.group(2))
    start_verse = int(start_match.group(3))

    end_book = end_match.group(1).strip() if end_match.group(1) else start_book
    end_chapter = int(end_match.group(2))
    end_verse = int(end_match.group(3))

    return start_book, end_book, start_chapter, start_verse, end_chapter, end_verse


async def try_load_range(sefaria_service: Any, ref: str) -> Optional[Dict[str, Any]]:
    """Attempt to load a range from Sefaria."""

    try:
        logger.debug("study.range.try_load.start", extra={"ref": ref})
        result = await sefaria_service.get_text(ref)
        ok = result.get("ok")
        has_data = bool(result.get("data"))
        logger.debug(
            "study.range.try_load.result",
            extra={"ref": ref, "ok": ok, "has_data": has_data},
        )
        if ok and has_data:
            return result["data"]
        logger.warning(
            "study.range.try_load.miss",
            extra={"ref": ref, "ok": ok, "has_data": has_data},
        )
        return None
    except Exception as exc:  # pragma: no cover - logging only
        logger.exception(
            "study.range.try_load.exception",
            extra={"ref": ref, "error": str(exc)},
        )
        return None


async def handle_jerusalem_talmud_range(
    ref: str,
    sefaria_service: Any,
    *,
    session_id: Optional[str] = None,
    redis_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """Handle Jerusalem Talmud ranges with triple structure: Chapter:Mishnah:Halakhah."""

    log_range_detected(ref, "jerusalem_talmud")
    logger.debug(
        "study.range.jerusalem.start",
        extra={"ref": ref, "session_id": session_id},
    )
    if "-" not in ref:
        return None

    start_ref, end_ref = ref.split("-", 1)
    book_parts = start_ref.split()
    book_name = " ".join(book_parts[:-1])
    start_parts = start_ref.split(":")[-3:]
    end_parts = end_ref.split(":")[-3:]

    if len(start_parts) < 3 or len(end_parts) < 3:
        logger.warning(
            "study.range.jerusalem.invalid_format",
            extra={"ref": ref},
        )
        return None

    start_chapter = int(start_parts[0])
    start_mishnah = int(start_parts[1])
    start_halakhah = int(start_parts[2])
    end_chapter = int(end_parts[0])
    end_mishnah = int(end_parts[1])
    end_halakhah = int(end_parts[2])

    logger.debug(
        "study.range.jerusalem.bounds",
        extra={
            "ref": ref,
            "start_chapter": start_chapter,
            "start_mishnah": start_mishnah,
            "start_halakhah": start_halakhah,
            "end_chapter": end_chapter,
            "end_mishnah": end_mishnah,
            "end_halakhah": end_halakhah,
        },
    )

    all_segments: List[Dict[str, Any]] = []
    current_chapter = start_chapter
    current_mishnah = start_mishnah
    current_halakhah = start_halakhah

    while current_chapter <= end_chapter:
        max_mishnah = end_mishnah if current_chapter == end_chapter else 20
        min_mishnah = current_mishnah if current_chapter == start_chapter else 1

        for mishnah in range(min_mishnah, max_mishnah + 1):
            max_halakhah_value = (
                end_halakhah
                if (current_chapter == end_chapter and mishnah == end_mishnah)
                else 20
            )
            min_halakhah_value = (
                current_halakhah
                if (current_chapter == start_chapter and mishnah == start_mishnah)
                else 1
            )

            for halakhah in range(min_halakhah_value, max_halakhah_value + 1):
                segment_ref = f"{book_name} {current_chapter}:{mishnah}:{halakhah}"
                try:
                    segment_result = await sefaria_service.get_text(segment_ref)
                except Exception as exc:  # pragma: no cover - logging only
                    logger.warning(
                        "study.range.jerusalem.segment_error",
                        extra={"ref": segment_ref, "error": str(exc)},
                    )
                    break

                if not (segment_result.get("ok") and segment_result.get("data")):
                    break

                data = segment_result["data"]
                en_text = data.get("en_text", "") or data.get("text", "")
                he_text = data.get("he_text", "") or data.get("he", "")
                if not (en_text or he_text):
                    break

                all_segments.append(
                    {
                        "ref": segment_ref,
                        "en_text": en_text,
                        "he_text": extract_hebrew_only(he_text),
                        "title": data.get("title", ref),
                        "indexTitle": data.get("indexTitle", ""),
                        "heRef": data.get("heRef", ""),
                    }
                )
                logger.debug(
                    "study.range.jerusalem.segment_loaded",
                    extra={"ref": segment_ref},
                )

            current_halakhah = 1
        current_mishnah = 1
        current_chapter += 1

    if not all_segments:
        logger.warning(
            "study.range.jerusalem.empty",
            extra={"ref": ref},
        )
        return None

    formatted_segments: List[Dict[str, Any]] = []
    total_segments = len(all_segments)
    for index, seg in enumerate(all_segments):
        formatted_segments.append(
            {
                "ref": seg.get("ref"),
                "text": seg.get("en_text", ""),
                "heText": seg.get("he_text", ""),
                "position": (index / (total_segments - 1)) if total_segments > 1 else 0.5,
                "metadata": {
                    "title": seg.get("title"),
                    "indexTitle": seg.get("indexTitle"),
                    "heRef": seg.get("heRef"),
                },
            }
        )

    if session_id and redis_client:
        try:
            count_key = f"daily:sess:{session_id}:total_segments"
            await redis_client.set(count_key, total_segments, ex=3600 * 24 * 7)
            logger.debug(
                "study.range.jerusalem.redis_total_set",
                extra={"ref": ref, "session_id": session_id, "total_segments": total_segments},
            )
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning(
                "study.range.jerusalem.redis_total_failed",
                extra={"ref": ref, "session_id": session_id, "error": str(exc)},
            )

    return {
        "segments": formatted_segments,
        "focusIndex": 0,
        "ref": ref,
        "he_ref": all_segments[0].get("heRef") if all_segments else None,
    }


async def _load_chapter_segments(
    sefaria_service: Any,
    book: str,
    chapter: int,
    start_verse: int,
    end_verse: Optional[int],
) -> List[Dict[str, Any]]:
    """Load a run of verses from a single chapter until ``end_verse`` or the API stops returning data."""

    segments: List[Dict[str, Any]] = []
    verse = start_verse
    while True:
        if end_verse is not None and verse > end_verse:
            break

        verse_ref = f"{book} {chapter}:{verse}"
        try:
            verse_result = await sefaria_service.get_text(verse_ref)
        except Exception as exc:  # pragma: no cover - logging only
            logger.error(
                "study.range.inter_chapter.fetch_error",
                extra={"ref": verse_ref, "error": str(exc)},
            )
            break

        if not (verse_result.get("ok") and verse_result.get("data")):
            break

        data = verse_result["data"]
        en_text = data.get("en_text", "") or data.get("text", "")
        he_text = data.get("he_text", "") or data.get("he", "")
        if not (en_text or he_text):
            break

        segments.append(
            {
                "ref": verse_ref,
                "en_text": clean_html(en_text),
                "he_text": clean_html(extract_hebrew_only(he_text)),
                "title": data.get("title", verse_ref),
                "indexTitle": data.get("indexTitle", ""),
                "heRef": data.get("heRef", ""),
            }
        )
        verse += 1

    return segments


async def handle_inter_chapter_range(
    ref: str,
    sefaria_service: Any,
    *,
    session_id: Optional[str] = None,
    redis_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """Handle inter-chapter ranges by loading each chapter separately."""

    log_range_detected(ref, "inter_chapter")
    logger.debug(
        "study.range.inter_chapter.start",
        extra={"ref": ref, "session_id": session_id},
    )

    bounds = _parse_inter_chapter_bounds(ref)
    if not bounds:
        logger.warning(
            "study.range.inter_chapter.invalid_format",
            extra={"ref": ref},
        )
        return None

    start_book, end_book, start_chapter, start_verse, end_chapter, end_verse = bounds

    logger.debug(
        "study.range.inter_chapter.bounds",
        extra={
            "ref": ref,
            "start_book": start_book,
            "end_book": end_book,
            "start_chapter": start_chapter,
            "start_verse": start_verse,
            "end_chapter": end_chapter,
            "end_verse": end_verse,
        },
    )

    segments: List[Dict[str, Any]] = []

    # Same-book range: walk chapters from start to end.
    if start_book == end_book:
        for chapter in range(start_chapter, end_chapter + 1):
            chapter_start = start_verse if chapter == start_chapter else 1
            chapter_end = end_verse if chapter == end_chapter else None
            segments.extend(
                await _load_chapter_segments(
                    sefaria_service,
                    start_book,
                    chapter,
                    chapter_start,
                    chapter_end,
                )
            )
    else:
        # Load to the end of the starting book until the API stops responding with data.
        chapter = start_chapter
        safety = 0
        while safety < 150:  # guard against runaway loops
            safety += 1
            chapter_segments = await _load_chapter_segments(
                sefaria_service,
                start_book,
                chapter,
                start_verse if chapter == start_chapter else 1,
                None,
            )
            if not chapter_segments:
                break
            segments.extend(chapter_segments)
            chapter += 1

        # Then load from the beginning of the destination book through the end bounds.
        for chapter in range(1, end_chapter + 1):
            chapter_segments = await _load_chapter_segments(
                sefaria_service,
                end_book,
                chapter,
                1,
                end_verse if chapter == end_chapter else None,
            )
            if not chapter_segments:
                break
            segments.extend(chapter_segments)

    if not segments:
        logger.warning(
            "study.range.inter_chapter.empty",
            extra={"ref": ref},
        )
        return None

    formatted_segments: List[Dict[str, Any]] = []
    total_segments = len(segments)
    for index, seg in enumerate(segments):
        formatted_segments.append(
            {
                "ref": seg.get("ref"),
                "text": seg.get("en_text", ""),
                "heText": seg.get("he_text", ""),
                "position": (index / (total_segments - 1)) if total_segments > 1 else 0.5,
                "metadata": {
                    "title": seg.get("title"),
                    "indexTitle": seg.get("indexTitle"),
                    "heRef": seg.get("heRef"),
                },
            }
        )

    if session_id and redis_client:
        try:
            count_key = f"daily:sess:{session_id}:total_segments"
            await redis_client.set(count_key, total_segments, ex=3600 * 24 * 7)
            logger.debug(
                "study.range.inter_chapter.redis_total_set",
                extra={"ref": ref, "session_id": session_id, "total_segments": total_segments},
            )
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning(
                "study.range.inter_chapter.redis_total_failed",
                extra={"ref": ref, "session_id": session_id, "error": str(exc)},
            )

    return {
        "segments": formatted_segments,
        "focusIndex": 0,
        "ref": ref,
        "he_ref": segments[0].get("heRef") if segments else None,
    }
