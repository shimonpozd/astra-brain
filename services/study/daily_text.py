"""Daily text assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from itertools import zip_longest
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .formatter import clean_html, extract_hebrew_only
from .range_handlers import (
    handle_inter_chapter_range,
    handle_jerusalem_talmud_range,
    try_load_range,
)
from .parsers import parse_ref

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DailyTextPayload:
    segments: List[Dict[str, Any]]
    focus_index: int
    ref: str
    he_ref: Optional[str]


async def build_full_daily_text(
    ref: str,
    sefaria_service: Any,
    index_service: Any,  # Used for canonical title resolution when primary lookups fail
    *,
    session_id: Optional[str] = None,
    redis_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """Return the full daily payload for ``ref`` using modular helpers."""

    resolved_ref, data = await _load_primary_range(ref, sefaria_service, index_service)
    if data is None:
        logger.debug("daily_text.primary_range.miss", extra={"ref": ref})
        special = await _handle_special_ranges(
            ref,
            sefaria_service,
            session_id=session_id,
            redis_client=redis_client,
        )
        if special:
            return special
        return None

    active_ref = resolved_ref or ref
    segments = await _resolve_segments(active_ref, data, sefaria_service)
    if not segments:
        logger.warning("daily_text.segments.empty", extra={"ref": active_ref})
        return None

    payload = _build_payload(active_ref, data, segments)
    if payload is None:
        logger.warning("daily_text.payload.empty", extra={"ref": ref})
        return None

    return {
        "segments": payload.segments,
        "focusIndex": payload.focus_index,
        "ref": payload.ref,
        "he_ref": payload.he_ref,
    }


async def _load_primary_range(
    ref: str,
    sefaria_service: Any,
    index_service: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Fetch the main Sefaria payload for the reference if available."""

    try:
        data = await try_load_range(sefaria_service, ref)
        if data:
            return ref, data

        normalized = _normalize_ref_with_index(ref, index_service)
        if normalized and normalized != ref:
            logger.debug(
                "daily_text.primary_range.retry_normalized",
                extra={"original": ref, "normalized": normalized},
            )
            data = await try_load_range(sefaria_service, normalized)
            if data:
                data.setdefault("ref", normalized)
                return normalized, data

        return None, None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("daily_text.primary_range.exception", extra={"ref": ref, "error": str(exc)})
        return None, None


async def _handle_special_ranges(
    ref: str,
    sefaria_service: Any,
    *,
    session_id: Optional[str],
    redis_client: Any,
) -> Optional[Dict[str, Any]]:
    """Fallback handlers for ranges that require bespoke orchestration."""

    lowered = ref.lower()
    if "jerusalem talmud" in lowered and ":" in ref and "-" in ref:
        logger.debug("daily_text.special_range.jerusalem", extra={"ref": ref})
        return await handle_jerusalem_talmud_range(
            ref,
            sefaria_service,
            session_id=session_id,
            redis_client=redis_client,
        )

    if _looks_like_inter_chapter_range(ref):
        logger.debug("daily_text.special_range.inter_chapter", extra={"ref": ref})
        return await handle_inter_chapter_range(
            ref,
            sefaria_service,
            session_id=session_id,
            redis_client=redis_client,
        )

    return None


async def _resolve_segments(
    ref: str,
    data: Dict[str, Any],
    sefaria_service: Any,
) -> List[Dict[str, Any]]:
    """Build the list of raw segments for the payload, applying fallbacks as needed."""

    segments = _extract_segments(ref, data)
    parsed = parse_ref(ref)

    if _should_use_talmud_fallback(parsed, segments, data):
        fallback_segments = await _build_talmud_segments(ref, data, sefaria_service)
        if fallback_segments:
            return fallback_segments

    return segments


def _build_payload(ref: str, data: Dict[str, Any], segments: List[Dict[str, Any]]) -> Optional[DailyTextPayload]:
    """Convert raw segments into the structured daily payload."""

    if not segments:
        return None

    formatted_segments = _format_segments_for_daily(segments)
    focus_index = 0

    return DailyTextPayload(
        segments=formatted_segments,
        focus_index=focus_index,
        ref=data.get("ref", ref),
        he_ref=data.get("heRef"),
    )


def _extract_segments(ref: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the Sefaria payload into raw segment dictionaries."""

    text = data.get("text_segments") or data.get("text")
    he = data.get("he_segments") or data.get("he")

    if _is_spanning_payload(data):
        return list(_iter_spanning_segments(ref, data, text, he))

    if isinstance(text, list):
        return list(_iter_flat_segments(ref, data, text, he))

    # Single blob of text; treat as one segment
    if isinstance(text, str) or isinstance(he, str):
        segment_ref = data.get("ref", ref)
        return [
            _raw_segment(
                segment_ref,
                text,
                he,
                data,
            )
        ]

    return []


def _is_spanning_payload(data: Dict[str, Any]) -> bool:
    segments = data.get("text_segments")
    if isinstance(segments, list) and any(isinstance(item, list) for item in segments):
        return True

    text = data.get("text")
    if isinstance(text, list) and any(isinstance(item, list) for item in text):
        return True

    return bool(data.get("isSpanning"))


def _iter_spanning_segments(
    ref: str,
    data: Dict[str, Any],
    text_sections: Any,
    he_sections: Any,
) -> Iterable[Dict[str, Any]]:
    text_sections = text_sections or []
    he_sections = he_sections or []
    spanning_refs = data.get("spanningRefs") or []

    for section_idx, text_section in enumerate(text_sections):
        he_section = he_sections[section_idx] if section_idx < len(he_sections) else []
        base_ref = spanning_refs[section_idx] if section_idx < len(spanning_refs) else data.get("ref", ref)
        if not isinstance(text_section, list):
            continue

        start_ref, start_ordinal = _parse_start_ref(base_ref)

        for line_idx, (en_line, he_line) in enumerate(
            zip_longest(text_section, he_section, fillvalue="")
        ):
            segment_ref = _compose_segment_ref(start_ref, start_ordinal + line_idx)
            yield _raw_segment(segment_ref, en_line, he_line, data)


def _iter_flat_segments(
    ref: str,
    data: Dict[str, Any],
    text_values: List[Any],
    he_values: Any,
) -> Iterable[Dict[str, Any]]:
    prefix, start_index = _parse_start_ref(data.get("ref", ref))

    for idx, en_line in enumerate(text_values):
        he_line = _safe_index(he_values, idx)
        segment_ref = _compose_segment_ref(prefix, start_index + idx)
        yield _raw_segment(segment_ref, en_line, he_line, data)


def _raw_segment(
    segment_ref: str,
    en_source: Any,
    he_source: Any,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    he_candidate = he_source
    if not he_candidate:
        he_candidate = data.get("he_text") or data.get("he")
    en_candidate = en_source
    if not en_candidate:
        en_candidate = data.get("en_text") or data.get("text")

    he_text_raw = extract_hebrew_only(he_candidate)
    he_text = str(he_text_raw or "")
    en_text = clean_html(str(en_candidate or "")).strip()

    text_value = he_text or en_text

    meta: Dict[str, Any] = {
        "title": data.get("title") or data.get("book") or "",
        "indexTitle": data.get("indexTitle") or data.get("title") or "",
        "heRef": data.get("heRef", ""),
    }

    parsed = parse_ref(segment_ref)
    if parsed.collection == "talmud":
        if parsed.page is not None:
            meta["page"] = parsed.page
        if parsed.amud:
            meta["amud"] = parsed.amud
        if parsed.segment is not None:
            meta["segment"] = parsed.segment
    else:
        if parsed.chapter is not None:
            meta["chapter"] = parsed.chapter
        if parsed.verse is not None:
            meta["verse"] = parsed.verse

    canonical_ref = segment_ref.strip()
    book_name = parsed.book or meta.get("title") or data.get("book") or ""
    if meta.get("chapter") is not None and meta.get("verse") is not None:
        canonical_ref = f"{book_name} {int(meta['chapter'])}:{int(meta['verse'])}"
    elif (
        meta.get("page") is not None
        and meta.get("amud")
        and meta.get("segment") is not None
    ):
        canonical_ref = f"{book_name} {int(meta['page'])}{meta['amud']}.{int(meta['segment'])}"

    return {
        "ref": canonical_ref,
        "text": text_value,
        "heText": he_text or text_value,
        "enText": en_text or None,
        "metadata": meta,
    }


def _format_segments_for_daily(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(segments)
    if total == 0:
        return []

    formatted: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        position = idx / (total - 1) if total > 1 else 0.5
        formatted.append(
            {
                "ref": segment["ref"],
                "text": segment["text"],
                "heText": segment["heText"],
                "position": float(position),
                "metadata": segment["metadata"],
            }
        )
    return formatted


def _compose_segment_ref(base_ref: Optional[str], ordinal: int) -> str:
    base_ref = (base_ref or "").strip()
    if not base_ref:
        return f"Segment {ordinal}"
    if ":" in base_ref:
        prefix = base_ref.rsplit(":", 1)[0]
        return f"{prefix}:{ordinal}"
    return f"{base_ref}:{ordinal}"


def _parse_start_ref(base_ref: str) -> Tuple[str, int]:
    if "-" in base_ref:
        start_ref = base_ref.split("-", 1)[0].strip()
    else:
        start_ref = base_ref.strip()

    if ":" in start_ref:
        prefix, maybe_num = start_ref.rsplit(":", 1)
        try:
            return prefix, int(maybe_num)
        except ValueError:
            return start_ref, 1
    return start_ref, 1


def _safe_index(source: Any, index: int) -> Any:
    if isinstance(source, list):
        if 0 <= index < len(source):
            return source[index]
        return ""
    return source


def _looks_like_inter_chapter_range(ref: str) -> bool:
    if "-" not in ref:
        return False

    try:
        start_part, end_part = ref.split("-", 1)
    except ValueError:
        return False

    start_match = re.search(r"(\d+)(?::\d+)?\s*$", start_part.strip())
    end_match = re.search(r"(\d+)(?::\d+)?\s*$", end_part.strip())
    if not start_match or not end_match:
        return False

    try:
        start_chapter = int(start_match.group(1))
        end_chapter = int(end_match.group(1))
    except ValueError:
        return False

    return start_chapter != end_chapter


def _normalize_ref_with_index(ref: str, index_service: Any) -> Optional[str]:
    """Use the Sefaria index aliases to coerce a ref into its canonical title."""

    resolver = getattr(index_service, "resolve_book_name", None)
    aliases = getattr(index_service, "aliases", None)
    if resolver is None and not aliases:
        return None

    book_part, suffix = _split_ref_parts(ref)
    candidate = resolver(book_part) if resolver else None
    if not candidate and aliases:
        lowered = ref.lower()
        best_alias: Optional[Tuple[str, str]] = None
        for alias_key, canonical in aliases.items():
            if lowered.startswith(alias_key):
                if best_alias is None or len(alias_key) > len(best_alias[0]):
                    best_alias = (alias_key, canonical)
        if best_alias:
            alias, canonical = best_alias
            suffix = ref[len(alias):].lstrip(", ")
            candidate = canonical

    if not candidate:
        return None

    normalized_suffix = suffix.lstrip(", ")
    if normalized_suffix:
        return f"{candidate} {normalized_suffix}".strip()
    return candidate.strip()


def _split_ref_parts(ref: str) -> Tuple[str, str]:
    """Split a reference into book/title and trailing numeric portion."""

    match = re.search(r"\d", ref)
    if not match:
        return ref.strip(" ,"), ""
    idx = match.start()
    book = ref[:idx].strip(" ,")
    suffix = ref[idx:].strip()
    return book, suffix


def _should_use_talmud_fallback(
    parsed: Any,
    segments: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> bool:
    """Determine whether the Bavli fallback should run."""

    type_label = str(data.get("type", "") or "").lower()
    is_talmud_payload = parsed.collection == "talmud" or type_label == "talmud"
    if not is_talmud_payload:
        return False
    if parsed.segment is not None:
        return False
    if len(segments) > 2:
        return False
    if not isinstance(data.get("text_segments"), list):
        return False
    return True


async def _build_talmud_segments(
    ref: str,
    data: Dict[str, Any],
    sefaria_service: Any,
) -> List[Dict[str, Any]]:
    """Fetch amud/segment level entries for Bavli references."""

    parsed = parse_ref(ref)
    book = parsed.book or data.get("indexTitle") or data.get("title")
    if not book:
        book = ref.rsplit(" ", 1)[0]

    daf = None
    if parsed.page is not None:
        daf = str(parsed.page)
    else:
        match = re.search(r"(\d+)", ref)
        if match:
            daf = match.group(1)
    if daf is None:
        return []

    sides = [parsed.amud] if parsed.amud in {"a", "b"} else ["a", "b"]
    metadata_source = {
        "title": data.get("title"),
        "indexTitle": data.get("indexTitle"),
        "heRef": data.get("heRef"),
    }

    collected: List[Dict[str, Any]] = []
    for side in sides:
        side_segments = await _collect_talmud_amud_segments(
            book,
            daf,
            side,
            sefaria_service,
            metadata_source,
        )
        collected.extend(side_segments)

    return collected


async def _collect_talmud_amud_segments(
    book: str,
    daf: str,
    side: str,
    sefaria_service: Any,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Load individual segments for a single amud."""

    segments: List[Dict[str, Any]] = []
    for segment_number in range(1, 41):
        segment_ref = f"{book} {daf}{side}:{segment_number}"
        try:
            response = await sefaria_service.get_text(segment_ref)
        except Exception as exc:  # pragma: no cover - network failure logging
            logger.warning(
                "daily_text.talmud.segment.error",
                extra={"ref": segment_ref, "error": str(exc)},
            )
            break

        if not isinstance(response, dict) or not response.get("ok"):
            break

        segment_data = response.get("data") or {}
        en_source = segment_data.get("en_text") or segment_data.get("text") or ""
        he_source = segment_data.get("he_text") or segment_data.get("he") or ""

        if isinstance(en_source, list):
            en_source = " ".join(str(item).strip() for item in en_source if item)
        if isinstance(he_source, list):
            he_source = " ".join(str(item).strip() for item in he_source if item)

        if not (en_source or he_source):
            break

        base_meta = {k: v for k, v in metadata.items() if v}
        combined_meta = {**base_meta, "book": metadata.get("indexTitle") or book}
        segment = _raw_segment(segment_ref, en_source, he_source, combined_meta)
        segments.append(segment)

    return segments

