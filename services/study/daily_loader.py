"""Daily loading planning and orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence
from time import perf_counter

from .config_schema import StudyConfig
from .formatter import clean_html, extract_hebrew_only
from .daily_text import build_full_daily_text
from .logging import log_daily_bg_loaded, log_daily_initial
from .parsers import parse_ref

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 60 * 60 * 24


@dataclass(slots=True)
class SegmentPlan:
    """Represents a slice of work for the daily loader."""

    ref: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(self.end - self.start, 0)


@dataclass(slots=True)
class DailyLoaderConfigView:
    """Immutable view of the knobs required by the daily loader."""

    initial_small: int
    initial_medium: int
    initial_large: int
    large_threshold: int
    batch_size: int
    redis_ttl_seconds: int
    lock_ttl_seconds: int
    max_total_segments: int
    retry_backoff_ms: Sequence[int]
    max_retries: int


class DailyLoader:
    """Plans and coordinates the initial/background loading workflow."""

    def __init__(
        self,
        sefaria_service: Any,
        index_service: Any,
        redis_repo: Any,
        config: StudyConfig,
    ) -> None:
        self._sefaria_service = sefaria_service
        self._index_service = index_service
        self._redis_repo = redis_repo
        self._sleep_between_requests = 0.1
        self.update_config(config)

    def update_config(self, config: StudyConfig) -> None:
        daily = config.daily
        self._config = DailyLoaderConfigView(
            initial_small=daily.initial_small,
            initial_medium=daily.initial_medium,
            initial_large=daily.initial_large,
            large_threshold=daily.large_threshold,
            batch_size=daily.batch_size,
            redis_ttl_seconds=daily.redis_ttl_days * _SECONDS_PER_DAY,
            lock_ttl_seconds=daily.lock_ttl_sec,
            max_total_segments=daily.max_total_segments,
            retry_backoff_ms=tuple(daily.retry_backoff_ms),
            max_retries=daily.max_retries,
        )

    @property
    def config(self) -> DailyLoaderConfigView:
        return self._config

    def plan_initial_segments(self, ref: str, total_segments: int) -> List[SegmentPlan]:
        """Return segment plans that should be served in the first response."""

        initial_target = self._initial_target(total_segments)
        return list(self._chunk(ref, 0, min(initial_target, total_segments)))

    def plan_background_segments(
        self, ref: str, total_segments: int, already_loaded: int
    ) -> List[SegmentPlan]:
        """Return remaining segment plans to be processed in the background."""

        if already_loaded >= total_segments:
            return []
        start = already_loaded
        remaining = total_segments - already_loaded
        return list(self._chunk(ref, start, remaining))

    async def load_initial(
        self,
        *,
        ref: str,
        session_id: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Fetch the initial set of segments and persist them for polling clients."""

        start_ts = perf_counter()
        ttl = ttl_seconds or self._config.redis_ttl_seconds
        try:
            full_payload = await build_full_daily_text(
                ref,
                self._sefaria_service,
                self._index_service,
                session_id=session_id,
                redis_client=None,
            )
        except Exception as exc:  # pragma: no cover - logging only
            logger.error(
                "initial load failed",
                extra={"ref": ref, "session_id": session_id, "error": str(exc)},
                exc_info=True,
            )
            raise

        if not full_payload:
            logger.warning(
                "initial load returned no payload",
                extra={"ref": ref, "session_id": session_id},
            )
            await self._redis_repo.clear_segments(session_id)
            await self._redis_repo.set_total(session_id, 0, ttl)
            duration_ms = (perf_counter() - start_ts) * 1000.0
            log_daily_initial(ref, loaded=0, total=0, duration_ms=duration_ms)
            return {
                "segments": [],
                "focusIndex": 0,
                "ref": ref,
                "total_segments": 0,
                "loaded": 0,
                "remaining_plan": [],
            }

        segments = list(full_payload.get("segments") or [])
        total_segments = len(segments)
        initial_plan = self.plan_initial_segments(ref, total_segments)

        await self._redis_repo.clear_segments(session_id)

        loaded_segments: List[Dict[str, Any]] = []
        loaded_count = 0

        for chunk in initial_plan:
            chunk_segments = segments[chunk.start : chunk.end]
            if not chunk_segments:
                continue
            for segment in chunk_segments:
                payload = self._segment_to_redis_payload(segment, ref)
                await self._redis_repo.push_segment(
                    session_id,
                    json.dumps(payload, ensure_ascii=False),
                    ttl,
                )
                loaded_segments.append(segment)
                loaded_count += 1

        await self._redis_repo.set_total(session_id, total_segments, ttl)

        remaining_plan = self.plan_background_segments(ref, total_segments, loaded_count)

        range_book_chapter, range_start_verse = self._split_book_and_verse(
            segments[0].get("ref") if segments else None
        )

        duration_ms = (perf_counter() - start_ts) * 1000.0
        log_daily_initial(full_payload.get("ref", ref), loaded_count, total_segments, duration_ms)

        return {
            "segments": loaded_segments,
            "focusIndex": full_payload.get("focusIndex", 0),
            "ref": full_payload.get("ref", ref),
            "he_ref": full_payload.get("he_ref"),
            "total_segments": total_segments,
            "loaded": loaded_count,
            "remaining_plan": [
                self._plan_to_dict(item, segments, range_book_chapter, range_start_verse, total_segments)
                for item in remaining_plan
            ],
        }

    async def load_background(
        self,
        *,
        ref: str,
        session_id: str,
        start_verse: int,
        end_verse: int,
        book_chapter: str,
        already_loaded: int,
        total_segments: int,
        ttl_seconds: int | None = None,
    ) -> None:
        """Load remaining segments for the session in the background."""

        task_id = self._background_task_id(ref, start_verse, end_verse)
        if await self._redis_repo.is_task_marked(session_id, task_id):
            logger.debug("background task already marked", extra={"session_id": session_id, "task_id": task_id})
            return

        lock_ttl = self._config.lock_ttl_seconds or 300
        acquired = await self._redis_repo.try_lock(session_id, lock_ttl)
        if not acquired:
            logger.debug("background lock busy", extra={"session_id": session_id})
            await self._redis_repo.set_loading(session_id, lock_ttl)
            return

        await self._redis_repo.set_loading(session_id, lock_ttl)
        try:
            await self._redis_repo.mark_task(session_id, task_id, lock_ttl)
            ttl = ttl_seconds or self._config.redis_ttl_seconds
            await self._load_segments(
                ref=ref,
                session_id=session_id,
                start_verse=start_verse,
                end_verse=end_verse,
                book_chapter=book_chapter,
                already_loaded=already_loaded,
                total_segments=total_segments,
                ttl_seconds=ttl,
            )
        except Exception as exc:  # pragma: no cover - logging only
            logger.error(
                "background load failed",
                extra={"session_id": session_id, "ref": ref, "error": str(exc)},
                exc_info=True,
            )
        finally:
            await self._redis_repo.clear_loading(session_id)
            await self._redis_repo.release_lock(session_id)

    async def _load_segments(
        self,
        *,
        ref: str,
        session_id: str,
        start_verse: int,
        end_verse: int,
        book_chapter: str,
        already_loaded: int,
        total_segments: int,
        ttl_seconds: int,
    ) -> None:
        start_ts = perf_counter()
        next_verse = start_verse + already_loaded
        start_cursor = max(next_verse, start_verse)
        planned_total = already_loaded + max(0, end_verse - start_cursor + 1)
        logger.debug(
            "study.daily.background.start",
            extra={
                "session_id": session_id,
                "ref": ref,
                "book_chapter": book_chapter,
                "start_verse": start_cursor,
                "end_verse": end_verse,
                "already_loaded": already_loaded,
                "planned_total": planned_total,
            },
        )

        segments_added = 0
        for verse_num in range(start_cursor, end_verse + 1):
            verse_ref = f"{book_chapter}:{verse_num}"
            try:
                verse_result = await self._sefaria_service.get_text(verse_ref)
                if not (verse_result.get("ok") and verse_result.get("data")):
                    continue

                verse_data = verse_result["data"]
                en_text = verse_data.get("en_text") or verse_data.get("text", "")
                he_text = verse_data.get("he_text") or verse_data.get("he", "")

                segment_data = {
                    "ref": verse_ref,
                    "en_text": clean_html(en_text),
                    "he_text": clean_html(extract_hebrew_only(he_text)),
                    "title": verse_data.get("title", ref),
                    "indexTitle": verse_data.get("indexTitle", ""),
                    "heRef": verse_data.get("heRef", ""),
                }

                payload = json.dumps(segment_data, ensure_ascii=False)
                await self._redis_repo.push_segment(session_id, payload, ttl_seconds)

                segments_loaded = already_loaded + (verse_num - start_cursor + 1)
                segments_added += 1
                total_to_record = max(total_segments, planned_total)
                await self._redis_repo.set_total(session_id, total_to_record, ttl_seconds)

                logger.debug(
                    "study.daily.background.segment_loaded",
                    extra={
                        "session_id": session_id,
                        "ref": ref,
                        "segment_ref": verse_ref,
                        "segments_loaded": segments_loaded,
                        "planned_total": planned_total,
                    },
                )

                if self._sleep_between_requests:
                    await asyncio.sleep(self._sleep_between_requests)
            except Exception as exc:  # pragma: no cover - logging only
                logger.error(
                    "study.daily.background.error",
                    extra={
                        "session_id": session_id,
                        "ref": ref,
                        "segment_ref": verse_ref,
                        "message": str(exc),
                    },
                    exc_info=True,
                )

        logger.debug(
            "study.daily.background.complete",
            extra={
                "session_id": session_id,
                "ref": ref,
                "book_chapter": book_chapter,
                "segments_added": segments_added,
                "planned_total": planned_total,
            },
        )
        duration_ms = (perf_counter() - start_ts) * 1000.0
        log_daily_bg_loaded(ref, segments_added, duration_ms, retry=already_loaded > 0)

    @staticmethod
    def _segment_to_redis_payload(segment: Dict[str, Any], fallback_ref: str) -> Dict[str, Any]:
        raw_metadata = segment.get("metadata") or {}
        metadata = dict(raw_metadata)
        ref_value = segment.get("ref", fallback_ref) or fallback_ref
        parsed = parse_ref(ref_value)
        en_text = segment.get("text") or segment.get("enText", "")
        he_text = segment.get("heText", "")

        meta_payload = {
            "title": metadata.get("title", fallback_ref),
            "indexTitle": metadata.get("indexTitle", ""),
            "heRef": metadata.get("heRef", ""),
        }
        if parsed.collection == "talmud":
            if parsed.page is not None:
                meta_payload["page"] = parsed.page
            if parsed.amud:
                meta_payload["amud"] = parsed.amud
            if parsed.segment is not None:
                meta_payload["segment"] = parsed.segment
        else:
            if parsed.chapter is not None:
                meta_payload["chapter"] = parsed.chapter
            if parsed.verse is not None:
                meta_payload["verse"] = parsed.verse

        return {
            "ref": ref_value,
            "en_text": clean_html(en_text or ""),
            "he_text": clean_html(he_text or ""),
            "title": meta_payload.get("title"),
            "indexTitle": meta_payload.get("indexTitle"),
            "heRef": meta_payload.get("heRef"),
            "metadata": meta_payload,
        }

    @staticmethod
    def _plan_to_dict(
        plan: SegmentPlan,
        segments: List[Dict[str, Any]],
        range_book_chapter: str | None,
        range_start_verse: int | None,
        total_segments: int,
    ) -> Dict[str, Any]:
        start_ref: str | None = None
        end_ref: str | None = None
        if 0 <= plan.start < len(segments):
            start_ref = segments[plan.start].get("ref")
        end_index = plan.end - 1
        if 0 <= end_index < len(segments):
            end_ref = segments[end_index].get("ref")
        book_chapter: str | None = None
        chunk_start_verse: int | None = None
        chunk_end_verse: int | None = None

        if start_ref:
            book_chapter, chunk_start_verse = DailyLoader._split_book_and_verse(start_ref)
        if end_ref:
            end_book, parsed_end = DailyLoader._split_book_and_verse(end_ref)
            if parsed_end is not None:
                chunk_end_verse = parsed_end
            if not book_chapter:
                book_chapter = end_book

        if not book_chapter:
            book_chapter = range_book_chapter

        return {
            "ref": plan.ref,
            "start": plan.start,
            "end": plan.end,
            "start_ref": start_ref,
            "end_ref": end_ref,
            "book_chapter": book_chapter,
            "chunk_start_verse": chunk_start_verse,
            "end_verse": chunk_end_verse,
            "range_start_verse": range_start_verse,
            "total_segments": total_segments,
        }

    @staticmethod
    def _split_book_and_verse(ref: str | None) -> tuple[str | None, int | None]:
        if not ref or ":" not in ref:
            return None, None
        book_part, verse_part = ref.rsplit(":", 1)
        if verse_part.isdigit():
            return book_part, int(verse_part)
        return None, None

    def _initial_target(self, total_segments: int) -> int:
        view = self._config
        if total_segments <= view.initial_small:
            return total_segments
        if total_segments <= view.large_threshold:
            return view.initial_medium
        return view.initial_large

    def _chunk(self, ref: str, start: int, count: int) -> Iterable[SegmentPlan]:
        view = self._config
        cursor = start
        end = start + max(count, 0)
        batch_size = max(view.batch_size, 1)
        while cursor < end:
            yield SegmentPlan(ref=ref, start=cursor, end=min(cursor + batch_size, end))
            cursor += batch_size

    @staticmethod
    def _background_task_id(ref: str, start_verse: int, end_verse: int) -> str:
        return f"{ref}:{start_verse}:{end_verse}"

