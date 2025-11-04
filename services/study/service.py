"""Thin facade over modular study collaborators."""

from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from .bookshelf import BookshelfService
from .config_schema import StudyConfig, load_study_config
from .daily_loader import DailyLoader
from .daily_text import build_full_daily_text
from .formatter import clean_html, extract_hebrew_only
from .navigator import generate_neighbors
from .parsers import parse_ref
from .prompt_builder import (
    PromptParts,
    TokenCounterFactory,
    TiktokenCounter,
    RatioTokenCounter,
    assemble_prompt,
)
from .prompt_budget import PromptBudget, build_budget
from .redis_repo import StudyRedisRepository
from .logging import log_window_built


class StudyService:
    """Facade entry point that orchestrates modular study collaborators."""

    def __init__(
        self,
        sefaria_service: Any,
        index_service: Any,
        redis_client: Any,
        config: StudyConfig | Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger("brain_service.study.facade")
        self._sefaria_service = sefaria_service
        self._index_service = index_service
        self._redis = redis_client

        self._study_config = (
            config if isinstance(config, StudyConfig) else load_study_config(config)
        )

        self._redis_repo = StudyRedisRepository(self._redis)
        self._token_counters = TokenCounterFactory()
        self._daily_loader = DailyLoader(
            sefaria_service=self._sefaria_service,
            index_service=self._index_service,
            redis_repo=self._redis_repo,
            config=self._study_config,
        )
        self._bookshelf_service = BookshelfService(
            sefaria_service=self._sefaria_service,
            index_service=self._index_service,
            redis_repo=self._redis_repo,
            config=self._study_config,
        )

    @property
    def study_config(self) -> StudyConfig:
        return self._study_config

    def update_config(self, config: StudyConfig | Dict[str, Any]) -> None:
        """Apply a new study configuration at runtime."""

        self._study_config = config if isinstance(config, StudyConfig) else load_study_config(config)
        self._daily_loader.update_config(self._study_config)
        self._bookshelf_service.update_config(self._study_config)

    # ------------------------------------------------------------------
    # Windowed study flows
    # ------------------------------------------------------------------
    async def get_text_with_window(
        self,
        ref: str,
        window_size: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the surrounding window for ``ref`` using the configured bounds."""

        resolved_window = self._resolve_window_size(window_size)
        self._logger.debug(
            "study.facade.get_text_with_window",
            extra={"ref": ref, "window_size": resolved_window},
        )

        start_ts = perf_counter()
        focus_result = await self._sefaria_service.get_text(ref)
        if not focus_result or not focus_result.get("ok"):
            self._logger.warning(
                "study.facade.get_text_with_window.focus_miss",
                extra={"ref": ref},
            )
            return None

        focus_data = focus_result.get("data") or {}
        prev_task = asyncio.create_task(
            generate_neighbors(
                ref,
                resolved_window,
                direction="prev",
                sefaria_service=self._sefaria_service,
                index_service=self._index_service,
            )
        )
        next_task = asyncio.create_task(
            generate_neighbors(
                ref,
                resolved_window,
                direction="next",
                sefaria_service=self._sefaria_service,
                index_service=self._index_service,
            )
        )

        prev_segments, next_segments = await asyncio.gather(prev_task, next_task)

        raw_segments: List[Dict[str, Any]] = []
        raw_segments.extend(prev_segments or [])
        raw_segments.append(focus_data)
        raw_segments.extend(next_segments or [])

        prev_non_empty = [segment for segment in (prev_segments or []) if segment]
        next_non_empty = [segment for segment in (next_segments or []) if segment]
        filtered_segments = prev_non_empty + ([focus_data] if focus_data else []) + next_non_empty
        formatted_segments = self._format_window_segments(filtered_segments)
        if not formatted_segments:
            self._logger.warning(
                "study.facade.get_text_with_window.empty",
                extra={"ref": ref},
            )
            return None

        focus_index = len(prev_non_empty)

        # For Tanakh-style references, attempt to hydrate the full chapter so the client
        # can render a continuous stream rather than a narrow window.
        parsed_focus = parse_ref(ref)
        if parsed_focus.collection == "bible" and parsed_focus.chapter is not None:
            try:
                chapter_length = await _ensure_chapter_length(
                    parsed_focus.book,
                    parsed_focus.chapter,
                    {},
                    self._sefaria_service,
                )
            except Exception:  # pragma: no cover - cache retrieval only
                chapter_length = None

            if chapter_length and chapter_length > 0:
                range_ref = (
                    f"{parsed_focus.book} {parsed_focus.chapter}:1-"
                    f"{parsed_focus.book} {parsed_focus.chapter}:{chapter_length}"
                )
                try:
                    full_payload = await build_full_daily_text(
                        range_ref,
                        self._sefaria_service,
                        self._index_service,
                        session_id=None,
                        redis_client=None,
                    )
                    if full_payload and full_payload.get("segments"):
                        formatted_segments = full_payload["segments"]
                        target_ref = focus_data.get("ref", ref)
                        for idx, segment in enumerate(formatted_segments):
                            if segment.get("ref") == target_ref:
                                focus_index = idx
                                break
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._logger.debug(
                        "study.facade.get_text_with_window.chapter_fallback_failed",
                        extra={
                            "ref": ref,
                            "range_ref": range_ref,
                            "error": str(exc),
                        },
                    )

        duration_ms = (perf_counter() - start_ts) * 1000
        log_window_built(ref, len(formatted_segments), resolved_window, duration_ms)

        return {
            "segments": formatted_segments,
            "focusIndex": min(focus_index, max(len(formatted_segments) - 1, 0)),
            "ref": focus_data.get("ref", ref),
            "he_ref": focus_data.get("heRef"),
        }

    def _resolve_window_size(self, requested: Optional[int]) -> int:
        cfg = self._study_config.window
        size = cfg.size_default if requested is None else requested
        return max(cfg.size_min, min(cfg.size_max, size))

    def _format_window_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total = len(segments)
        formatted: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments):
            formatted.append(self._format_single_segment(segment, index, total))
        return formatted

    @staticmethod
    def _format_single_segment(segment: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
        ref = segment.get("ref") or segment.get("heRef") or ""

        he_raw = segment.get("he_text")
        if he_raw is None:
            he_raw = segment.get("he")
        he_text = clean_html(extract_hebrew_only(he_raw))

        en_raw = (
            segment.get("en_text")
            if segment.get("en_text") is not None
            else segment.get("text")
        )
        en_text = StudyService._coerce_text(en_raw)

        text_value = he_text or en_text

        base_metadata = segment.get("metadata") or {}
        metadata: Dict[str, Any] = {
            "title": segment.get("title") or base_metadata.get("title", ""),
            "indexTitle": segment.get("indexTitle") or base_metadata.get("indexTitle", ""),
            "heRef": segment.get("heRef") or base_metadata.get("heRef", ""),
        }
        for key in ("chapter", "verse", "amud", "page"):
            if key in segment and segment[key] is not None:
                metadata[key] = segment[key]
            elif key in base_metadata:
                metadata[key] = base_metadata[key]

        position = index / (total - 1) if total > 1 else 0.5

        return {
            "ref": ref,
            "text": text_value,
            "heText": he_text or text_value,
            "enText": en_text,
            "position": float(position),
            "metadata": metadata,
        }

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, list):
            for item in value:
                if item:
                    return clean_html(str(item))
            return ""
        if value is None:
            return ""
        return clean_html(str(value))

    @staticmethod
    def _normalize_ref_for_compare(ref: str) -> str:
        if not ref:
            return ""
        return "".join(ref.lower().split())

    def _filter_bookshelf_payload(
        self,
        payload: Optional[Dict[str, Any]],
        ref: str,
    ) -> Optional[Dict[str, Any]]:
        if not payload or not isinstance(payload, dict):
            return payload

        items = payload.get("items")
        if not items:
            return payload

        target = self._normalize_ref_for_compare(ref)

        def _get_attr(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _matches(item: Any) -> bool:
            anchors = (
                _get_attr(item, "anchorRefExpanded")
                or _get_attr(item, "anchorRef")
                or _get_attr(item, "ref")
            )
            if isinstance(anchors, list):
                return any(
                    isinstance(anchor, str)
                    and self._normalize_ref_for_compare(anchor) == target
                    for anchor in anchors
                )
            if isinstance(anchors, str):
                return self._normalize_ref_for_compare(anchors) == target
            return False

        filtered_items = [item for item in items if _matches(item)]
        if not filtered_items:
            return payload

        counts: Dict[str, int] = {}
        for item in filtered_items:
            category = _get_attr(item, "category") or "Unknown"
            counts[category] = counts.get(category, 0) + 1

        filtered_payload = dict(payload)
        filtered_payload["items"] = filtered_items
        filtered_payload["counts"] = counts
        return filtered_payload

    # ------------------------------------------------------------------
    # Daily study flows
    # ------------------------------------------------------------------
    async def get_full_daily_text(
        self, ref: str, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the full daily text payload (legacy-compatible)."""

        payload = await build_full_daily_text(
            ref,
            self._sefaria_service,
            self._index_service,
            session_id=session_id,
            redis_client=self._redis,
        )
        if not payload:
            self._logger.warning(
                "study.facade.get_full_daily_text.empty",
                extra={"ref": ref, "session_id": session_id},
            )
        return payload

    async def load_daily_initial(
        self,
        ref: str,
        session_id: str,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Expose daily loader initialisation for callers that opt-in."""

        return await self._daily_loader.load_initial(
            ref=ref,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
        )

    async def schedule_daily_background(
        self,
        ref: str,
        *,
        session_id: str,
        start_verse: int,
        end_verse: int,
        book_chapter: str,
        already_loaded: int,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        await self._daily_loader.load_background(
            ref=ref,
            session_id=session_id,
            start_verse=start_verse,
            end_verse=end_verse,
            book_chapter=book_chapter,
            already_loaded=already_loaded,
            ttl_seconds=ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Bookshelf flows
    # ------------------------------------------------------------------
    async def get_bookshelf_for(
        self,
        ref: str,
        limit: int = 40,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        bookshelf = await self._bookshelf_service.get_for(
            ref,
            limit=limit,
            categories=categories,
        )
        return {"counts": bookshelf.counts, "items": bookshelf.items}

    # ------------------------------------------------------------------
    # Prompting (placeholder for Phase 6)
    # ------------------------------------------------------------------
    async def build_prompt_payload(
        self,
        *,
        ref: str,
        mode: str,
        system_prompt: str,
        stm_context: str = "",
        study_segments: Iterable[str] = (),
        extra_segments: Iterable[str] = (),
    ) -> Dict[str, Any]:
        budget = build_budget(self._study_config.prompt_budget)
        parts = PromptParts(
            system=system_prompt,
            stm=stm_context,
            study=study_segments,
            extras=extra_segments,
        )
        token_counter = self._token_counters.get(None)
        prompt = assemble_prompt(parts, budget, token_counter=token_counter)
        return prompt
