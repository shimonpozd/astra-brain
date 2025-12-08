from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from brain_service.models.talmudic_concept import TalmudicConcept
from brain_service.utils.text_processing import build_search_patterns

logger = logging.getLogger(__name__)


def _safe_patterns(raw: Iterable[str]) -> list[str]:
    patterns: list[str] = []
    for item in raw or []:
        if isinstance(item, str) and item.strip():
            patterns.append(item.strip())
    # Deduplicate while preserving order
    uniq: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        if pat not in seen:
            seen.add(pat)
            uniq.append(pat)
    return uniq


class TalmudicConceptService:
    def __init__(self, session_factory: async_sessionmaker):
        self._session_factory = session_factory

    async def _to_dict(self, obj: TalmudicConcept) -> Dict[str, Any]:
        return {
            "slug": obj.slug,
            "term_he": obj.term_he,
            "search_patterns": obj.search_patterns or [],
            "short_summary_html": obj.short_summary_html,
            "full_article_html": obj.full_article_html,
            "status": obj.status,
            "created_at": obj.created_at.isoformat() if getattr(obj, "created_at", None) else None,
            "updated_at": obj.updated_at.isoformat() if getattr(obj, "updated_at", None) else None,
            "generated_at": obj.generated_at.isoformat() if getattr(obj, "generated_at", None) else None,
        }

    def _ensure_patterns(self, term_he: str, patterns: Optional[List[str]] = None, *, synonyms: Iterable[str] | None = None) -> list[str]:
        base_patterns = _safe_patterns(patterns or [])
        generated = build_search_patterns(term_he, *(synonyms or []))
        for pat in generated:
            if pat not in base_patterns:
                base_patterns.append(pat)
        return base_patterns

    async def list_published(self) -> list[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(TalmudicConcept).where(TalmudicConcept.status == "published").order_by(TalmudicConcept.slug)
            )
            items = [await self._to_dict(obj) for obj in result]
            return items

    async def list_all(self) -> list[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.scalars(select(TalmudicConcept).order_by(TalmudicConcept.slug))
            return [await self._to_dict(obj) for obj in result]

    async def get(self, slug: str) -> Optional[Dict[str, Any]]:
        if not slug:
            return None
        async with self._session_factory() as session:
            obj = await session.get(TalmudicConcept, slug)
            if obj:
                return await self._to_dict(obj)
        return None

    async def upsert(
        self,
        *,
        slug: str,
        term_he: str,
        search_patterns: Optional[List[str]] = None,
        short_summary_html: Optional[str] = None,
        full_article_html: Optional[str] = None,
        status: str = "draft",
        synonyms: Iterable[str] | None = None,
        generated_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        patterns = self._ensure_patterns(term_he, search_patterns, synonyms=synonyms)
        if not patterns:
            patterns = build_search_patterns(term_he)

        async with self._session_factory() as session:
            async with session.begin():
                obj = await session.get(TalmudicConcept, slug)
                if not obj:
                    obj = TalmudicConcept(slug=slug, term_he=term_he)
                    session.add(obj)
                obj.term_he = term_he
                obj.search_patterns = patterns
                obj.short_summary_html = short_summary_html
                obj.full_article_html = full_article_html
                obj.status = status or "draft"
                obj.generated_at = generated_at
                await session.flush()
                await session.refresh(obj)
                return await self._to_dict(obj)

    async def delete(self, slug: str) -> None:
        if not slug:
            return
        async with self._session_factory() as session:
            async with session.begin():
                obj = await session.get(TalmudicConcept, slug)
                if obj:
                    await session.delete(obj)
