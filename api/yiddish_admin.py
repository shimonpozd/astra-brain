from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, or_, select, update

from brain_service.core.database import session_scope
from brain_service.core.dependencies import require_admin_user
from brain_service.models.db import User
from brain_service.models.yiddish import YiddishWordCard

router = APIRouter()


def _get_session_factory(request: Request):
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database session factory unavailable")
    return session_factory


@router.get("/yiddish/wordcards")
async def list_yiddish_wordcards(
    request: Request,
    _: User = Depends(require_admin_user),
    ui_lang: str = "ru",
    prefix: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    version: Optional[int] = Query(default=1, ge=1),
):
    session_factory = _get_session_factory(request)
    filters = [
        YiddishWordCard.ui_lang == ui_lang,
        YiddishWordCard.source == "wiktionary",
    ]
    if version is not None:
        filters.append(YiddishWordCard.version == version)
    if prefix:
        filters.append(YiddishWordCard.lemma.ilike(f"{prefix}%"))
    if q:
        q_like = f"%{q}%"
        filters.append(or_(YiddishWordCard.lemma.ilike(q_like), YiddishWordCard.word_surface.ilike(q_like)))

    async with session_scope(session_factory) as session:
        total = await session.scalar(select(func.count()).select_from(YiddishWordCard).where(*filters))
        result = await session.execute(
            select(YiddishWordCard)
            .where(*filters)
            .order_by(YiddishWordCard.lemma.asc())
            .limit(limit)
            .offset(offset)
        )
        items: List[Dict[str, Any]] = []
        for row in result.scalars().all():
            data = row.data or {}
            popup = data.get("popup") or {}
            items.append(
                {
                    "lemma": row.lemma,
                    "word_surface": row.word_surface,
                    "pos_default": row.pos_default,
                    "ui_lang": row.ui_lang,
                    "version": row.version,
                    "retrieved_at": row.retrieved_at.isoformat() if row.retrieved_at else None,
                    "translit_ru": data.get("translit_ru"),
                    "glosses": popup.get("gloss_ru_short_list") or [],
                }
            )

    return {"ok": True, "total": total or 0, "items": items}


@router.get("/yiddish/wordcards/{lemma}")
async def get_yiddish_wordcard(
    lemma: str,
    request: Request,
    _: User = Depends(require_admin_user),
    ui_lang: str = "ru",
    version: Optional[int] = Query(default=1, ge=1),
):
    session_factory = _get_session_factory(request)
    async with session_scope(session_factory) as session:
        result = await session.execute(
            select(YiddishWordCard).where(
                YiddishWordCard.lemma == lemma,
                YiddishWordCard.ui_lang == ui_lang,
                YiddishWordCard.source == "wiktionary",
                YiddishWordCard.version == version,
            )
        )
        card = result.scalar_one_or_none()
        if not card or not card.data:
            raise HTTPException(status_code=404, detail="WordCard not found")
        return {"ok": True, "data": card.data, "evidence": card.evidence}


@router.patch("/yiddish/wordcards/{lemma}")
async def update_yiddish_wordcard(
    lemma: str,
    payload: Dict[str, Any],
    request: Request,
    _: User = Depends(require_admin_user),
    ui_lang: str = "ru",
    version: Optional[int] = Query(default=1, ge=1),
):
    session_factory = _get_session_factory(request)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Payload must include data object")

    data["lemma"] = lemma
    data.setdefault("ui_lang", ui_lang)
    data.setdefault("lang", "yi")
    data.setdefault("schema", "astra.yiddish.wordcard.v1")

    pos_default = data.get("pos_default")
    word_surface = data.get("word_surface")
    evidence = payload.get("evidence")

    async with session_scope(session_factory) as session:
        result = await session.execute(
            select(YiddishWordCard).where(
                YiddishWordCard.lemma == lemma,
                YiddishWordCard.ui_lang == ui_lang,
                YiddishWordCard.source == "wiktionary",
                YiddishWordCard.version == version,
            )
        )
        existing = result.scalar_one_or_none()
        if not existing:
            raise HTTPException(status_code=404, detail="WordCard not found")

        await session.execute(
            update(YiddishWordCard)
            .where(YiddishWordCard.id == existing.id)
            .values(
                data=data,
                evidence=evidence if isinstance(evidence, dict) else existing.evidence,
                pos_default=pos_default,
                word_surface=word_surface,
                retrieved_at=datetime.now(timezone.utc),
            )
        )

    return {"ok": True, "data": data}
