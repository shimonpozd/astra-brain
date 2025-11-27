from __future__ import annotations

import hashlib
import math
import json
from datetime import datetime as dt
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from brain_service.services.achievement_service import AchievementService


@dataclass
class XpProfile:
    xp_total: int
    level: int
    xp_in_level: int
    xp_to_next: int
    last_level_up_at: Optional[int] = None

    def to_payload(self) -> dict:
        return {
            "xp_total": self.xp_total,
            "level": self.level,
            "xp_in_level": self.xp_in_level,
            "xp_to_next": self.xp_to_next,
            "last_level_up_at": self.last_level_up_at,
        }


class XpService:
    BASE_LEVEL_XP = 300
    LEVEL_GROWTH = 1.18
    EVENT_TTL_SECONDS = 60 * 60 * 24
    HISTORY_LIMIT = 100

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        achievement_service: Optional[AchievementService] = None,
    ):
        self.redis = redis_client
        self.session_factory = session_factory
        self.achievement_service = achievement_service

    def _key_total(self, user_id: str) -> str:
        return f"xp:user:{user_id}:total"

    def _key_event(self, user_id: str, event_id: str) -> str:
        return f"xp:event:{user_id}:{event_id}"

    def _key_history(self, user_id: str) -> str:
        return f"xp:user:{user_id}:history"

    @staticmethod
    def calculate_level(xp_total: int) -> XpProfile:
        level = 1
        remaining = max(0, xp_total)
        xp_for_level = XpService.BASE_LEVEL_XP

        while remaining >= xp_for_level:
            remaining -= xp_for_level
            level += 1
            xp_for_level = round(XpService.BASE_LEVEL_XP * (XpService.LEVEL_GROWTH ** (level - 1)))

        xp_to_next = max(0, xp_for_level - remaining)
        return XpProfile(
            xp_total=xp_total,
            level=level,
            xp_in_level=remaining,
            xp_to_next=xp_to_next,
        )

    async def _ensure_row(self, user_id: str) -> None:
        if not self.session_factory:
            return
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO user_xp (user_id, xp_total, level, xp_in_level, xp_to_next, last_level_up_at)
                    VALUES (:user_id, 0, 1, 0, :xp_next, NULL)
                    ON CONFLICT (user_id) DO NOTHING
                    """
                ),
                {"user_id": user_id, "xp_next": self.BASE_LEVEL_XP},
            )
            await session.commit()

    async def get_profile(self, user_id: str) -> XpProfile:
        # Prefer DB if available
        if self.session_factory:
            await self._ensure_row(user_id)
            async with self.session_factory() as session:
                res = await session.execute(
                    text(
                        """
                        SELECT xp_total, level, xp_in_level, xp_to_next, last_level_up_at
                        FROM user_xp WHERE user_id = :user_id
                        """
                    ),
                    {"user_id": user_id},
                )
                row = res.first()
                if row:
                    return XpProfile(
                        xp_total=int(row.xp_total or 0),
                        level=int(row.level or 1),
                        xp_in_level=int(row.xp_in_level or 0),
                        xp_to_next=int(row.xp_to_next or self.BASE_LEVEL_XP),
                        last_level_up_at=int(row.last_level_up_at.timestamp() * 1000) if row.last_level_up_at else None,
                    )
        # Fallback to Redis only
        if self.redis:
            raw = await self.redis.get(self._key_total(user_id))
            total = int(raw or 0)
            return self.calculate_level(total)
        return self.calculate_level(0)

    def _build_event_id(self, payload: dict) -> str:
        raw = (
            f"{payload.get('source','')}|{payload.get('verb','')}|"
            f"{payload.get('session_id','')}|{payload.get('ref','')}|"
            f"{payload.get('title','')}|{payload.get('bucket','')}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _compute_amount(self, payload: dict) -> int:
        source = payload.get("source")
        verb = payload.get("verb")
        chars = int(payload.get("chars") or 0)
        duration_ms = int(payload.get("duration_ms") or 0)

        if source == "focus":
            if verb == "switch":
                return 2
            if verb == "translate":
                return min(8, 3 + math.ceil(chars / 350)) if chars else 3
            if verb == "listen":
                return min(8, max(2, math.ceil(duration_ms / 30000) * 2)) if duration_ms else 2
        if source == "workbench":
            if verb == "load":
                return 4
            if verb == "translate":
                return min(12, 4 + math.ceil(chars / 320)) if chars else 4
            if verb == "listen":
                return min(9, max(3, math.ceil(duration_ms / 20000) * 3)) if duration_ms else 3
        if source == "chat":
            if verb in {"ask", "reply"}:
                return min(25, 3 + math.ceil(chars / 220)) if chars else 3
            if verb == "listen":
                base = min(25, 3 + math.ceil(chars / 220)) if chars else 3
                return max(1, math.ceil(base * 0.5))
        if source == "lexicon":
            return 25
        if source == "daily":
            # bonus streak не считаем здесь — бэкенд может дописать в payload.amount
            return 60
        return 0

    async def record_event(self, user_id: str, payload: dict) -> XpProfile:
        event_id = payload.get("event_id") or self._build_event_id(payload)
        amount = payload.get("amount")
        try:
            amount_int = int(amount) if amount is not None else None
        except (TypeError, ValueError):
            amount_int = None
        if amount_int is None:
            amount_int = self._compute_amount(payload)

        # idempotency (only if redis available)
        was_new = True
        if self.redis:
            was_new = await self.redis.set(self._key_event(user_id, event_id), "1", nx=True, ex=self.EVENT_TTL_SECONDS)
            if not was_new:
                return await self.get_profile(user_id)

        if amount_int <= 0:
            return await self.get_profile(user_id)

        ts_ms = int(payload.get("ts") or int(math.floor(dt.utcnow().timestamp() * 1000)))

        # базовый профиль до начисления
        current_profile = await self.get_profile(user_id)
        new_total = current_profile.xp_total + amount_int
        new_profile = self.calculate_level(new_total)

        # Update Redis (cache/history)
        if self.redis:
            pipe = self.redis.pipeline()
            pipe.incrby(self._key_total(user_id), amount_int)
            pipe.lpush(
                self._key_history(user_id),
                json.dumps(
                    {
                        "source": payload.get("source"),
                        "verb": payload.get("verb"),
                        "amount": amount_int,
                        "ref": payload.get("ref"),
                        "title": payload.get("title"),
                        "ts": ts_ms,
                    }
                ),
            )
            pipe.ltrim(self._key_history(user_id), 0, self.HISTORY_LIMIT - 1)
            await pipe.execute()

        if self.achievement_service:
            try:
                await self.achievement_service.update_with_event(user_id, payload)
            except Exception:
                pass

        # Persist in DB if available
        if self.session_factory:
            try:
                await self._ensure_row(user_id)
                async with self.session_factory() as session:
                    await session.execute(
                        text(
                            """
                            INSERT INTO user_xp (user_id, xp_total, level, xp_in_level, xp_to_next, last_level_up_at)
                            VALUES (:user_id, :xp_total, :level, :xp_in_level, :xp_to_next, :last_level_up_at)
                            ON CONFLICT (user_id) DO UPDATE
                            SET xp_total = :xp_total,
                                level = :level,
                                xp_in_level = :xp_in_level,
                                xp_to_next = :xp_to_next,
                                last_level_up_at = CASE
                                    WHEN :level > user_xp.level THEN :last_level_up_at
                                    ELSE user_xp.last_level_up_at
                                END
                            """
                        ),
                        {
                            "user_id": user_id,
                            "xp_total": new_profile.xp_total,
                            "level": new_profile.level,
                            "xp_in_level": new_profile.xp_in_level,
                            "xp_to_next": new_profile.xp_to_next,
                            "last_level_up_at": dt.utcfromtimestamp(ts_ms / 1000),
                        },
                    )
                    await session.commit()
            except Exception:
                # тихо логируем, но не срываем выдачу профиля
                pass

        return new_profile

    async def list_history(self, user_id: str, limit: int = 50) -> list[dict]:
        if not self.redis:
            return []
        raw = await self.redis.lrange(self._key_history(user_id), 0, max(0, limit - 1))
        events: list[dict] = []
        for item in raw:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    events.append(parsed)
            except Exception:
                continue
        return events
