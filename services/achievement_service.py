from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def is_rambam_ref(ref: Optional[str]) -> bool:
    if not ref:
        return False
    ref_l = ref.lower()
    return "mishneh torah" in ref_l or "rambam" in ref_l


def is_daf_yomi_ref(ref: Optional[str]) -> bool:
    if not ref:
        return False
    ref_l = ref.lower()
    if "daf yomi" in ref_l:
        return True
    # простая эвристика: строки вида "Berakhot 2a" / "Shabbat 14b"
    return any(suffix in ref_l for suffix in ["a", "b"]) and any(char.isdigit() for char in ref_l)


@dataclass
class AchievementProgress:
    discipline: int
    lexicon: int
    rambam: int
    daf: int

    def to_payload(self) -> dict:
        return {
            "discipline": self.discipline,
            "lexicon": self.lexicon,
            "rambam": self.rambam,
            "daf": self.daf,
        }


LEVELS = {
    "discipline": [
        ("bronze", 7),
        ("silver", 30),
        ("gold", 100),
        ("platinum", 365),
    ],
    "lexicon": [
        ("bronze", 10),
        ("silver", 50),
        ("gold", 150),
        ("platinum", 500),
    ],
    "rambam": [
        ("bronze", 10),
        ("silver", 40),
        ("gold", 120),
        ("platinum", 300),
    ],
    "daf": [
        ("bronze", 7),
        ("silver", 30),
        ("gold", 100),
        ("platinum", 365),
    ],
}


class AchievementService:
    def __init__(self, session_factory: Optional[async_sessionmaker[AsyncSession]]):
        self.session_factory = session_factory

    async def _ensure_row(self, user_id: str):
        if not self.session_factory:
            return
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO user_achievement_stats (user_id, discipline_total, lexicon_total, rambam_refs, daf_refs)
                    VALUES (:user_id, 0, 0, 0, 0)
                    ON CONFLICT (user_id) DO NOTHING
                    """
                ),
                {"user_id": user_id},
            )
            await session.commit()

    async def update_with_event(self, user_id: str, payload: dict):
        if not self.session_factory:
            return
        source = payload.get("source")
        verb = payload.get("verb")
        ref = payload.get("ref") or ""
        category = (payload.get("category") or payload.get("title") or "").lower()
        if not source:
            return
        await self._ensure_row(user_id)
        async with self.session_factory() as session:
            inc = {"discipline": 0, "lexicon": 0, "rambam": 0, "daf": 0}
            if source == "daily" and verb == "complete":
                inc["discipline"] = 1
                if "daf yomi" in category:
                    inc["daf"] = 1
                if "rambam" in category:
                    inc["rambam"] = 1
            if source == "lexicon":
                inc["lexicon"] = 1
            if source in {"focus", "workbench"} and is_rambam_ref(ref):
                inc["rambam"] = 1
            if source in {"focus", "workbench", "daily"} and is_daf_yomi_ref(ref):
                inc["daf"] = 1
            if all(v == 0 for v in inc.values()):
                return
            await session.execute(
                text(
                    """
                    UPDATE user_achievement_stats
                    SET discipline_total = discipline_total + :d,
                        lexicon_total = lexicon_total + :l,
                        rambam_refs = rambam_refs + :r,
                        daf_refs = daf_refs + :daf
                    WHERE user_id = :user_id
                    """
                ),
                {
                    "d": inc["discipline"],
                    "l": inc["lexicon"],
                    "r": inc["rambam"],
                    "daf": inc["daf"],
                    "user_id": user_id,
                },
            )
            await session.commit()

    async def get_progress(self, user_id: str) -> AchievementProgress:
        if not self.session_factory:
            return AchievementProgress(0, 0, 0, 0)
        await self._ensure_row(user_id)
        async with self.session_factory() as session:
            res = await session.execute(
                text(
                    """
                    SELECT discipline_total, lexicon_total, rambam_refs, daf_refs
                    FROM user_achievement_stats
                    WHERE user_id = :user_id
                    """
                ),
                {"user_id": user_id},
            )
            row = res.first()
            if not row:
                return AchievementProgress(0, 0, 0, 0)
            return AchievementProgress(
                discipline=int(row.discipline_total or 0),
                lexicon=int(row.lexicon_total or 0),
                rambam=int(row.rambam_refs or 0),
                daf=int(row.daf_refs or 0),
            )

    def compute_levels(self, progress: AchievementProgress) -> dict:
        result = {}
        for key, thresholds in LEVELS.items():
            value = getattr(progress, key)
            current = None
            next_needed = None
            for level_name, threshold in thresholds:
                if value >= threshold:
                    current = level_name
                elif next_needed is None:
                    next_needed = threshold - value
                    break
            result[key] = {"value": value, "level": current or "none", "to_next": next_needed}
        return result
